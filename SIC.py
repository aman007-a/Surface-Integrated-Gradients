#!/usr/bin/env python3

import torch
import numpy as np
import os
import io
import time
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scipy import interpolate
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple
import core.config.configuration as cnfg
from core.vae import VAE
from core import classifier as clf
from core.classifier import ResClassifier, VGG16Classifier, InceptionClassifier

# -----------------------------
# Device & Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# OFFICIAL PAIR-CODE/SALIENCY IMPLEMENTATION
# -----------------------------

def create_blurred_image(full_img: np.ndarray, pixel_mask: np.ndarray,
    method: str = 'linear') -> np.ndarray:
  """Creates a blurred (interpolated) image - Official PAIR-code implementation"""
  data_type = full_img.dtype
  has_color_channel = full_img.ndim > 2
  if not has_color_channel:
    full_img = np.expand_dims(full_img, axis=2)
  channels = full_img.shape[2]

  # Always include corners
  pixel_mask = pixel_mask.copy()
  height = pixel_mask.shape[0]
  width = pixel_mask.shape[1]
  pixel_mask[
    [0, 0, height - 1, height - 1], [0, width - 1, 0, width - 1]] = True

  mean_color = np.mean(full_img, axis=(0, 1))

  # If the mask consists of all pixels set to True then return the original image
  if np.all(pixel_mask):
    return full_img

  blurred_img = full_img * np.expand_dims(pixel_mask, axis=2).astype(np.float32)

  # Interpolate the unmasked values of the image pixels
  for channel in range(channels):
    data_points = np.argwhere(pixel_mask > 0)
    data_values = full_img[:, :, channel][tuple(data_points.T)]
    unknown_points = np.argwhere(pixel_mask == 0)
    interpolated_values = interpolate.griddata(np.array(data_points),
                                               np.array(data_values),
                                               np.array(unknown_points),
                                               method=method,
                                               fill_value=mean_color[channel])
    blurred_img[:, :, channel][tuple(unknown_points.T)] = interpolated_values

  if not has_color_channel:
    blurred_img = blurred_img[:, :, 0]

  if issubclass(data_type.type, np.integer):
    blurred_img = np.round(blurred_img)

  return blurred_img.astype(data_type)


def generate_random_mask(image_height: int, image_width: int,
    fraction=0.01) -> np.ndarray:
  """Generates a random pixel mask - Official implementation"""
  mask = np.zeros(shape=[image_height, image_width], dtype=bool)
  size = mask.size
  indices = np.random.choice(size, replace=False, size=int(size * fraction))
  mask[np.unravel_index(indices, mask.shape)] = True
  return mask


def estimate_image_entropy(image: np.ndarray) -> float:
  """Estimates the amount of information in a given image using WebP compression"""
  buffer = io.BytesIO()
  pil_image = Image.fromarray(image)
  pil_image.save(buffer, format='webp', lossless=True, quality=100)
  buffer.seek(0, os.SEEK_END)
  length = buffer.tell()
  buffer.close()
  return length


class ComputePicMetricError(Exception):
  """An error that can be raised by the compute_pic_metric(...) method."""
  pass


class PicMetricResult(NamedTuple):
  """Holds results of compute_pic_metric(...) method."""
  curve_x: Sequence[float]
  curve_y: Sequence[float]
  blurred_images: Sequence[np.ndarray]
  predictions: Sequence[float]
  thresholds: Sequence[float]
  auc: float


def compute_pic_metric(
    img: np.ndarray,
    saliency_map: np.ndarray,
    random_mask: np.ndarray,
    pred_func: Callable[[np.ndarray], Sequence[float]],
    saliency_thresholds: Sequence[float],
    min_pred_value: float = 0.05,
    keep_monotonous: bool = True,
    num_data_points: int = 1000
) -> PicMetricResult:
  """Computes Performance Information Curve for a single image - Official implementation"""
  if img.dtype.type != np.uint8:
    raise ValueError('The `img` array that holds the input image should be of'
                     ' type uint8. The actual type is {}.'.format(img.dtype))
  blurred_images = []
  predictions = []
  entropy_pred_tuples = []

  # Estimate entropy of the original image
  original_img_entropy = estimate_image_entropy(img)

  # Estimate entropy of the completely blurred image
  fully_blurred_img = create_blurred_image(full_img=img, pixel_mask=random_mask)
  fully_blurred_img_entropy = estimate_image_entropy(fully_blurred_img)

  # Compute model prediction for the original image
  original_img_pred = pred_func(img[np.newaxis, ...])[0]

  if original_img_pred < min_pred_value:
    message = ('The model prediction score on the original image is lower than'
               ' `min_pred_value`. Skip this image or decrease the'
               ' value of `min_pred_value` argument. min_pred_value'
               ' = {}, the image prediction'
               ' = {}.'.format(min_pred_value, original_img_pred))
    raise ComputePicMetricError(message)

  # Compute model prediction for the completely blurred image
  fully_blurred_img_pred = pred_func(fully_blurred_img[np.newaxis, ...])[0]

  blurred_images.append(fully_blurred_img)
  predictions.append(fully_blurred_img_pred)

  # Validation checks
  if fully_blurred_img_entropy >= original_img_entropy:
    message = (
        'The entropy in the completely blurred image is not lower than'
        ' the entropy in the original image. Catch the error and exclude this'
        ' image from evaluation. Blurred entropy: {}, original'
        ' entropy {}'.format(fully_blurred_img_entropy, original_img_entropy))
    raise ComputePicMetricError(message)

  if fully_blurred_img_pred >= original_img_pred:
    message = (
        'The model prediction score on the completely blurred image is not'
        ' lower than the score on the original image. Catch the error and'
        ' exclude this image from the evaluation. Blurred score: {}, original'
        ' score {}'.format(fully_blurred_img_pred, original_img_pred))
    raise ComputePicMetricError(message)

  # Iterate through saliency thresholds
  max_normalized_pred = 0.0
  for threshold in saliency_thresholds:
    quantile = np.quantile(saliency_map, 1 - threshold)
    pixel_mask = saliency_map >= quantile
    pixel_mask = np.logical_or(pixel_mask, random_mask)
    blurred_image = create_blurred_image(full_img=img, pixel_mask=pixel_mask)
    entropy = estimate_image_entropy(blurred_image)
    pred = pred_func(blurred_image[np.newaxis, ...])[0]
    # Normalize the values to [0, 1] interval
    normalized_entropy = (entropy - fully_blurred_img_entropy) / (
        original_img_entropy - fully_blurred_img_entropy)
    normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)
    normalized_pred = (pred - fully_blurred_img_pred) / (
        original_img_pred - fully_blurred_img_pred)
    normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
    max_normalized_pred = max(max_normalized_pred, normalized_pred)

    if keep_monotonous:
      entropy_pred_tuples.append((normalized_entropy, max_normalized_pred))
    else:
      entropy_pred_tuples.append((normalized_entropy, normalized_pred))

    blurred_images.append(blurred_image)
    predictions.append(pred)

  # Interpolate the PIC curve
  entropy_pred_tuples.append((0.0, 0.0))
  entropy_pred_tuples.append((1.0, 1.0))

  entropy_data, pred_data = zip(*entropy_pred_tuples)
  interp_func = interpolate.interp1d(x=entropy_data, y=pred_data)

  curve_x = np.linspace(start=0.0, stop=1.0, num=num_data_points, endpoint=False)
  curve_y = np.asarray([interp_func(x) for x in curve_x])

  curve_x = np.append(curve_x, 1.0)
  curve_y = np.append(curve_y, 1.0)

  auc = np.trapz(curve_y, curve_x)

  blurred_images.append(img)
  predictions.append(original_img_pred)

  thresholds = [0.0] + list(saliency_thresholds) + [1.0]

  return PicMetricResult(curve_x=curve_x, curve_y=curve_y,
                         blurred_images=blurred_images,
                         predictions=predictions, thresholds=thresholds,
                         auc=auc)


class AggregateMetricResult(NamedTuple):
  """Holds results of aggregate_individual_pic_results(...) method."""
  curve_x: Sequence[float]
  curve_y: Sequence[float]
  auc: float


def aggregate_individual_pic_results(
    compute_pic_metrics_results: List[PicMetricResult],
    method: str = 'median') -> AggregateMetricResult:
  """Aggregates PIC metrics - Official implementation"""
  if not compute_pic_metrics_results:
    raise ValueError('The list of results should have at least one element.')

  curve_ys = [r.curve_y for r in compute_pic_metrics_results]
  curve_ys = np.asarray(curve_ys)

  curve_xs = [r.curve_x for r in compute_pic_metrics_results]
  curve_xs = np.asarray(curve_xs)
  _, counts = np.unique(curve_xs, axis=1, return_counts=True)
  if not np.all(counts == 1):
    raise ValueError('Individual results have different x-axis data points.')

  if method == 'mean':
    aggr_curve_y = np.mean(curve_ys, axis=0)
  elif method == 'median':
    aggr_curve_y = np.median(curve_ys, axis=0)
  else:
    raise ValueError('Unknown method {}.'.format(method))

  auc = np.trapz(aggr_curve_y, curve_xs[0])

  return AggregateMetricResult(curve_x=curve_xs[0], curve_y=aggr_curve_y, auc=auc)

# -----------------------------
# Model Integration
# -----------------------------

def load_models():
    """Load VAE and classifier models"""
    # Load VAE model
    vae_net = VAE(
        channel_in=3,
        ch=cnfg.channels,
        blocks=cnfg.blocks,
        latent_channels=cnfg.latent_channels
    ).to(device)

    # Load VAE checkpoint
    vae_checkpoint = torch.load(cnfg.save_dir + "/Models/" + cnfg.model_name + ".pt", map_location="cpu")
    vae_net.load_state_dict(vae_checkpoint['model_state_dict'])
    vae_net.eval()

    # Load classifier
    backbone = cnfg.backbone_type
    if backbone == "resnet":
        classifier = ResClassifier()
    elif backbone == "vgg16":
        classifier = VGG16Classifier()
    else:
        classifier = InceptionClassifier()

    # Load classifier weights
    ckpt_path = os.path.join("Models", f"{cnfg.model_name}_classifier_{backbone}.pt")
    state_dict = torch.load(ckpt_path, map_location=device)["model_state_dict"]
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()

    return vae_net, classifier

def load_image_by_index(img_index, img_size=192):
    """Load specific image by index from Oxford Pets dataset"""
    image_dir = "../datasets/oxford_pets/images"
    image_files = sorted(os.listdir(image_dir))

    if img_index >= len(image_files):
        raise IndexError(f"Image index {img_index} out of range. Dataset has {len(image_files)} images.")

    img_name = image_files[img_index]
    img_path = os.path.join(image_dir, img_name)

    # Load and transform image
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)

    return img_tensor

def create_pred_func(vae_model, classifier_model):
    """Create prediction function that returns softmax scores for SIC"""
    def pred_func(images_batch):
        """Returns softmax scores for the predicted class"""
        # Convert numpy to torch tensor
        if isinstance(images_batch, np.ndarray):
            if images_batch.dtype == np.uint8:
                images_tensor = torch.from_numpy(images_batch).float() / 255.0
            else:
                images_tensor = torch.from_numpy(images_batch).float()
        else:
            images_tensor = images_batch

        # Add batch dimension if needed
        if images_tensor.dim() == 3:
            images_tensor = images_tensor.unsqueeze(0)

        # Ensure correct shape [B, C, H, W]
        if images_tensor.shape[1] != 3:
            images_tensor = images_tensor.permute(0, 3, 1, 2)

        images_tensor = images_tensor.to(device)

        with torch.no_grad():
            # VAE reconstruction
            reconstructed, _, _ = vae_model(images_tensor)

            # Classifier prediction
            logits = classifier_model(reconstructed)
            softmax_scores = torch.softmax(logits, dim=1)

            # Get softmax for predicted class
            pred_classes = torch.argmax(logits, dim=1)
            target_softmax = softmax_scores[torch.arange(len(images_tensor)), pred_classes]

            return target_softmax.cpu().numpy().tolist()

    return pred_func

# -----------------------------
# FULL DATASET SIC EVALUATION
# -----------------------------
def evaluate_mig_sic_full_dataset():  # Changed function name
    """Complete SIC evaluation for MIG method on ENTIRE dataset"""  # Changed comment
    print("="*60)
    print("SIC EVALUATION - MIG FULL DATASET")  # Changed print
    print("="*60)

    start_time = time.time()

    # SIC parameters
    saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75]

    # Load MIG attribution data - CHANGE 1
    print(" Loading MIG attribution data...")
    try:
        mig_data = torch.load('whole_dataset_mig_attributions.pt', map_location='cpu')  # Changed filename
        attribution_maps = mig_data['feature_attributions']
        image_indices = mig_data['image_indices']

        total_images = len(attribution_maps)
        print(f"✅ Loaded {total_images} attribution maps")
        print(f"Attribution tensor shape: {attribution_maps.shape}")

    except Exception as e:
        print(f" Error loading attribution data: {e}")
        return None

    # Load models
    print(" Loading models...")
    try:
        vae_net, classifier = load_models()
        pred_func = create_pred_func(vae_net, classifier)
        print(" Models and prediction function ready")
    except Exception as e:
        print(f" Error loading models: {e}")
        return None

    print(f" Processing ALL {total_images} images...")
    print(f" Estimated time: {total_images * 0.15 / 60:.1f} minutes")

    # Storage for results
    pic_results = []
    successful_evaluations = 0
    failed_evaluations = 0

    # Process each image in the dataset
    for i in tqdm(range(total_images), desc="Computing SIC"):
        try:
            # Prepare attribution map
            attr_3d = attribution_maps[i].reshape(3, 192, 192)
            saliency_map = attr_3d.abs().sum(dim=0).numpy()  # [192, 192]

            # Load original image
            img_idx = image_indices[i].item()
            original_img_tensor = load_image_by_index(img_idx)
            img_np = (original_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Generate random mask for this image
            random_mask = generate_random_mask(192, 192, fraction=0.01)

            # OFFICIAL SIC COMPUTATION
            pic_result = compute_pic_metric(
                img=img_np,
                saliency_map=saliency_map,
                random_mask=random_mask,
                pred_func=pred_func,
                saliency_thresholds=saliency_thresholds,
                min_pred_value=0.05,  # Adjusted for your model
                keep_monotonous=True,
                num_data_points=1000
            )

            pic_results.append(pic_result)
            successful_evaluations += 1

            # Memory cleanup every 100 images
            if i % 100 == 0:
                torch.cuda.empty_cache()

            # Progress update every 1000 images
            if (i + 1) % 1000 == 0:
                current_mean_auc = np.mean([r.auc for r in pic_results[-1000:]])
                elapsed_time = time.time() - start_time
                print(f"Progress: {i+1}/{total_images} | Recent Mean AUC: {current_mean_auc:.4f} | Time: {elapsed_time/60:.1f}min")

        except ComputePicMetricError as e:
            failed_evaluations += 1
            if failed_evaluations <= 10:  # Only print first 10 errors
                pass  # Skip verbose error printing for full dataset
            continue
        except Exception as e:
            failed_evaluations += 1
            if failed_evaluations <= 10:
                pass  # Skip verbose error printing for full dataset
            continue

    # OFFICIAL AGGREGATION
    if pic_results:
        print(f"\n Aggregating results from {len(pic_results)} successful evaluations...")
        aggregate_result = aggregate_individual_pic_results(pic_results, method='median')

        # Extract individual AUC scores for statistics
        individual_aucs = [r.auc for r in pic_results]

        results = {
            'method': 'MIG',  # Changed method name
            'dataset': 'Oxford Pets (Full)',
            'implementation': 'Official PAIR-code/saliency',
            'total_images': total_images,
            'successful_evaluations': successful_evaluations,
            'failed_evaluations': failed_evaluations,
            'success_rate': successful_evaluations / total_images,
            'aggregated_auc': aggregate_result.auc,
            'individual_aucs': individual_aucs,
            'mean_individual_auc': np.mean(individual_aucs),
            'std_individual_auc': np.std(individual_aucs),
            'median_individual_auc': np.median(individual_aucs),
            'min_individual_auc': np.min(individual_aucs),
            'max_individual_auc': np.max(individual_aucs),
            'evaluation_time_minutes': (time.time() - start_time) / 60,
            'parameters': {
                'saliency_thresholds': saliency_thresholds,
                'min_pred_value': 0.05,
                'keep_monotonous': True,
                'random_mask_fraction': 0.01,
                'aggregation_method': 'median'
            }
        }

        # Display results - FOCUSED OUTPUT
        print("\n" + "="*60)
        print("FINAL SIC RESULTS")
        print("="*60)
        print(f"Method: {results['method']}")
        print(f"Dataset: {results['dataset']}")
        print(f"Implementation: {results['implementation']}")
        print(f"Images Processed: {results['successful_evaluations']}/{results['total_images']}")
        print(f"Success Rate: {results['success_rate']*100:.1f}%")
        print(f"Evaluation Time: {results['evaluation_time_minutes']:.1f} minutes")
        print()
        print(" MAIN SIC RESULTS:")
        print(f"Aggregated SIC AUC: {results['aggregated_auc']:.4f}")
        print(f"Mean Individual AUC: {results['mean_individual_auc']:.4f}")
        print(f"Std Individual AUC: {results['std_individual_auc']:.4f}")
        print(f"Median Individual AUC: {results['median_individual_auc']:.4f}")
        print()

        # Performance assessment
        agg_auc = results['aggregated_auc']
        if agg_auc > 0.8:
            performance = "Excellent"
        elif agg_auc > 0.7:
            performance = "Good"
        elif agg_auc > 0.6:
            performance = "Moderate"
        elif agg_auc > 0.5:
            performance = "Poor but above chance"
        else:
            performance = "At or below chance level"

        print(f"Performance Assessment: {performance}")
        print(f"Standard Reporting: SIC AUC = {results['aggregated_auc']:.4f} (n={results['successful_evaluations']})")

        # Save results - CHANGE 2
        output_file = 'mig_sic_full_dataset_results.pt'  # Changed filename
        torch.save(results, output_file)
        print(f"\n Results saved to: {output_file}")

        return results
    else:
        print("\n No successful evaluations!")
        return None

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("🚀 OFFICIAL SIC EVALUATION - FULL DATASET")
    print("Using exact PAIR-code/saliency implementation")
    print("This will process ALL images in your dataset.")
    print()

    try:
        results = evaluate_mig_sic_full_dataset()  # Changed function call

        if results:
            print("\n FULL DATASET SIC EVALUATION COMPLETED SUCCESSFULLY!")
            print()
            print("Next Steps:")
            print("1. Compare with AUC-ROC results for comprehensive analysis")
            print("2. Implement MIG evaluation using same framework")
            print("3. Generate comparative research publication")
            print("4. Use results for method validation and improvement")
        else:
            print("\n Evaluation failed.")

    except Exception as e:
        print(f"\n Critical error: {e}")
        import traceback
        traceback.print_exc()
