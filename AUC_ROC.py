#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from skimage.transform import resize
from numpy import random
import core.config.configuration as cnfg
from core.vae import VAE
from core import classifier as clf
from core.classifier import ResClassifier, VGG16Classifier, InceptionClassifier
import time

# -----------------------------
# Device & Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Model Loading Functions
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

# -----------------------------
# Ground Truth Generation
# -----------------------------
def create_ground_truth_fixations(image, vae_model, classifier_model, top_k_percent=0.1):
    """Create binary fixation maps using model gradients"""
    # Ensure input requires gradients
    image = image.clone().detach().requires_grad_(True)

    # Forward pass through VAE + Classifier
    with torch.enable_grad():
        # VAE reconstruction
        reconstructed, _, _ = vae_model(image)

        # Classifier prediction
        output = classifier_model(reconstructed)
        pred_class = torch.argmax(output, dim=1)
        target_score = output[0, pred_class]

    # Compute gradients
    grad = torch.autograd.grad(target_score, image,
                              create_graph=False, retain_graph=False)[0]

    # Create importance map by averaging across channels
    importance = grad.abs().mean(dim=1).squeeze()  # [H, W]

    # Create binary fixation map (top k% pixels = 1, rest = 0)
    threshold = torch.quantile(importance, 1 - top_k_percent)
    fixation_map = (importance >= threshold).int().cpu().numpy()

    return fixation_map

# -----------------------------
# AUC-ROC Metric Implementation
# -----------------------------
def normalize_range(data):
    """Normalize data to [0, 1] range"""
    data_min = data.min()
    data_max = data.max()
    if data_max > data_min:
        return (data - data_min) / (data_max - data_min)
    else:
        return data

def AUC_Judd(saliency_map, fixation_map, jitter=True):
    """AUC-ROC metric from MIT Saliency Benchmark"""
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5

    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        return np.nan

    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='nearest')

    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += random.rand(*saliency_map.shape) * 1e-7

    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize_range(saliency_map)

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)

    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1

    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)
        tp[k+1] = (k + 1) / float(n_fix)
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix)

    return np.trapz(tp, fp)

# -----------------------------
# Complete Evaluation Pipeline - MIG DATASET
# -----------------------------
def evaluate_mig_auc_roc_full_dataset():
    """
    Complete AUC-ROC evaluation for MIG method on ENTIRE dataset
    """
    print("="*60)
    print("AUC-ROC EVALUATION - MIG FULL DATASET")
    print("="*60)

    start_time = time.time()

    # Load MIG attribution data
    print("📂 Loading MIG attribution data...")
    try:
        mig_data = torch.load('whole_dataset_mig_attributions.pt', map_location='cpu')
        attribution_maps = mig_data['feature_attributions']
        image_indices = mig_data['image_indices']

        total_images = len(attribution_maps)
        print(f"✅ Loaded {total_images} MIG attribution maps")
        print(f"Attribution tensor shape: {attribution_maps.shape}")

    except Exception as e:
        print(f"❌ Error loading MIG attribution data: {e}")
        return None

    # Load models
    print("🔄 Loading VAE and classifier models...")
    try:
        vae_net, classifier = load_models()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None

    print(f"📊 Processing ALL {total_images} images...")
    print(f"⏱️ Estimated time: {total_images * 0.1 / 60:.1f} minutes")

    # Storage for results
    auc_scores = []
    successful_evaluations = 0
    failed_evaluations = 0

    # Process each image in the dataset (OUT-OF-ORDER SAFE)
    for i in tqdm(range(total_images), desc="Evaluating MIG AUC-ROC"):
        try:
            # Reshape attribution map from flattened to 3D, then to 2D
            attr_3d = attribution_maps[i].reshape(3, 192, 192)  # [3, 192, 192]
            attr_map = attr_3d.abs().sum(dim=0).numpy()  # [192, 192] for AUC computation

            # 🔑 Get corresponding ORIGINAL image index (handles out-of-order)
            img_idx = image_indices[i].item()

            # Load the corresponding original image
            original_img = load_image_by_index(img_idx)

            # Create ground truth fixation map using model gradients
            with torch.no_grad():
                fixation_map = create_ground_truth_fixations(
                    original_img, vae_net, classifier, top_k_percent=0.1
                )

            # Compute AUC-ROC using MIT Saliency Benchmark implementation
            auc_score = AUC_Judd(attr_map, fixation_map, jitter=True)

            if not np.isnan(auc_score):
                auc_scores.append(auc_score)
                successful_evaluations += 1
            else:
                failed_evaluations += 1

            # Memory cleanup every 100 images
            if i % 100 == 0:
                torch.cuda.empty_cache()

            # Progress update every 1000 images
            if (i + 1) % 1000 == 0:
                current_mean = np.mean(auc_scores) if auc_scores else 0.0
                elapsed_time = time.time() - start_time
                print(f"Progress: {i+1}/{total_images} | Current Mean AUC: {current_mean:.4f} | Time: {elapsed_time/60:.1f}min")

        except Exception as e:
            failed_evaluations += 1
            if i < 10:  # Only print first few errors to avoid spam
                print(f"❌ Error processing image {i}: {e}")
            continue

    # Calculate final results
    if auc_scores:
        results = {
            'method': 'MIG',  # Updated method name
            'dataset': 'Oxford Pets (Full)',
            'total_images': total_images,
            'successful_evaluations': successful_evaluations,
            'failed_evaluations': failed_evaluations,
            'success_rate': successful_evaluations / total_images,
            'mean_auc': np.mean(auc_scores),
            'std_auc': np.std(auc_scores),
            'median_auc': np.median(auc_scores),
            'min_auc': np.min(auc_scores),
            'max_auc': np.max(auc_scores),
            'individual_scores': auc_scores,
            'evaluation_time_minutes': (time.time() - start_time) / 60,
            'parameters': {
                'top_k_percent': 0.1,
                'jitter': True,
                'device': str(device)
            }
        }

        # Display results - FOCUSED OUTPUT
        print("\n" + "="*60)
        print("FINAL MIG AUC-ROC RESULTS")
        print("="*60)
        print(f"Method: {results['method']}")
        print(f"Dataset: {results['dataset']}")
        print(f"Images Processed: {results['successful_evaluations']}/{results['total_images']}")
        print(f"Success Rate: {results['success_rate']*100:.1f}%")
        print(f"Evaluation Time: {results['evaluation_time_minutes']:.1f} minutes")
        print()
        print("🎯 MAIN RESULTS:")
        print(f"Mean AUC: {results['mean_auc']:.4f}")
        print(f"Std AUC:  {results['std_auc']:.4f}")
        print(f"Median AUC: {results['median_auc']:.4f}")
        print()

        # Performance assessment
        mean_auc = results['mean_auc']
        if mean_auc > 0.8:
            performance = "Excellent"
        elif mean_auc > 0.7:
            performance = "Good"
        elif mean_auc > 0.6:
            performance = "Moderate"
        elif mean_auc > 0.5:
            performance = "Poor but above chance"
        else:
            performance = "At or below chance level"

        print(f"Performance Assessment: {performance}")
        print(f"Standard Reporting: {results['mean_auc']:.4f} ± {results['std_auc']:.4f} (n={results['successful_evaluations']})")

        # Save results with updated filename
        output_file = 'mig_auc_roc_full_dataset_results.pt'
        torch.save(results, output_file)
        print(f"\n✅ Results saved to: {output_file}")

        return results
    else:
        print("\n❌ No successful evaluations!")
        return None

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("Starting AUC-ROC evaluation for MIG attribution method...")
    print("🎯 FULL DATASET EVALUATION")
    print("This will process ALL 7,390 images in your MIG dataset.")
    print()

    try:
        results = evaluate_mig_auc_roc_full_dataset()

        if results:
            print("\n🎉 MIG FULL DATASET EVALUATION COMPLETED SUCCESSFULLY!")
            print()
            print("Next Steps:")
            print("1. Compare with S_attr method for comprehensive evaluation")
            print("2. Implement SIC metric for complete paper validation")
            print("3. Use results for research publication")
        else:
            print("\n❌ Evaluation failed.")

    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        import traceback
        traceback.print_exc()
