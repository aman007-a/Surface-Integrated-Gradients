import os
import torch
from torchvision.transforms import functional as TF
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import core.config.configuration as cnfg
from core.vae import VAE
from core import classifier as clf
from core.utils import interpolate
import core.geodesic as gdsc
import core.attribution_methods as att_mthds
import time

# ==== Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== Load VAE ====
vae_path = "Models/VAE_Perceptual_oxford_pets_192.pt"
vae = VAE(channel_in=cnfg.channel_in, ch=cnfg.channels,
          blocks=cnfg.blocks, latent_channels=cnfg.latent_channels).to(device)
vae.load_state_dict(torch.load(vae_path, map_location=device)["model_state_dict"])
vae.eval()

# ==== Load Classifier ====
clf_path = "Models/VAE_Perceptual_oxford_pets_192_classifier_resnet.pt"
if cnfg.backbone_type == "resnet":
    classifier = clf.ResClassifier().to(device)
elif cnfg.backbone_type == "vgg16":
    classifier = clf.VGG16Classifier().to(device)
else:
    classifier = clf.InceptionClassifier().to(device)

classifier.load_state_dict(torch.load(clf_path, map_location=device)["model_state_dict"])
classifier.eval()

# ==== OPTIMIZED: Load Images One-by-One (Memory Efficient) ====
def load_image_by_index(img_index, img_size=192):
    """Load single image by index - Memory efficient approach"""
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
    img_tensor = transform(image).to(device)  # (3, H, W)

    return img_tensor

def get_dataset_size():
    """Get total number of images without loading them"""
    image_dir = "../datasets/oxford_pets/images"
    image_files = sorted(os.listdir(image_dir))
    return len(image_files)

# ==== Enhanced Memory Management Functions ====
def aggressive_memory_cleanup():
    """Comprehensive CUDA memory cleanup"""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        # Additional cleanup
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()

def print_memory_usage():
    """Print detailed GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")

def check_memory_and_cleanup(threshold_gb=10.0):
    """Check memory usage and cleanup if above threshold"""
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        if allocated_gb > threshold_gb:
            aggressive_memory_cleanup()
            return True
    return False

# ==== MIG Attribution Computation - MEMORY OPTIMIZED ====
def compute_mig_attributions_whole_dataset_optimized():
    """Compute MIG attributions for ENTIRE dataset - MEMORY OPTIMIZED"""

    start_time = time.time()

    # Get dataset size without loading all images
    total_images = get_dataset_size()
    print(f"Found {total_images} images in dataset")

    # Create black baseline image
    black = torch.zeros(1, 3, 192, 192).to(device)

    # Encode black image once
    print("🔄 Encoding black baseline image...")
    with torch.no_grad():
        black_z, _, _ = vae.encoder(black)

    # Storage for results - EXACT SAME FORMAT AS MC
    all_results = {
        'feature_attributions': [],      # Same key as MC
        'variances': [],                 # Same key as MC (placeholder for MIG)
        'image_indices': [],             # Same key as MC
        'parameters': {                  # Same key as MC
            'num_samples': cnfg.num_interpolants,
            'batch_size': 1,
            'seed': 1337,
            'method': 'MIG',
            'baseline': 'black_image',
            'alpha': cnfg.default_alpha,
            'beta': cnfg.beta,
            'epsilon': cnfg.epsilon,
            'max_iterations': 2
        }
    }

    print(f"\n🔄 Computing MIG attributions for {total_images} images...")
    print(f"⏱️ Estimated time: {total_images * 0.2 / 60:.1f} minutes")
    print_memory_usage()

    successful_evaluations = 0
    failed_evaluations = 0

    # Process each image ONE-BY-ONE (MEMORY EFFICIENT)
    for i in tqdm(range(total_images), desc="Processing MIG"):
        try:
            # OPTIMIZED: Load single image on demand
            img_tensor = load_image_by_index(i)
            im = img_tensor.unsqueeze(0).to(device)  # (1, 3, 192, 192)

            with torch.no_grad():
                # Encode image
                x_z, _, _ = vae.encoder(im)
                # Reconstruct for classification
                rec_im, _, _ = vae(im)

            # Class prediction
            out = classifier(rec_im)
            target_class = out.argmax(dim=1)

            # Geodesic interpolation
            z_path = interpolate(black_z, x_z, cnfg.num_interpolants)
            geo_path = gdsc.geodesic_path_algorithm(
                vae, z_path, alpha=cnfg.default_alpha, T=cnfg.num_interpolants,
                beta=cnfg.beta, epsilon=cnfg.epsilon, max_iterations=2
            )

            # Decode geodesic images
            geo_images = [vae.decoder(z) for z in geo_path]

            # MIG Attribution
            int_grads_geo = att_mthds.integrated_gradients_geo(
                classifier, geo_images, target_class, steps=cnfg.num_interpolants
            )

            # FLATTEN TO MATCH MC FORMAT: [3, 192, 192] -> [110592]
            # FIXED: Detach gradients to prevent memory issues
            flattened_attribution = int_grads_geo.detach().cpu().view(-1)  # [110592]

            # Create placeholder variance
            placeholder_variance = (int_grads_geo.detach().cpu() ** 2).view(-1)  # [110592]

            # Store in MC format
            all_results['feature_attributions'].append(flattened_attribution)
            all_results['variances'].append(placeholder_variance)
            all_results['image_indices'].append(i)

            successful_evaluations += 1

            # ENHANCED: Comprehensive memory cleanup after each image
            del img_tensor, im, x_z, rec_im, out, target_class, z_path, geo_path, geo_images, int_grads_geo
            del flattened_attribution, placeholder_variance
            aggressive_memory_cleanup()

            # OPTIMIZED: More frequent progress updates with memory monitoring
            if (i + 1) % 50 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_image = elapsed_time / (i + 1)
                eta_minutes = (total_images - i - 1) * avg_time_per_image / 60
                print(f"Progress: {i+1}/{total_images} ({(i+1)/total_images*100:.1f}%) | "
                      f"Time: {elapsed_time/60:.1f}min | ETA: {eta_minutes:.1f}min")
                print_memory_usage()

            # ENHANCED: Preemptive memory cleanup for large datasets
            if (i + 1) % 10 == 0:
                check_memory_and_cleanup(threshold_gb=8.0)

        except Exception as e:
            print(f"❌ Error processing image {i}: {e}")
            failed_evaluations += 1
            # ENHANCED: Cleanup even on error
            try:
                del img_tensor, im, x_z, rec_im, out, target_class, z_path, geo_path, geo_images, int_grads_geo
            except:
                pass
            aggressive_memory_cleanup()
            continue

    # Final cleanup before tensor stacking
    print("\n🔄 Final cleanup and tensor stacking...")
    aggressive_memory_cleanup()

    # Convert lists to tensors - EXACT SAME FORMAT AS MC
    if all_results['feature_attributions']:
        print("Creating final tensors...")
        all_results['feature_attributions'] = torch.stack(all_results['feature_attributions'])  # [N, 110592]
        all_results['variances'] = torch.stack(all_results['variances'])                        # [N, 110592]
        all_results['image_indices'] = torch.tensor(all_results['image_indices'])              # [N]

        # Add summary statistics
        all_results['summary'] = {
            'total_images_processed': successful_evaluations,
            'total_images_failed': failed_evaluations,
            'success_rate': successful_evaluations / total_images,
            'evaluation_time_minutes': (time.time() - start_time) / 60,
            'memory_optimized': True
        }

        print(f"\n✅ Tensors created in MC format:")
        print(f"   feature_attributions shape: {all_results['feature_attributions'].shape}")
        print(f"   variances shape: {all_results['variances'].shape}")
        print(f"   image_indices shape: {all_results['image_indices'].shape}")

    return all_results

# ==== Main Execution ====
if __name__ == "__main__":
    print("="*60)
    print("MIG ATTRIBUTION - MEMORY OPTIMIZED (WHOLE DATASET)")
    print("="*60)
    print("This script will:")
    print("1. Load images ONE-BY-ONE (memory efficient)")
    print("2. Compute MIG feature attributions for each image")
    print("3. Aggressive memory cleanup after each image")
    print("4. Store in EXACT SAME FORMAT as whole_dataset_mc_attributions.pt")
    print("5. Output: whole_dataset_mig_attributions.pt")
    print()

    try:
        # Compute MIG attributions for entire dataset
        all_mig_data = compute_mig_attributions_whole_dataset_optimized()

        # Save to file with same naming convention as MC
        save_path = "whole_dataset_mig_attributions.pt"
        torch.save(all_mig_data, save_path)

        print("\n" + "="*60)
        print("MIG ATTRIBUTION COMPUTATION COMPLETED - WHOLE DATASET")
        print("="*60)
        print(f"✅ All MIG attributions saved to: {save_path}")

        if 'feature_attributions' in all_mig_data and all_mig_data['feature_attributions'] is not None:
            print(f"✅ feature_attributions shape: {all_mig_data['feature_attributions'].shape}")
            print(f"✅ variances shape: {all_mig_data['variances'].shape}")
            print(f"✅ image_indices shape: {all_mig_data['image_indices'].shape}")
            print(f"✅ Success rate: {all_mig_data['summary']['success_rate']*100:.1f}%")
            print(f"✅ Total time: {all_mig_data['summary']['evaluation_time_minutes']:.1f} minutes")
            print(f"✅ Data format: IDENTICAL to MC attribution format")
            print(f"✅ Memory optimized: {all_mig_data['summary']['memory_optimized']}")
            print()
            print("🔄 Ready for direct comparison with S_attr method!")
            print("🔄 Compatible with existing AUC-ROC and SIC evaluation scripts!")

        # Final memory status
        print(f"\n📊 FINAL MEMORY STATUS:")
        print_memory_usage()

    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        import traceback
        traceback.print_exc()
        # Emergency cleanup
        aggressive_memory_cleanup()
