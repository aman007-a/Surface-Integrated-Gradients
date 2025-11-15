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

# ============================================
# Device
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================
# Load VAE
# ============================================

vae_checkpoint = f"Models/VAE_Perceptual_{cnfg.dataset_name}_{cnfg.image_width}.pt"

vae = VAE(
    channel_in=cnfg.channel_in,
    ch=cnfg.channels,
    blocks=cnfg.blocks,
    latent_channels=cnfg.latent_channels
).to(device)

vae.load_state_dict(torch.load(vae_checkpoint, map_location=device)["model_state_dict"])
vae.eval()


# ============================================
# Load Classifier
# ============================================

clf_checkpoint = f"Models/VAE_Perceptual_{cnfg.dataset_name}_{cnfg.image_width}_classifier_resnet.pt"

if cnfg.backbone_type == "resnet":
    classifier = clf.ResClassifier().to(device)
elif cnfg.backbone_type == "vgg16":
    classifier = clf.VGG16Classifier().to(device)
else:
    classifier = clf.InceptionClassifier().to(device)

classifier.load_state_dict(torch.load(clf_checkpoint, map_location=device)["model_state_dict"])
classifier.eval()


# ============================================
# Image Loading (Matches Training Exactly)
# ============================================

def load_image_by_index(img_index, img_size=cnfg.image_width):
    """Load single dataset image with EXACT same transforms as VAE training."""
    
    image_dir = "../datasets/oxford_pets/images"
    image_files = sorted(os.listdir(image_dir))

    if img_index >= len(image_files):
        raise IndexError(f"Index {img_index} out of range ({len(image_files)} images).")

    img_path = os.path.join(image_dir, image_files[img_index])

    image = Image.open(img_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cnfg.mean, std=cnfg.std)
    ])

    return transform(image).to(device)   # (3,H,W)


def get_dataset_size():
    image_dir = "/home/mtech0/24CS60R35/Manifold-Integrated-Gradients/datasets/oxford_pets/images"
    return len(os.listdir(image_dir))


# ============================================
# Memory Management
# ============================================

def aggressive_memory_cleanup():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()


def print_memory_usage():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        res = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory — allocated: {alloc:.2f} GB | reserved: {res:.2f} GB")


# ============================================
# MIG Attribution Computation
# ============================================

def compute_mig_attributions_whole_dataset_optimized():
    """Memory-efficient, correct preprocessing, K80-safe MIG computation."""

    start_time = time.time()
    total_images = get_dataset_size()

    print(f"\nFound {total_images} images in dataset")

    # >>> Baseline image (correctly normalized)
    black = torch.zeros(1, 3, cnfg.image_width, cnfg.image_width, device=device)
    for c in range(3):
        black[:, c] = (black[:, c] - cnfg.mean[c]) / cnfg.std[c]

    with torch.no_grad():
        black_z, _, _ = vae.encoder(black)

    # Output container (matches MC format exactly)
    all_results = {
        'feature_attributions': [],
        'variances': [],
        'image_indices': [],
        'parameters': {
            'num_samples': cnfg.num_interpolants,
            'batch_size': 1,
            'seed': 1337,
            'method': 'MIG',
            'baseline': 'black_image',
            'alpha': cnfg.default_alpha,
            'beta': cnfg.beta,
            'epsilon': cnfg.epsilon,
            'max_iterations': cnfg.max_iterations,
        }
    }

    successful = 0
    failed = 0

    print(f"\nComputing MIG attributions for {total_images} images...\n")
    print_memory_usage()

    # Process images one-by-one (memory efficient)
    for i in tqdm(range(total_images), desc="Processing MIG"):

        try:
            img = load_image_by_index(i)
            im = img.unsqueeze(0)   # (1,3,H,W)

            with torch.no_grad():
                x_z, _, _ = vae.encoder(im)
                rec_im, _, _ = vae(im)

            out = classifier(rec_im)
            target_class = out.argmax(dim=1)

            # Geodesic computation
            z_path = interpolate(black_z, x_z, cnfg.num_interpolants)

            geo_path = gdsc.geodesic_path_algorithm(
                vae, z_path,
                alpha=cnfg.default_alpha,
                T=cnfg.num_interpolants,
                beta=cnfg.beta,
                epsilon=cnfg.epsilon,
                max_iterations=cnfg.max_iterations
            )

            geo_images = [vae.decoder(z) for z in geo_path]

            int_grads_geo = att_mthds.integrated_gradients_geo(
                classifier, geo_images, target_class, steps=cnfg.num_interpolants
            )

            flat_attr = int_grads_geo.detach().cpu().view(-1)
            var_est = (int_grads_geo.detach().cpu() ** 2).view(-1)

            all_results['feature_attributions'].append(flat_attr)
            all_results['variances'].append(var_est)
            all_results['image_indices'].append(i)

            successful += 1

            del img, im, x_z, rec_im, out, z_path, geo_path, geo_images, int_grads_geo
            aggressive_memory_cleanup()

        except Exception as e:
            print(f"❌ Error at image {i}: {e}")
            failed += 1
            aggressive_memory_cleanup()

    # =============================
    # Final Tensor Conversion
    # =============================
    if all_results['feature_attributions']:
        all_results['feature_attributions'] = torch.stack(all_results['feature_attributions'])
        all_results['variances'] = torch.stack(all_results['variances'])
        all_results['image_indices'] = torch.tensor(all_results['image_indices'])

        all_results['summary'] = {
            'processed': successful,
            'failed': failed,
            'success_rate': successful / total_images,
            'minutes': (time.time() - start_time) / 60,
            'memory_optimized': True
        }

    return all_results


# ============================================
# Main
# ============================================
if __name__ == "__main__":

    print("="*60)
    print("MIG ATTRIBUTION - FULL DATASET - MEMORY OPTIMIZED")
    print("="*60)

    all_mig_data = compute_mig_attributions_whole_dataset_optimized()

    save_path = f"whole_dataset_mig_attributions_{cnfg.dataset_name}_{cnfg.image_width}.pt"
    torch.save(all_mig_data, save_path)

    print("\nSaved MIG attribution file to:", save_path)
    print("Final memory:")
    print_memory_usage()
