import torch
import gc
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from sklearn.decomposition import PCA
import core.config.configuration as cnfg
from core.vae import VAE
import numpy as np
from tqdm import tqdm
from core import classifier as clf
from core.classifier import ResClassifier, VGG16Classifier, InceptionClassifier

# -----------------------------
# Device & Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VAE model
vae = VAE(
    channel_in=3,
    ch=cnfg.channels,
    blocks=cnfg.blocks,
    latent_channels=cnfg.latent_channels
).to(device)

# Load VAE checkpoint
vae_checkpoint = torch.load(cnfg.save_dir + "/Models/" + cnfg.model_name + ".pt", map_location="cpu")
vae.load_state_dict(vae_checkpoint['model_state_dict'])
vae.eval()

# -----------------------------
# Load All Images from Oxford Pets
# -----------------------------
def load_all_pet_images(img_size=192):
    """Load all images from Oxford Pets dataset"""
    image_dir = "../datasets/oxford_pets/images"
    image_files = sorted(os.listdir(image_dir))

    all_images = []
    valid_indices = []

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    print(f"Loading {len(image_files)} images...")
    for idx, img_name in enumerate(tqdm(image_files, desc="Loading images")):
        try:
            img_path = os.path.join(image_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)  # Keep on GPU
            all_images.append(img_tensor)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error loading image {idx} ({img_name}): {e}")
            continue

    print(f"Successfully loaded {len(all_images)} images")
    return all_images, valid_indices

# -----------------------------
# Sample from Posterior q(z|x) and get PCA directions
# -----------------------------
def get_local_posterior_pca(encoder, img, num_samples=1000):
    """
    Get local PCA directions from the posterior distribution around z0
    """
    with torch.no_grad():
        z_sample, mu, logvar = encoder(img.to(device), sample=True)  # z_sample: [1,64,12,12]

    # Flatten latent maps to vectors
    mu = mu.view(-1)             # [64*12*12] = [9216]
    logvar = logvar.view(-1)
    std = torch.exp(0.5 * logvar)

    # Sample from the local posterior
    eps = torch.randn(num_samples, mu.shape[0], device=mu.device)    # [1000, 9216]
    samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)               # [1000, 9216]

    # Perform PCA on local samples
    pca = PCA(n_components=2)
    pca.fit(samples.cpu().numpy())
    v1 = torch.tensor(pca.components_[0], device=mu.device, dtype=mu.dtype)
    v2 = torch.tensor(pca.components_[1], device=mu.device, dtype=mu.dtype)

    # Return z0 and orthonormal directions v1, v2
    return mu, v1, v2

# -----------------------------
# Circular Surface with Fixed Area
# -----------------------------
def circular_surface(sigma, tau, z0, v1, v2, area=0.7):
    r = (area / (2 * torch.pi))**0.5  # Ensure surface area is fixed
    s = torch.cos(2 * torch.pi * sigma) * torch.cos(torch.pi * tau)
    t = torch.sin(2 * torch.pi * sigma) * torch.cos(torch.pi * tau)
    return z0 + r * (v1 * s + v2 * t)

# -----------------------------
# Build surface function from image
# -----------------------------
def build_surface_fn_from_image_tensor(img_tensor, area=0.7):
    """Build surface function from loaded image tensor"""
    z0, v1, v2 = get_local_posterior_pca(vae.encoder, img_tensor)

    def z_surface(sigma, tau):
        return circular_surface(sigma, tau, z0, v1, v2, area)

    return z_surface, z0, v1, v2

# Load classifier
backbone = cnfg.backbone_type

# Instantiate correct classifier
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

def compute_mc_feature_attributions_posterior_batched(z_surface, z0, decoder, model, device,
                                                      num_samples=1000, batch_size=5, seed=42):
    """
    Computes Attribution(xᵢ) ≈ E[∇f(x(σ,τ))_i * √det(g(σ,τ))] using Monte Carlo samples from posterior
    at z₀. Processes samples in batches to reduce memory usage.
    """
    torch.manual_seed(seed)

    z0 = z0.detach()
    z0_flat = z0.view(-1)
    latent_dim = z0_flat.shape[0]

    # Estimate covariance from posterior (assume diagonal covariance)
    with torch.no_grad():
        mu = z0_flat
        std = torch.ones_like(mu) * 0.1

    # Get the correct dimension for feature attributions (image space)
    # Generate a sample to determine the correct size
    sample_z = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(1, latent_dim, device=device)
    sample_z = sample_z.view(1, 64, 12, 12)
    with torch.no_grad():
        sample_decoded = decoder(sample_z)
        feature_dim = sample_decoded.numel()  # This will be 110592

    # Initialize accumulators with correct dimensions (image space)
    feature_attr_sum = torch.zeros(feature_dim, device=device)
    feature_attr_sq_sum = torch.zeros(feature_dim, device=device)

    # Compute metric tensor determinant once (since it's constant for your surface)
    sigma = torch.tensor(0.5, device=device, requires_grad=True)
    tau = torch.tensor(0.5, device=device, requires_grad=True)
    z_sample = z_surface(sigma, tau)
    grad_s, grad_t = torch.autograd.grad(z_sample, [sigma, tau],
                                         grad_outputs=torch.ones_like(z_sample),
                                         create_graph=True)
    g11 = (grad_s * grad_s).sum()
    g12 = (grad_s * grad_t).sum()
    g22 = (grad_t * grad_t).sum()
    det_g = g11 * g22 - g12 ** 2
    sqrt_det_g = torch.sqrt(torch.relu(det_g) + 1e-8)

    # Process samples in batches
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        # Calculate current batch size (handles the last batch)
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)

        # Generate batch of samples
        samples_z = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(current_batch_size, latent_dim, device=device)
        samples_z = samples_z.view(current_batch_size, 64, 12, 12)

        # Decode batch
        decoded_x = decoder(samples_z)
        decoded_x.requires_grad_(True)

        # Compute classifier output and gradients
        f_x = model(decoded_x)
        if f_x.ndim > 1:
            f_scalar = f_x.max(dim=1).values
        else:
            f_scalar = f_x.squeeze()

        grads = torch.autograd.grad(f_scalar.sum(), decoded_x, create_graph=False)[0]
        grads = grads.view(current_batch_size, -1)

        # Compute contributions for this batch
        contribs = grads * sqrt_det_g

        # Accumulate statistics
        feature_attr_sum += contribs.sum(dim=0)
        feature_attr_sq_sum += (contribs ** 2).sum(dim=0)

        # Clear variables from this batch
        del samples_z, decoded_x, f_x, grads, contribs

        # Clear GPU cache periodically

        torch.cuda.empty_cache()



    # Compute final statistics
    feature_attr_mean = feature_attr_sum / num_samples
    feature_attr_var = (feature_attr_sq_sum / num_samples) - (feature_attr_mean ** 2)

    # Normalize attributions
    #total_sum = feature_attr_mean.sum() + 1e-8
    #normalized_attr = feature_attr_mean / total_sum

    # Clean up before returning
    del  feature_attr_sq_sum, mu, std, z0_flat
    torch.cuda.empty_cache()

    return feature_attr_mean.cpu(), feature_attr_var.cpu()

# -----------------------------
# Memory Management Helper Functions
# -----------------------------
def aggressive_memory_cleanup():
    """Aggressive CUDA memory cleanup"""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def print_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# -----------------------------
# Process All Images and Store Results
# -----------------------------
def compute_attributions_for_all_images():
    """Compute Monte Carlo feature attributions for all images in the dataset"""

    # Load all images (kept on GPU)
    all_images, valid_indices = load_all_pet_images()

    # Storage for all results
    all_results = {
        'feature_attributions': [],
        'variances': [],
        'image_indices': [],
        'parameters': {
            'num_samples': 1000,
            'batch_size': 5,  # Changed from 10 to 5
            'seed': 1337,
            'area': 0.3
        }
    }

    print(f"\n🔍 Computing Monte Carlo feature attributions for {len(all_images)} images...")

    for i, img_tensor in enumerate(tqdm(all_images, desc="Processing images")):
       try:
        # Build surface function for this image
        z_surface, z0, v1, v2 = build_surface_fn_from_image_tensor(img_tensor, area=0.3)

        # Compute attributions for this image
        attr, var = compute_mc_feature_attributions_posterior_batched(
            z_surface=z_surface,
            z0=z0,
            decoder=vae.decoder,
            model=classifier,
            device=device,
            num_samples=1000,
            batch_size=5,
            seed=1337
        )

        # ✅ Move results to CPU *before* storing
        attr_cpu = attr.detach().cpu()
        var_cpu = var.detach().cpu()
        all_results['feature_attributions'].append(attr_cpu)
        all_results['variances'].append(var_cpu)
        all_results['image_indices'].append(valid_indices[i])

        # ✅ Aggressive cleanup of everything GPU-based
        del z_surface, z0, v1, v2, attr, var, attr_cpu, var_cpu
        torch.cuda.empty_cache()
        gc.collect()

        # ✅ Optional forced memory cleanup every 10 images
        #if (i + 1) % 10 == 0:
            #print(f"🔁 Forced cleanup after {i+1} images")
            #print_memory_usage()

        # ✅ Progress log every 50 images
        #if (i + 1) % 100 == 0:
            #print(f"Processed {i + 1}/{len(all_images)} images")
            #print_memory_usage()


       except Exception as e:
        print(f"Error processing image {valid_indices[i]}: {e}")
        try:
            del z_surface, z0, v1, v2, attr, var
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        continue







    # Final cleanup before tensor stacking
    aggressive_memory_cleanup()

    # Convert lists to tensors
    all_results['feature_attributions'] = torch.stack(all_results['feature_attributions'])
    all_results['variances'] = torch.stack(all_results['variances'])
    all_results['image_indices'] = torch.tensor(all_results['image_indices'])

    return all_results

# === Main Execution ===
if __name__ == "__main__":
    # Process all images and get attributions
    all_attribution_data = compute_attributions_for_all_images()

    # Save to single file
    save_path = "whole_dataset_mc_attributions.pt"
    torch.save(all_attribution_data, save_path)

    print(f"\n✅ All Monte Carlo attributions saved to {save_path}")
    print(f"Total images processed: {len(all_attribution_data['feature_attributions'])}")
    print(f"Attribution tensor shape: {all_attribution_data['feature_attributions'].shape}")
    print(f"Variance tensor shape: {all_attribution_data['variances'].shape}")
    print(f"Feature dimension per image: {all_attribution_data['feature_attributions'].shape[1]}")
