import os
import gc
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from tqdm import tqdm

import core.config.configuration as cnfg
from core.vae import VAE
from core.classifier import ResClassifier, VGG16Classifier, InceptionClassifier


# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# Load VAE
# ============================================================
vae_ckpt_path = os.path.join("Models", f"{cnfg.model_name}.pt")

vae = VAE(
    channel_in=cnfg.channel_in,
    ch=cnfg.channels,
    blocks=cnfg.blocks,
    latent_channels=cnfg.latent_channels
).to(device)

vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device)['model_state_dict'])
vae.eval()


# ============================================================
# Load Classifier
# ============================================================
backbone = cnfg.backbone_type

if backbone == "resnet":
    classifier = ResClassifier()
elif backbone == "vgg16":
    classifier = VGG16Classifier()
else:
    classifier = InceptionClassifier()

clf_ckpt_path = os.path.join("Models", f"{cnfg.model_name}_classifier_{backbone}.pt")
classifier.load_state_dict(torch.load(clf_ckpt_path, map_location=device)["model_state_dict"])
classifier = classifier.to(device)
classifier.eval()


# ============================================================
# Image Loader (EXACT Training Preprocessing!!)
# ============================================================
def load_all_pet_images(img_size=cnfg.image_width):
    """Loads all Oxford Pets images using the SAME transforms as during training."""
    
    image_dir = "../datasets/oxford_pets/images"
    image_files = sorted(os.listdir(image_dir))

    all_images = []
    valid_indices = []

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cnfg.mean, std=cnfg.std)
    ])

    print(f"Loading {len(image_files)} images...")
    for idx, img_name in enumerate(tqdm(image_files, desc="Loading images")):
        try:
            img_path = os.path.join(image_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)

            all_images.append(img_tensor)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error loading image {idx} ({img_name}): {e}")

    print(f"Successfully loaded {len(all_images)} images")
    return all_images, valid_indices


# ============================================================
# Posterior PCA Directions
# ============================================================
def get_local_posterior_pca(encoder, img, num_samples=1000):
    """Estimate PCA directions from posterior samples q(z|x)."""
    
    with torch.no_grad():
        z_sample, mu, logvar = encoder(img, sample=True)

    mu_flat = mu.view(-1)
    std = torch.exp(0.5 * logvar.view(-1))

    eps = torch.randn(num_samples, mu_flat.shape[0], device=device)
    samples = mu_flat.unsqueeze(0) + eps * std.unsqueeze(0)

    pca = PCA(n_components=2)
    pca.fit(samples.cpu().numpy())

    v1 = torch.tensor(pca.components_[0], device=device, dtype=mu_flat.dtype)
    v2 = torch.tensor(pca.components_[1], device=device, dtype=mu_flat.dtype)

    return mu_flat, v1, v2


# ============================================================
# Circular surface for MC sampling
# ============================================================
def circular_surface(sigma, tau, z0, v1, v2, area=0.7):
    r = (area / (2 * torch.pi)) ** 0.5
    return (z0
            + r * v1 * torch.cos(2 * torch.pi * sigma) * torch.cos(torch.pi * tau)
            + r * v2 * torch.sin(2 * torch.pi * sigma) * torch.cos(torch.pi * tau))


def build_surface_fn_from_image(img_tensor, area=0.7):
    z0, v1, v2 = get_local_posterior_pca(vae.encoder, img_tensor)
    return lambda s, t: circular_surface(s, t, z0, v1, v2, area), z0, v1, v2


# ============================================================
# Monte Carlo Feature Attribution
# ============================================================
def compute_mc_feature_attributions_posterior_batched(
    z_surface, z0, decoder, model, device,
    num_samples=1000, batch_size=5, seed=42):

    torch.manual_seed(seed)

    z0 = z0.detach()
    latent_dim = z0.shape[0]

    mu = z0
    std = torch.ones_like(mu) * 0.1

    # Determine feature dimension
    test_z = mu.unsqueeze(0).view(1, latent_dim)
    test_z = test_z.view(1, 64, 12, 12)
    with torch.no_grad():
        test_dec = decoder(test_z)
    feature_dim = test_dec.numel()

    # Prepare accumulators
    attr_sum = torch.zeros(feature_dim, device=device)
    attr_sq_sum = torch.zeros(feature_dim, device=device)

    # Compute metric determinant sqrt(det g)
    sigma = torch.tensor(0.5, device=device, requires_grad=True)
    tau = torch.tensor(0.5, device=device, requires_grad=True)
    z_sample = z_surface(sigma, tau)

    grad_s, grad_t = torch.autograd.grad(
        z_sample, [sigma, tau],
        grad_outputs=torch.ones_like(z_sample),
        create_graph=True
    )

    g11 = (grad_s * grad_s).sum()
    g12 = (grad_s * grad_t).sum()
    g22 = (grad_t * grad_t).sum()
    det_g = g11 * g22 - g12**2
    sqrt_det_g = torch.sqrt(torch.relu(det_g) + 1e-8)

    # MC Processing
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        bs = min(batch_size, num_samples - batch_idx * batch_size)

        eps = torch.randn(bs, latent_dim, device=device)
        samples_z = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        samples_z = samples_z.view(bs, 64, 12, 12)

        decoded = decoder(samples_z)
        decoded.requires_grad_(True)

        f_out = model(decoded)
        f_scalar = f_out.max(dim=1).values

        grads = torch.autograd.grad(f_scalar.sum(), decoded)[0]
        grads = grads.view(bs, -1)

        contrib = grads * sqrt_det_g

        attr_sum += contrib.sum(dim=0)
        attr_sq_sum += (contrib**2).sum(dim=0)

        del samples_z, decoded, f_out, grads, contrib
        torch.cuda.empty_cache()

    mean_attr = attr_sum / num_samples
    var_attr = (attr_sq_sum / num_samples) - (mean_attr ** 2)

    del test_dec, std, mu, z0
    torch.cuda.empty_cache()

    return mean_attr.cpu(), var_attr.cpu()


# ============================================================
# Memory helpers
# ============================================================
def aggressive_memory_cleanup():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


# ============================================================
# Full Dataset Processing
# ============================================================
def compute_attributions_for_all_images():

    all_images, valid_indices = load_all_pet_images()

    results = {
        'feature_attributions': [],
        'variances': [],
        'image_indices': [],
        'parameters': {
            'num_samples': 1000,
            'batch_size': 5,
            'seed': 1337,
            'area': 0.7
        }
    }

    print("\nStarting MC attribution computation...")
    for i, img_tensor in enumerate(tqdm(all_images, desc="Processing")):
        try:
            z_surface, z0, v1, v2 = build_surface_fn_from_image(img_tensor)

            attr, var = compute_mc_feature_attributions_posterior_batched(
                z_surface, z0, decoder=vae.decoder, model=classifier,
                device=device, num_samples=1000, batch_size=5
            )

            results['feature_attributions'].append(attr)
            results['variances'].append(var)
            results['image_indices'].append(valid_indices[i])

            del z_surface, z0, v1, v2, attr, var
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error at image {valid_indices[i]}: {e}")

    aggressive_memory_cleanup()

    results['feature_attributions'] = torch.stack(results['feature_attributions'])
    results['variances'] = torch.stack(results['variances'])
    results['image_indices'] = torch.tensor(results['image_indices'])

    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    outputs = compute_attributions_for_all_images()
    
    save_path = "whole_dataset_mc_attributions.pt"
    torch.save(outputs, save_path)

    print("\n====================================")
    print("MC Attribution Computation Completed")
    print("====================================")
    print(f"Saved to: {save_path}")
    print(f"Images processed: {len(outputs['feature_attributions'])}")
    print(f"Attribution shape: {outputs['feature_attributions'].shape}")
    print(f"Variance shape: {outputs['variances'].shape}")
