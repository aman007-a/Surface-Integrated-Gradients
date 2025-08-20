import torch
from core.vae import VAE
from core import classifier as clf
import core.attribution_methods as att_mthds
from core.utils import interpolate
import core.config.configuration as cnfg
import core.geodesic as gdsc
from core.attributional_attack import target_attack_ig
from torchvision import transforms
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np

# == Setup ==
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# == Load VAE ==
vae_path = "Models/VAE_Perceptual_oxford_pets_192.pt"
vae = VAE(channel_in=cnfg.channel_in, ch=cnfg.channels,
          blocks=cnfg.blocks, latent_channels=cnfg.latent_channels).to(device)
vae.load_state_dict(torch.load(vae_path, map_location=device)["model_state_dict"])
vae.eval()

# == Load Classifier ==
clf_path = "Models/VAE_Perceptual_oxford_pets_192_classifier_resnet.pt"
if cnfg.backbone_type == "resnet":
    classifier = clf.ResClassifier().to(device)
elif cnfg.backbone_type == "vgg16":
    classifier = clf.VGG16Classifier().to(device)
else:
    classifier = clf.InceptionClassifier().to(device)
classifier.load_state_dict(torch.load(clf_path, map_location=device)["model_state_dict"])
classifier.eval()

# == Load Image ==
def load_image_by_index(img_index, img_size=192):
    image_dir = "../datasets/oxford_pets/images"
    image_files = sorted(os.listdir(image_dir))
    img_name = image_files[img_index]
    img_path = os.path.join(image_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).to(device)
    return img_tensor.unsqueeze(0), img_name  # shape: (1, 3, 192, 192)

# == Helper: load precomputed attribution map ==
def find_attribution_for_index(mc_data, idx):
    image_indices = mc_data['image_indices']
    attributions = mc_data['feature_attributions']
    positions = (image_indices == idx).nonzero(as_tuple=False)
    if len(positions) == 0:
        raise ValueError(f"Image index {idx} not found in attribution data")
    pos = positions[0].item()
    attr = attributions[pos]
    attr_3d = attr.view(3, 192, 192)
    attr_resized = torch.nn.functional.interpolate(attr_3d.unsqueeze(0), size=224, mode='bilinear').squeeze(0)
    return attr_resized.detach().cpu().numpy()

# == MIG Attribution (for attacked images) ==
def compute_mig_attr(input_image):
    black = torch.zeros_like(input_image).to(device)
    with torch.no_grad():
        black_z, _, _ = vae.encoder(black)
        x_z, _, _ = vae.encoder(input_image)
        rec_im, _, _ = vae(input_image)
        out = classifier(rec_im)
        target_class = out.argmax(dim=1).item()
        z_path = interpolate(black_z, x_z, cnfg.num_interpolants)
        geo_path = gdsc.geodesic_path_algorithm(
            vae, z_path, alpha=cnfg.default_alpha, T=cnfg.num_interpolants,
            beta=cnfg.beta, epsilon=cnfg.epsilon, max_iterations=2
        )
        geo_images = [vae.decoder(z) for z in geo_path]

    int_grads_geo = att_mthds.integrated_gradients_geo(
        classifier, geo_images, target_class, steps=cnfg.num_interpolants
    )

    # --- Resize to 224x224 so it matches the mc.pt attribution maps ---
    attr = int_grads_geo.squeeze(0).detach()
    if attr.shape[-1] != 224:  # resize only if needed
        attr = torch.nn.functional.interpolate(attr.unsqueeze(0), size=224, mode='bilinear').squeeze(0)

    return attr.cpu().numpy()

# == SSI Calculation ==
def compute_ssi(original_attr, attacked_attr):
    if original_attr.ndim == 3:
        original_attr = np.mean(original_attr, axis=0)
    if attacked_attr.ndim == 3:
        attacked_attr = np.mean(attacked_attr, axis=0)

    # normalize both to [0,1] before SSI (stability)
    original_attr = (original_attr - original_attr.min()) / (original_attr.max() - original_attr.min() + 1e-8)
    attacked_attr = (attacked_attr - attacked_attr.min()) / (attacked_attr.max() - attacked_attr.min() + 1e-8)

    score = ssim(original_attr, attacked_attr, data_range=1.0)
    return score

# == MAIN ==
if __name__ == "__main__":
    # ---- Load precomputed attribution maps ----
    mc_path = "whole_dataset_mc_attributions.pt"
    mc_data = torch.load(mc_path, map_location=device)

    # ---- Step 1: Load original image (index 5) ----
    index = 5
    original_img, img_name = load_image_by_index(index)

    # ---- Step 2: Load precomputed attribution for original ----
    original_attr = find_attribution_for_index(mc_data, index)

    # ---- Step 3: Generate attacked image ----
    baseline = torch.zeros_like(original_img).to(device)
    with torch.no_grad():
        rec_im, _, _ = vae(original_img)
        out = classifier(rec_im)
        target_class = out.argmax(dim=1).item()
    target_index = 10
    target_img, _ = load_image_by_index(target_index)
    attacked_img, adv_expl, org_expl, target_expl = target_attack_ig(
        classifier, baseline, target_class, original_img, target_img
    )

    # ---- Step 4: Compute MIG for attacked image ----
    attacked_attr = compute_mig_attr(attacked_img)

    # ---- Step 5: Compute SSI ----
    score = compute_ssi(original_attr, attacked_attr)
    print(f"SSI Score for image '{img_name}' (index {index}): {score:.4f}")
