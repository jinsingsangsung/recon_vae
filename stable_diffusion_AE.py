from diffusers import AutoencoderKL
import torch
import torchvision

# Load the pretrained autoencoder
# autoencoder = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
autoencoder = AutoencoderKL.from_pretrained("sd-x2-latent-upscaler")

# Your input tensor (B, C, H, W) - height and width can be arbitrary
x = torchvision.io.read_image("/home/nsml/MambaNeRV/data/bunny/0001.png").unsqueeze(0).float()  # (1, 3, H, W)


# Get input dimensions
_, _, H, W = x.shape
print(x.shape)
# Calculate number of patches needed
patch_size = 256
num_patches_h = (H + patch_size - 1) // patch_size  # Ceiling division
num_patches_w = (W + patch_size - 1) // patch_size

# Create padded input to handle non-divisible dimensions
padded_h = num_patches_h * patch_size
padded_w = num_patches_w * patch_size
padded_x = torch.nn.functional.pad(x, (0, padded_w - W, 0, padded_h - H))

# Initialize tensor for decoded result
decoded = torch.zeros_like(padded_x)

# Process each patch
for i in range(num_patches_h):
    for j in range(num_patches_w):
        # Extract patch
        h_start = i * patch_size
        w_start = j * patch_size
        patch = padded_x[:, :, h_start:h_start+patch_size, w_start:w_start+patch_size]
        
        # Encode and decode patch
        latents = autoencoder.encode(patch).latent_dist.sample()
        decoded_patch = autoencoder.decode(latents).sample
        
        # Place decoded patch back in the right position
        decoded[:, :, h_start:h_start+patch_size, w_start:w_start+patch_size] = decoded_patch

# Crop back to original size
decoded = decoded[:, :, :H, :W]

# Save reconstructed image
torchvision.utils.save_image(decoded, "decoded.png")

# Calculate PSNR
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

# Ensure images are in same range [0,1] and same device
x = x / 255.0  # Original image was loaded as 0-255
psnr_value = calculate_psnr(x, decoded)
print(f"PSNR between original and reconstructed: {psnr_value:.2f} dB")
