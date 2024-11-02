import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
from torchvision import transforms
from PIL import Image
import requests
from pathlib import Path
import torchvision
import torch.nn.functional as F

def download_checkpoint():
    # Common checkpoint: ImageNet 16384
    url = "https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1"
    checkpoint_path = Path("vqgan_imagenet_f16_16384.ckpt")
    
    if not checkpoint_path.exists():
        response = requests.get(url)
        checkpoint_path.write_bytes(response.content)
    
    return checkpoint_path

def get_config():
    # Get config
    config_url = "https://raw.githubusercontent.com/CompVis/taming-transformers/master/configs/custom_vqgan.yaml"
    config = OmegaConf.create(requests.get(config_url).text)
    return config

def preprocess_image(image_path, target_size=256):
    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)  # Normalizes to [-1, 1]
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def load_vqgan(device='cuda'):
    config = get_config()
    model = VQModel(**config.model.params)
    checkpoint = torch.load(download_checkpoint())
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    return model

# Load model
vqgan = load_vqgan()
# Calculate and print total number of parameters
total_params = sum(p.numel() for p in vqgan.parameters())
print(f"Total number of parameters in VQGAN: {total_params:,}")

# Calculate trainable parameters
trainable_params = sum(p.numel() for p in vqgan.parameters() if p.requires_grad)
print(f"Number of trainable parameters in VQGAN: {trainable_params:,}")

# Your input tensor (B, C, H, W)
# x = torch.randn(1, 3, 256, 256)
image_tensor = preprocess_image("/home/nsml/MambaNeRV/data/bunny/0001.png").cuda()
# Encode
encoded = vqgan.encode(image_tensor)
z_q = encoded[0]  # This is the quantized representation
print("Encoded frame shape:", z_q.shape)

emb_loss = encoded[1]  # Commitment loss if you're training

# Decode
decoded = vqgan.decode(z_q)

torchvision.utils.save_image(F.interpolate(decoded, size=(720, 1280), mode='bilinear', align_corners=False), "decoded_vqgan.png")
# Denormalize decoded image (from [-1,1] to [0,1])
decoded = (decoded + 1) / 2
image_tensor = (image_tensor + 1) / 2

# Calculate PSNR
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

# Calculate PSNR between original and reconstructed
original_image = torchvision.io.read_image("/home/nsml/MambaNeRV/data/bunny/0001.png").float().cuda() / 255.0
decoded_resized = F.interpolate(decoded, size=original_image.shape[-2:], mode='bilinear', align_corners=False)
psnr_value = calculate_psnr(original_image.unsqueeze(0), decoded_resized)
print(f"PSNR between original and reconstructed: {psnr_value:.2f} dB")

