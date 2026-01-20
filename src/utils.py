import torch
from PIL import Image
import torchvision.transforms as transforms
import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(path, size, device):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0) # [C,H,W] -> [1,C,H,W]
    return image.to(device)

def save_image(tensor, path):
    # tensor shape [1,C,H,W]
    image = tensor.detach().cpu().squeeze(0) #[C,H,W]
    image = image.permute(1, 2, 0) #[H,W,C]
    image = (image * 255).clamp(0,255).byte() # convert from [0,1]/pixel -> [0,255]
    Image.fromarray(image.numpy()).save(path)