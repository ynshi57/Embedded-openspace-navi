# predict.py
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from model_mobilenet_unet import MobileNetV2_UNet
from utils import plot_prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV2_UNet().to(device)
model.load_state_dict(torch.load("runs/mobilenet_unet_freespace.pth", map_location=device))
model.eval()

# 读取一张图像
image_path = "freespace_dataset/images/0000.png"
mask_path = "freespace_dataset/masks/0000.png"
image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

img_tensor = ToTensor()(image).unsqueeze(0).to(device)
mask_tensor = ToTensor()(mask)

with torch.no_grad():
    pred = model(img_tensor)

# 可视化
plot_prediction(img_tensor[0].cpu(), mask_tensor, pred[0].cpu())
