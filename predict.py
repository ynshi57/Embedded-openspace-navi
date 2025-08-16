# predict.py
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from model_mobilenet_unet import MobileNetV2_UNet
from utils import plot_prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 使用设备:", device)

# 初始化模型
model = MobileNetV2_UNet().to(device)

# 加载新格式的模型文件
model_path = "runs/freespace_model.pth"
print(f"📁 加载模型: {model_path}")

# 加载模型（处理新格式）
checkpoint = torch.load(model_path, map_location=device)
if 'model_state_dict' in checkpoint:
    # 新格式：包含额外信息的字典
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 模型加载成功！")
    print(f"📊 训练轮次: {checkpoint.get('final_epoch', 'N/A')}")
    print(f"🏆 最佳Loss: {checkpoint.get('best_loss', 'N/A'):.4f}")
    if 'final_metrics' in checkpoint:
        metrics = checkpoint['final_metrics']
        print(f"📈 最终指标 - IoU: {metrics.get('iou', 'N/A'):.4f}, "
              f"Dice: {metrics.get('dice', 'N/A'):.4f}, "
              f"Acc: {metrics.get('acc', 'N/A'):.4f}")
else:
    # 旧格式：直接是state_dict
    model.load_state_dict(checkpoint)
    print("✅ 模型加载成功！（旧格式）")

model.eval()

# 读取一张图像
image_path = "freespace_dataset/images/0000.png"
mask_path = "freespace_dataset/masks/0000.png"
image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

img_tensor = ToTensor()(image).unsqueeze(0).to(device)
mask_tensor = ToTensor()(mask)

print(f"🖼️ 输入图像尺寸: {img_tensor.shape}")
print(f"🎯 真实标签尺寸: {mask_tensor.shape}")

with torch.no_grad():
    pred = model(img_tensor)
    print(f"🔮 预测输出尺寸: {pred.shape}")
    print(f"📊 预测值范围: [{pred.min():.4f}, {pred.max():.4f}]")

# 可视化
plot_prediction(img_tensor[0].cpu(), mask_tensor, pred[0].cpu())
