# train.py（增强版本）
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_mobilenet_unet import MobileNetV2_UNet
from utils import FreespaceDataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 使用设备:", device)

# 初始化模型和优化器
model = MobileNetV2_UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# 支持训练轮次的函数
def train_one_round(epoch_start, epoch_end):
    dataset = FreespaceDataset("freespace_dataset/images", "freespace_dataset/masks")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(epoch_start, epoch_end):
        model.train()
        epoch_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"[{epoch+1}/{epoch_end}] Loss: {epoch_loss / len(dataloader):.4f}")

    # 每轮结束保存一次
    os.makedirs("runs", exist_ok=True)
    torch.save(model.state_dict(), f"runs/round_{epoch_end}.pth")
    print(f"✅ 保存模型：runs/round_{epoch_end}.pth")

# 初次训练 5 轮
train_one_round(0, 5)

# 你之后可以继续训练：
# train_one_round(5, 10)
# train_one_round(10, 20)
