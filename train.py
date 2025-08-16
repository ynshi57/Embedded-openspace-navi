# train.py（连续训练版本）
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_mobilenet_unet import MobileNetV2_UNet
from utils import FreespaceDataset
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 使用设备:", device)

# 初始化模型和优化器
model = MobileNetV2_UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# 计算IoU指标
def calculate_iou(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-6)
    return iou.item()

# 计算Dice系数
def calculate_dice(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    dice = (2 * intersection) / (pred.sum() + target.sum() + 1e-6)
    return dice.item()

# 计算像素准确率
def calculate_pixel_accuracy(pred, target):
    pred = (pred > 0.5).float()
    correct = (pred == target).sum()
    total = target.numel()
    return (correct / total).item()

def train_all_data(total_epochs=50, save_interval=10):
    """连续训练所有数据，每save_interval轮保存一次，最终只保留一个模型"""
    dataset = FreespaceDataset("freespace_dataset/images", "freespace_dataset/masks")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"📊 数据集大小: {len(dataset)} 张图片")
    print(f"🔄 开始训练 {total_epochs} 轮...")
    
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_dice = 0.0
        epoch_acc = 0.0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            
            # 前向传播
            preds = model(images)
            loss = criterion(preds, masks)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算指标
            with torch.no_grad():
                iou = calculate_iou(preds, masks)
                dice = calculate_dice(preds, masks)
                acc = calculate_pixel_accuracy(preds, masks)
            
            epoch_loss += loss.item()
            epoch_iou += iou
            epoch_dice += dice
            epoch_acc += acc
        
        # 计算平均指标
        avg_loss = epoch_loss / len(dataloader)
        avg_iou = epoch_iou / len(dataloader)
        avg_dice = epoch_dice / len(dataloader)
        avg_acc = epoch_acc / len(dataloader)
        
        # 记录最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
        
        print(f"[{epoch+1}/{total_epochs}] "
              f"Loss: {avg_loss:.4f} | "
              f"IoU: {avg_iou:.4f} | "
              f"Dice: {avg_dice:.4f} | "
              f"Acc: {avg_acc:.4f}")
        
        # 每save_interval轮保存一次检查点
        if (epoch + 1) % save_interval == 0:
            os.makedirs("runs", exist_ok=True)
            checkpoint_path = f"runs/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'iou': avg_iou,
                'dice': avg_dice,
                'acc': avg_acc
            }, checkpoint_path)
            print(f"💾 保存检查点: {checkpoint_path}")
    
    # 训练完成，保存最终模型
    os.makedirs("runs", exist_ok=True)
    final_model_path = "runs/freespace_model.pth"
    torch.save({
        'final_epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'final_metrics': {
            'loss': avg_loss,
            'iou': avg_iou,
            'dice': avg_dice,
            'acc': avg_acc
        }
    }, final_model_path)
    
    print(f"\n🎉 训练完成！")
    print(f"📁 最终模型: {final_model_path}")
    print(f"🏆 最佳Loss: {best_loss:.4f} (第{best_epoch}轮)")
    print(f"📊 最终指标:")
    print(f"   Loss: {avg_loss:.4f}")
    print(f"   IoU: {avg_iou:.4f}")
    print(f"   Dice: {avg_dice:.4f}")
    print(f"   Acc: {avg_acc:.4f}")
    
    # 清理检查点文件，只保留最终模型
    for i in range(save_interval, total_epochs + 1, save_interval):
        checkpoint_path = f"runs/checkpoint_epoch_{i}.pth"
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"🗑️ 删除检查点: {checkpoint_path}")
    
    return final_model_path

# 开始训练
if __name__ == "__main__":
    # 训练50轮，每10轮保存一次检查点
    final_model = train_all_data(total_epochs=50, save_interval=10)
