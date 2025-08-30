# train.py（连续训练版本）
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_mobilenet_unet import MobileNetV2_UNet
from utils import KittiRoadDataset
import os
import numpy as np
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 使用设备:", device)

# 初始化模型和优化器
model = MobileNetV2_UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss(ignore_index=255)

# 计算IoU指标
# 注意：pred为logits (B,C,H,W)，target为LongTensor (B,H,W)，忽略255
# 二类时按前景(类1)计算IoU

def calculate_iou_logits(pred_logits, target, num_classes=2, ignore_index=255, foreground_class=1):
    with torch.no_grad():
        pred = torch.argmax(pred_logits, dim=1)  # (B,H,W)
        valid = target != ignore_index
        if valid.sum() == 0:
            return 0.0
        pred_fg = (pred == foreground_class) & valid
        target_fg = (target == foreground_class) & valid
        intersection = (pred_fg & target_fg).sum().float()
        union = (pred_fg | target_fg).sum().float()
        iou = intersection / (union + 1e-6)
        return iou.item()

# Dice系数（针对前景类）
def calculate_dice_logits(pred_logits, target, ignore_index=255, foreground_class=1):
    with torch.no_grad():
        pred = torch.argmax(pred_logits, dim=1)
        valid = target != ignore_index
        if valid.sum() == 0:
            return 0.0
        pred_fg = (pred == foreground_class) & valid
        target_fg = (target == foreground_class) & valid
        intersection = (pred_fg & target_fg).sum().float()
        dice = (2 * intersection) / (pred_fg.sum().float() + target_fg.sum().float() + 1e-6)
        return dice.item()

# 像素准确率（忽略255）
def calculate_pixel_accuracy_logits(pred_logits, target, ignore_index=255):
    with torch.no_grad():
        pred = torch.argmax(pred_logits, dim=1) # 返回每一行最大值的索引，dim=0表示列
        valid = target != ignore_index
        if valid.sum() == 0:
            return 0.0
        correct = (pred[valid] == target[valid]).sum().float()
        total = valid.sum().float()
        return (correct / (total + 1e-6)).item()


def train_all_data(total_epochs=50, save_interval=10):
    """连续训练所有数据，每save_interval轮保存一次，最终只保留一个模型。加入验证划分并保存最佳模型。"""
    full_dataset = KittiRoadDataset("freespace_dataset/images", "freespace_dataset/masks", augment=True)
    n = len(full_dataset)
    indices = list(range(n))
    # 固定划分 80/20
    split = int(0.8 * n)
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    # 验证集不做增强
    val_dataset = KittiRoadDataset("freespace_dataset/images", "freespace_dataset/masks", augment=False)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)
    
    print(f"📊 训练集: {len(train_subset)} 张 | 验证集: {len(val_subset)} 张")

    best_val_iou = -1.0
    best_epoch = 0

    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_dice = 0.0
        epoch_acc = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            preds = model(images)
            loss = criterion(preds, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                iou = calculate_iou_logits(preds, masks)
                dice = calculate_dice_logits(preds, masks)
                acc = calculate_pixel_accuracy_logits(preds, masks)
            
            epoch_loss += loss.item()
            epoch_iou += iou
            epoch_dice += dice
            epoch_acc += acc
        
        avg_loss = epoch_loss / len(train_loader)
        avg_iou = epoch_iou / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        avg_acc = epoch_acc / len(train_loader)

        # 验证
        model.eval()
        val_iou = 0.0
        val_dice = 0.0
        val_acc = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                val_iou += calculate_iou_logits(preds, masks)
                val_dice += calculate_dice_logits(preds, masks)
                val_acc += calculate_pixel_accuracy_logits(preds, masks)
        if len(val_loader) > 0:
            val_loss /= len(val_loader)
            val_iou /= len(val_loader)
            val_dice /= len(val_loader)
            val_acc /= len(val_loader)
        else:
            val_loss = float('nan')
            val_iou = float('nan')
            val_dice = float('nan')
            val_acc = float('nan')
        
        print(f"[{epoch+1}/{total_epochs}] Train Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f} | Acc: {avg_acc:.4f} || Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳
        if not np.isnan(val_iou) and val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            os.makedirs("runs", exist_ok=True)
            best_path = "runs/best_model_val_iou.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': {
                    'loss': avg_loss,
                    'iou': avg_iou,
                    'dice': avg_dice,
                    'acc': avg_acc
                },
                'val_metrics': {
                    'loss': val_loss,
                    'iou': val_iou,
                    'dice': val_dice,
                    'acc': val_acc
                }
            }, best_path)
            print(f"🏆 更新最佳模型(Val IoU={val_iou:.4f}) → {best_path}")
        
        # 周期性保存
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
                'acc': avg_acc,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_dice': val_dice,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"💾 保存检查点: {checkpoint_path}")

    # 训练完成，保存最终模型
    os.makedirs("runs", exist_ok=True)
    final_model_path = "runs/freespace_model.pth"
    torch.save({
        'final_epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'best_val_iou': best_val_iou,
        'best_epoch': best_epoch,
        'final_metrics': {
            'loss': avg_loss,
            'iou': avg_iou,
            'dice': avg_dice,
            'acc': avg_acc,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_dice': val_dice,
            'val_acc': val_acc
        }
    }, final_model_path)

    print(f"\n🎉 训练完成！")
    print(f"📁 最终模型: {final_model_path}")
    print(f"🏆 最佳验证IoU: {best_val_iou:.4f} (第{best_epoch}轮)")
    print(f"📊 最终指标:")
    print(f"   Train Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f} | Acc: {avg_acc:.4f}")
    print(f"   Val   Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f} | Acc: {val_acc:.4f}")

    # 清理旧的周期性检查点（可选）
    for i in range(save_interval, total_epochs + 1, save_interval):
        checkpoint_path = f"runs/checkpoint_epoch_{i}.pth"
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"🗑️ 删除检查点: {checkpoint_path}")

    return final_model_path

# 开始训练
if __name__ == "__main__":
    # 训练50轮，每10轮保存一次检查点
    final_model = train_all_data(total_epochs=30, save_interval=10)
