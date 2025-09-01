# train.py（连续训练版本）
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_mobilenet_unet import MobileNetV2_UNet
from utils import KittiRoadDataset
import os
import numpy as np
import argparse
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 使用设备:", device)

# 初始化模型
model = MobileNetV2_UNet().to(device)

def create_optimizer_with_different_lr(model, encoder_lr=1e-5, decoder_lr=1e-4, weight_decay=0):
    """创建分层学习率的优化器
    
    Args:
        model: MobileNetV2_UNet模型
        encoder_lr: 编码器学习率（预训练部分，较小）
        decoder_lr: 解码器学习率（新训练部分，较大）
        weight_decay: 权重衰减
    
    Returns:
        optimizer: 配置好的优化器
    """
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:  # 只处理需要梯度的参数
            if 'enc' in name:  # 编码器参数
                encoder_params.append(param)
            else:  # 解码器参数（up1, up2, up3, up4, final_up, out_conv）
                decoder_params.append(param)
    
    param_groups = []
    
    if encoder_params:
        param_groups.append({
            'params': encoder_params,
            'lr': encoder_lr,
            'weight_decay': weight_decay
        })
        print(f"编码器参数组: {len(encoder_params)} 个参数, LR={encoder_lr}")
    
    if decoder_params:
        param_groups.append({
            'params': decoder_params,
            'lr': decoder_lr,
            'weight_decay': weight_decay
        })
        print(f"解码器参数组: {len(decoder_params)} 个参数, LR={decoder_lr}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(param_groups)
    return optimizer

# 创建优化器（默认配置）
optimizer = create_optimizer_with_different_lr(model)

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
        pred = torch.argmax(pred_logits, dim=1)
        valid = target != ignore_index
        if valid.sum() == 0:
            return 0.0
        correct = (pred[valid] == target[valid]).sum().float()
        total = valid.sum().float()
        return (correct / (total + 1e-6)).item()


def freeze_encoder(model: MobileNetV2_UNet, freeze: bool):
    for m in [model.enc0, model.enc1, model.enc2, model.enc3, model.enc4, model.enc5]:
        for p in m.parameters():
            p.requires_grad = not (not not False) if False else (not False)
    # 上面只是占位，真正逻辑如下：
    for m in [model.enc0, model.enc1, model.enc2, model.enc3, model.enc4, model.enc5]:
        for p in m.parameters():
            p.requires_grad = not freeze

def build_concat_dataset(image_dirs, mask_dirs, augment: bool):
    assert len(image_dirs) == len(mask_dirs), "image_dirs 与 mask_dirs 数量需一致"
    datasets = []
    for img_dir, msk_dir in zip(image_dirs, mask_dirs):
        if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
            print(f"跳过无效数据目录: {img_dir} | {msk_dir}")
            continue
        ds = KittiRoadDataset(img_dir, msk_dir, augment=augment)
        if len(ds) == 0:
            print(f"空数据集: {img_dir} | {msk_dir}")
            continue
        print(f"加载数据集: {img_dir} ({len(ds)} 张)")
        datasets.append(ds)
    if len(datasets) == 0:
        raise ValueError("未找到有效的数据集目录")
    return ConcatDataset(datasets)

def build_group_datasets(old_imgs, old_msks, new_imgs, new_msks):
    """返回: (train_old, val_old, train_new, val_new) 按各自80/20划分"""
    train_old = val_old = train_new = val_new = None
    if old_imgs and old_msks:
        full_old_aug = build_concat_dataset(old_imgs, old_msks, augment=True)
        n_old = len(full_old_aug)
        idx_old = list(range(n_old))
        split_old = int(0.8 * n_old)
        train_old = torch.utils.data.Subset(full_old_aug, idx_old[:split_old])
        full_old_noaug = build_concat_dataset(old_imgs, old_msks, augment=False)
        val_old = torch.utils.data.Subset(full_old_noaug, idx_old[split_old:])
    if new_imgs and new_msks:
        full_new_aug = build_concat_dataset(new_imgs, new_msks, augment=True)
        n_new = len(full_new_aug)
        idx_new = list(range(n_new))
        split_new = int(0.8 * n_new)
        train_new = torch.utils.data.Subset(full_new_aug, idx_new[:split_new])
        full_new_noaug = build_concat_dataset(new_imgs, new_msks, augment=False)
        val_new = torch.utils.data.Subset(full_new_noaug, idx_new[split_new:])
    return train_old, val_old, train_new, val_new

def train_all_data(total_epochs=50, save_interval=10, resume_from: str = None, resume_optimizer: bool = True,
                   finetune: bool = False, finetune_lr: float = 1e-5, freeze_encoder_epochs: int = 0,
                   image_dirs=None, mask_dirs=None,
                   old_image_dirs=None, old_mask_dirs=None, new_image_dirs=None, new_mask_dirs=None,
                   new_ratio: float = 0.8, encoder_lr: float = 1e-5, decoder_lr: float = 1e-4, weight_decay: float = 0):
    """连续训练所有数据, 每save_interval轮保存一次, 最终只保留一个模型。加入验证划分并保存最佳模型。
    支持增量训练：
      - resume_from: 路径，加载检查点继续训练
      - finetune: 仅加载权重,重建优化器并用较小lr, 可在前若干epoch冻结encoder
      - image_dirs/mask_dirs: 传入多个数据目录以混合训练（旧+新）
      - old/new_* + new_ratio: 分别指定旧域和新域数据，按比例采样混合训练，分别汇报验证指标
    """
    # 处理增量训练/微调
    start_epoch = 0
    if resume_from:
        if os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            
            # 重新创建优化器（使用分层学习率）
            if finetune:
                # 微调模式：使用更小的学习率
                optimizer = create_optimizer_with_different_lr(
                    model, 
                    encoder_lr=finetune_lr * 0.1,  # 编码器更小学习率
                    decoder_lr=finetune_lr,        # 解码器使用指定学习率
                    weight_decay=weight_decay
                )
                print(f"微调模式: 仅加载权重, 重新创建优化器")
                print(f"  编码器LR: {finetune_lr * 0.1}, 解码器LR: {finetune_lr}")
            else:
                # 断点续训：使用正常学习率
                optimizer = create_optimizer_with_different_lr(
                    model, 
                    encoder_lr=encoder_lr, 
                    decoder_lr=decoder_lr, 
                    weight_decay=weight_decay
                )
                
                if resume_optimizer and 'optimizer_state_dict' in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                        print("断点续训: 已恢复优化器状态")
                    except Exception as e:
                        print(f"优化器状态恢复失败: {e}")
                        print("将使用新的优化器配置继续训练")
                
                if 'epoch' in ckpt:
                    start_epoch = int(ckpt['epoch'])
                elif 'final_epoch' in ckpt:
                    start_epoch = int(ckpt['final_epoch'])
                else:
                    start_epoch = 0
                print(f"从第 {start_epoch+1} 轮继续训练")
        else:
            print(f"resume_from 路径不存在: {resume_from}，将从头训练")
            # 从头训练：创建新的优化器
            optimizer = create_optimizer_with_different_lr(
                model, 
                encoder_lr=encoder_lr, 
                decoder_lr=decoder_lr, 
                weight_decay=weight_decay
            )
    else:
        # 从头训练：创建新的优化器
        optimizer = create_optimizer_with_different_lr(
            model, 
            encoder_lr=encoder_lr, 
            decoder_lr=decoder_lr, 
            weight_decay=weight_decay
        )

    # 数据集构建
    # 优先使用 old/new 分组；若未提供，则回退到 image_dirs/mask_dirs；再回退到默认 freespace_dataset
    if (old_image_dirs and old_mask_dirs) or (new_image_dirs and new_mask_dirs):
        train_old, val_old, train_new, val_new = build_group_datasets(old_image_dirs, old_mask_dirs, new_image_dirs, new_mask_dirs)
        train_parts = []
        val_parts = []
        n_old = len(train_old) if train_old is not None else 0
        n_new = len(train_new) if train_new is not None else 0
        if train_old is not None:
            train_parts.append(train_old)
        if train_new is not None:
            train_parts.append(train_new)
        if len(train_parts) == 0:
            raise ValueError("未提供有效的训练数据")
        train_mixed = ConcatDataset(train_parts)
        # 验证集：分别保留
        val_mixed = None
        if val_old is not None and val_new is not None:
            val_mixed = ConcatDataset([val_old, val_new])
        elif val_old is not None:
            val_mixed = val_old
        elif val_new is not None:
            val_mixed = val_new
        # 采样权重：按 new_ratio 对新域采样倾斜
        sample_weights = []
        for i in range(len(train_mixed)):
            # ConcatDataset将子集顺序拼接：先old后new（若两者都存在且按上述append顺序）
            if train_old is not None and i < len(train_old):
                # old
                w = max(1e-6, (1.0 - new_ratio) / max(1, n_old))
            else:
                # new
                w = max(1e-6, new_ratio / max(1, n_new))
            sample_weights.append(w)
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(train_mixed), replacement=True)
        train_loader = DataLoader(train_mixed, batch_size=4, sampler=sampler)
        val_loader_mixed = DataLoader(val_mixed, batch_size=4, shuffle=False) if val_mixed is not None else None
        val_loader_old = DataLoader(val_old, batch_size=4, shuffle=False) if val_old is not None else None
        val_loader_new = DataLoader(val_new, batch_size=4, shuffle=False) if val_new is not None else None
        total_train = len(train_mixed)
        total_val = len(val_mixed) if val_mixed is not None else 0
        print(f"📊 训练集(混合): {total_train} 张 | 验证(混合): {total_val} 张 | 验证(旧): {len(val_old) if val_old else 0} | 验证(新): {len(val_new) if val_new else 0}")
    else:
        # 回退：原先混合列表或默认数据
        if not image_dirs:
            image_dirs = ["freespace_dataset/images"]
        if not mask_dirs:
            mask_dirs = ["freespace_dataset/masks"]
        full_dataset = build_concat_dataset(image_dirs, mask_dirs, augment=True)
        n = len(full_dataset)
        indices = list(range(n))
        split = int(0.8 * n)
        train_indices = indices[:split]
        val_indices = indices[split:]
        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        val_full = build_concat_dataset(image_dirs, mask_dirs, augment=False)
        val_subset = torch.utils.data.Subset(val_full, val_indices)
        train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
        val_loader_mixed = DataLoader(val_subset, batch_size=4, shuffle=False)
        val_loader_old = None
        val_loader_new = None
        print(f"📊 训练集: {len(train_subset)} 张 | 验证集: {len(val_subset)} 张 (总计: {n})")

    best_val_iou_mixed = -1.0

    for epoch in range(start_epoch, total_epochs):
        if freeze_encoder_epochs > 0 and (epoch - start_epoch) < freeze_encoder_epochs:
            freeze_encoder(model, True)
            if (epoch - start_epoch) == 0:
                print(f"🧊 冻结Encoder参数 {freeze_encoder_epochs} 个epoch 以稳定微调")
        else:
            freeze_encoder(model, False)

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
        def eval_loader(dl):
            if dl is None or len(dl) == 0:
                return float('nan'), float('nan'), float('nan'), float('nan')
            viou = vdice = vacc = vloss = 0.0
            with torch.no_grad():
                for images, masks in dl:
                    images, masks = images.to(device), masks.to(device)
                    preds = model(images)
                    loss = criterion(preds, masks)
                    vloss += loss.item()
                    viou += calculate_iou_logits(preds, masks)
                    vdice += calculate_dice_logits(preds, masks)
                    vacc += calculate_pixel_accuracy_logits(preds, masks)
            n = len(dl)
            return vloss / n, viou / n, vdice / n, vacc / n

        val_loss_m, val_iou_m, val_dice_m, val_acc_m = eval_loader(val_loader_mixed)
        val_loss_o, val_iou_o, val_dice_o, val_acc_o = eval_loader(val_loader_old)
        val_loss_n, val_iou_n, val_dice_n, val_acc_n = eval_loader(val_loader_new)
        
        print(f"[{epoch+1}/{total_epochs}] Train Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f} | Acc: {avg_acc:.4f}")
        print(f"                 Val(mix) Loss: {val_loss_m:.4f} | IoU: {val_iou_m:.4f} | Dice: {val_dice_m:.4f} | Acc: {val_acc_m:.4f}")
        if not np.isnan(val_iou_o):
            print(f"                 Val(old) Loss: {val_loss_o:.4f} | IoU: {val_iou_o:.4f} | Dice: {val_dice_o:.4f} | Acc: {val_acc_o:.4f}")
        if not np.isnan(val_iou_n):
            print(f"                 Val(new) Loss: {val_loss_n:.4f} | IoU: {val_iou_n:.4f} | Dice: {val_dice_n:.4f} | Acc: {val_acc_n:.4f}")

        # 保存最佳（以混合验证IoU为准）
        if not np.isnan(val_iou_m) and val_iou_m > best_val_iou_mixed:
            best_val_iou_mixed = val_iou_m
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
                'val_mixed': {
                    'loss': val_loss_m,
                    'iou': val_iou_m,
                    'dice': val_dice_m,
                    'acc': val_acc_m
                },
                'val_old': {
                    'loss': val_loss_o,
                    'iou': val_iou_o,
                    'dice': val_dice_o,
                    'acc': val_acc_o
                },
                'val_new': {
                    'loss': val_loss_n,
                    'iou': val_iou_n,
                    'dice': val_dice_n,
                    'acc': val_acc_n
                }
            }, best_path)
            print(f"🏆 更新最佳模型(Val-mix IoU={val_iou_m:.4f}) → {best_path}")
        
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
                'val_mixed': {
                    'loss': val_loss_m,
                    'iou': val_iou_m,
                    'dice': val_dice_m,
                    'acc': val_acc_m
                },
                'val_old': {
                    'loss': val_loss_o,
                    'iou': val_iou_o,
                    'dice': val_dice_o,
                    'acc': val_acc_o
                },
                'val_new': {
                    'loss': val_loss_n,
                    'iou': val_iou_n,
                    'dice': val_dice_n,
                    'acc': val_acc_n
                }
            }, checkpoint_path)
            print(f"💾 保存检查点: {checkpoint_path}")

    # 训练完成，保存最终模型
    os.makedirs("runs", exist_ok=True)
    final_model_path = "runs/freespace_model.pth"
    torch.save({
        'final_epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'best_val_iou_mixed': best_val_iou_mixed,
        'final_metrics': {
            'loss': avg_loss,
            'iou': avg_iou,
            'dice': avg_dice,
            'acc': avg_acc,
            'val_mixed_iou': val_iou_m,
            'val_old_iou': val_iou_o,
            'val_new_iou': val_iou_n
        }
    }, final_model_path)

    print(f"\n🎉 训练完成！")
    print(f"📁 最终模型: {final_model_path}")
    print(f"🏆 最佳混合验证IoU: {best_val_iou_mixed:.4f}")
    print(f"📊 最终指标:")
    print(f"   Train Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f} | Acc: {avg_acc:.4f}")
    print(f"   Val(mix)  Loss: {val_loss_m:.4f} | IoU: {val_iou_m:.4f} | Dice: {val_dice_m:.4f} | Acc: {val_acc_m:.4f}")
    if not np.isnan(val_iou_o):
        print(f"   Val(old)  Loss: {val_loss_o:.4f} | IoU: {val_iou_o:.4f} | Dice: {val_dice_o:.4f} | Acc: {val_acc_o:.4f}")
    if not np.isnan(val_iou_n):
        print(f"   Val(new)  Loss: {val_loss_n:.4f} | IoU: {val_iou_n:.4f} | Dice: {val_dice_n:.4f} | Acc: {val_acc_n:.4f}")

    for i in range(save_interval, total_epochs + 1, save_interval):
        checkpoint_path = f"runs/checkpoint_epoch_{i}.pth"
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"🗑️ 删除检查点: {checkpoint_path}")

    return final_model_path

# 开始训练
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 训练总轮数（默认50）。增量/微调场景下一样生效：会从起始epoch继续到该轮数
    parser.add_argument('--epochs', type=int, default=50)

    # 周期性保存间隔（单位：epoch）。仅用于中途检查点；训练结束会清理这些检查点
    parser.add_argument('--save_interval', type=int, default=10)

    # 恢复训练/微调的权重路径（.pth）。可为 runs/best_model_val_iou.pth 或自定义
    parser.add_argument('--resume_from', type=str, default=None, help='checkpoint path to resume/finetune from')

    # 是否在断点续训时一并恢复优化器状态（动量/学习率等）。仅断点续训建议开启，微调一般关闭
    parser.add_argument('--resume_optimizer', action='store_true', help='resume optimizer state when resuming')

    # 微调模式：仅加载模型权重，重建优化器，以较小学习率在新数据上继续训练
    parser.add_argument('--finetune', action='store_true', help='finetune on new data (load weights only)')

    # 微调学习率（默认1e-5）。与 --finetune 搭配使用
    parser.add_argument('--finetune_lr', type=float, default=1e-5, help='lr for finetune')

    # 微调时可先冻结编码器若干轮（默认0），稳定特征再解冻。典型设置：2~5
    parser.add_argument('--freeze_encoder_epochs', type=int, default=0, help='freeze encoder for first N epochs')

    # 统一混合训练模式：可传入多个图像/掩码目录进行合并训练（未提供 old/new 时生效）
    parser.add_argument('--image_dirs', nargs='+', type=str, default=None, help='one or more image directories')
    parser.add_argument('--mask_dirs', nargs='+', type=str, default=None, help='one or more mask directories')

    # 分域混合训练：分别指定旧域与新域数据目录，用于“新数据为主+旧数据回放”的持续学习范式
    parser.add_argument('--old_image_dirs', nargs='+', type=str, default=None, help='old domain image dirs')
    parser.add_argument('--old_mask_dirs', nargs='+', type=str, default=None, help='old domain mask dirs')
    parser.add_argument('--new_image_dirs', nargs='+', type=str, default=None, help='new domain image dirs')
    parser.add_argument('--new_mask_dirs', nargs='+', type=str, default=None, help='new domain mask dirs')
    
    # 新域采样占比（0~1，默认0.8）。仅在提供 old/new 目录时生效，用于加权采样新域样本
    parser.add_argument('--new_ratio', type=float, default=0.8, help='sampling ratio for new domain in training (0~1)')
    
    # 分层学习率参数
    parser.add_argument('--encoder_lr', type=float, default=1e-5, help='learning rate for encoder (pretrained parts)')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder (new training parts)')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for all parameters')

    # 使用示例：
    # 1) 断点续训（同一数据继续训练，恢复优化器）
    #    python3 train.py --epochs 50 \
    #        --resume_from runs/best_model_val_iou.pth --resume_optimizer
    # 2) 新数据微调（小学习率 + 冻结编码器前2轮），新域占比80%，混入旧数据回放20%
    #    python3 train.py \
    #        --old_image_dirs freespace_dataset/images \
    #        --old_mask_dirs freespace_dataset/masks \
    #        --new_image_dirs NEW/images \
    #        --new_mask_dirs NEW/masks \
    #        --new_ratio 0.8 \
    #        --resume_from runs/best_model_val_iou.pth \
    #        --finetune --finetune_lr 1e-5 --freeze_encoder_epochs 2
    # 3) 分层学习率训练（编码器小学习率，解码器大学习率）
    #    python3 train.py --epochs 50 \
    #        --encoder_lr 1e-5 --decoder_lr 1e-4 --weight_decay 1e-4
    # 4) 微调时使用分层学习率
    #    python3 train.py --epochs 30 \
    #        --resume_from runs/best_model_val_iou.pth \
    #        --finetune --finetune_lr 1e-5 \
    #        --encoder_lr 1e-6 --decoder_lr 1e-5 --weight_decay 1e-4

    args = parser.parse_args()

    final_model = train_all_data(
        total_epochs=args.epochs,
        save_interval=args.save_interval,
        resume_from=args.resume_from,
        resume_optimizer=args.resume_optimizer,
        finetune=args.finetune,
        finetune_lr=args.finetune_lr,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        image_dirs=args.image_dirs,
        mask_dirs=args.mask_dirs,
        old_image_dirs=args.old_image_dirs,
        old_mask_dirs=args.old_mask_dirs,
        new_image_dirs=args.new_image_dirs,
        new_mask_dirs=args.new_mask_dirs,
        new_ratio=args.new_ratio,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        weight_decay=args.weight_decay,
    )
