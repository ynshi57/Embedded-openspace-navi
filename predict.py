# predict.py
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from model_mobilenet_unet import MobileNetV2_UNet
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 使用设备:", device)

def load_model(model_path, device):
    """加载模型并返回模型和检查点信息"""
    model = MobileNetV2_UNet().to(device)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None, None
    
    print(f"📁 加载模型: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            # 新格式：包含额外信息的字典
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 模型加载成功！")
            print(f"📊 训练轮次: {checkpoint.get('final_epoch', checkpoint.get('epoch', 'N/A'))}")
            
            # 显示训练指标
            if 'final_metrics' in checkpoint:
                metrics = checkpoint['final_metrics']
                print(f"📈 最终指标 - IoU: {metrics.get('iou', 'N/A'):.4f}, "
                      f"Dice: {metrics.get('dice', 'N/A'):.4f}, "
                      f"Acc: {metrics.get('acc', 'N/A'):.4f}")
                if 'val_iou' in metrics:
                    print(f"🏆 验证IoU: {metrics.get('val_iou', 'N/A'):.4f}")
            elif 'val_metrics' in checkpoint:
                val_metrics = checkpoint['val_metrics']
                print(f"🏆 验证指标 - IoU: {val_metrics.get('iou', 'N/A'):.4f}, "
                      f"Dice: {val_metrics.get('dice', 'N/A'):.4f}, "
                      f"Acc: {val_metrics.get('acc', 'N/A'):.4f}")
        else:
            # 旧格式：直接是state_dict
            model.load_state_dict(checkpoint)
            print("✅ 模型加载成功！（旧格式）")
        
        model.eval()
        return model, checkpoint
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def preprocess_image(image_path, target_size=(224, 224)):
    """预处理图像，返回tensor和原始图像"""
    
    # 图像预处理（与训练时一致）
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagNet standard normalization
    ])
    
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    return img_tensor, image

def postprocess_prediction(pred_logits):
    """后处理预测结果：logits -> 概率 -> 二值化mask"""
    # 应用softmax得到概率
    probs = torch.softmax(pred_logits, dim=1)
    
    # 取道路类（第1类）的概率
    road_prob = probs[:, 1:2, :, :]  # [B, 1, H, W]
    
    # 二值化（阈值0.5）
    road_mask = (road_prob > 0.5).float()
    
    return road_prob, road_mask

def load_gt_mask_binary(mask_path, target_size=(224, 224)):
    """
    读取原始彩色GT mask并转换为二值道路掩码，遵循训练时的颜色映射：
      - 红色(255,0,0) -> 背景 0
      - 粉色/洋红(255,0,255) -> 道路 1
      - 蓝色(0,0,255) -> 背景 0
      - 黑色(0,0,0) -> 忽略 255
    返回: (gt_mask, valid_mask)
      gt_mask: np.ndarray[H,W], 值∈{0,1,255}
      valid_mask: np.ndarray[H,W], 布尔数组，True表示可参与评估的位置
    """
    m = Image.open(mask_path).convert("RGB")
    m = m.resize(target_size, Image.NEAREST)
    m = np.array(m)
    gt = np.zeros(m.shape[:2], dtype=np.uint8)
    # 背景 红
    gt[(m[:, :, 0] == 255) & (m[:, :, 1] == 0) & (m[:, :, 2] == 0)] = 0
    # 道路 洋红/粉
    gt[(m[:, :, 0] == 255) & (m[:, :, 1] == 0) & (m[:, :, 2] == 255)] = 1
    # 蓝色 -> 背景
    gt[(m[:, :, 0] == 0) & (m[:, :, 1] == 0) & (m[:, :, 2] == 255)] = 0
    # 黑色 -> 忽略
    gt[(m[:, :, 0] == 0) & (m[:, :, 1] == 0) & (m[:, :, 2] == 0)] = 255
    valid = gt != 255
    return gt, valid

def visualize_predictions(image, true_mask, 
                        pred1_prob, pred1_mask, 
                        pred2_prob, pred2_mask, 
                        model1_name, model2_name):
    """
    参数:
        image: 输入图像(tensor[C,H,W], 已Normalize; 或PIL/ndarray).
        true_mask: 真实标签mask(tensor[1,H,W] 或 ndarray[H,W]).
        pred1_prob: 模型1的道路类概率图(tensor[B,1,H,W]).
        pred1_mask: 模型1的二值化道路mask(tensor[B,1,H,W], 阈值后0/1).
        pred2_prob: 模型2的道路类概率图(tensor[B,1,H,W]).
        pred2_mask: 模型2的二值化道路mask(tensor[B,1,H,W]).
        model1_name: 模型1名称, 用于标题显示.
        model2_name: 模型2名称, 用于标题显示.
    """
    
    if isinstance(image, torch.Tensor):
        # 是 ToTensor() 的逆操作,将 PyTorch Tensor 转换回可显示的图像格式。
        image = image.permute(1, 2, 0).cpu().numpy()
        # 反标准化 x_original = x_normalized * std + mean
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        # 将image限制在[0,1]之间,由于浮点运算误差和标准化过程,反标准化后的值可能会略微超出 [0, 1] 范围
        image = np.clip(image, 0, 1)
    
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.squeeze().cpu().numpy()
    
    #  从 [B, 1, H, W] 压到 [H, W]
    pred1_prob = pred1_prob.squeeze().cpu().numpy()
    pred1_mask = pred1_mask.squeeze().cpu().numpy()
    pred2_prob = pred2_prob.squeeze().cpu().numpy()
    pred2_mask = pred2_mask.squeeze().cpu().numpy()
    
    # 计算评估指标（与GT对齐）。支持true_mask中含有255为忽略像素
    # 统一将预测与GT转为布尔并按valid区域统计
    def compute_metrics(pred_bin: np.ndarray, gt: np.ndarray):
        valid = gt != 255
        if valid.sum() == 0:
            return np.nan, np.nan, np.nan, np.nan
        pred_b = pred_bin.astype(bool) & valid
        gt_b = (gt == 1) & valid
        tp = np.logical_and(pred_b, gt_b).sum()
        fp = np.logical_and(pred_b, np.logical_not(gt_b)).sum()
        fn = np.logical_and(np.logical_not(pred_b), gt_b).sum()
        union = tp + fp + fn
        iou = tp / (union + 1e-6)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        return float(iou), float(dice), float(precision), float(recall)

    iou1, dice1, prec1, rec1 = compute_metrics(pred1_mask, true_mask)
    iou2, dice2, prec2, rec2 = compute_metrics(pred2_mask, true_mask)

    # 计算与GT的差异图（忽略像素不计入差异）
    gt_bin = (true_mask == 1).astype(np.uint8)
    valid = (true_mask != 255)
    diff_m1_gt = np.zeros_like(gt_bin)
    diff_m2_gt = np.zeros_like(gt_bin)
    diff_m1_gt[valid] = np.abs(pred1_mask.astype(np.uint8)[valid] - gt_bin[valid])
    diff_m2_gt[valid] = np.abs(pred2_mask.astype(np.uint8)[valid] - gt_bin[valid])

    # 布局：2行5列
    # 第一行：原图 / 真实标签 / 模型1概率图 / 模型1二值mask / 模型1 vs GT 差异
    # 第二行：模型2概率图 / 模型2二值mask / 模型2 vs GT 差异 / 两模型差异 / 统计信息
    fig, axes = plt.subplots(2, 5, figsize=(24, 10), constrained_layout=True)
    
    # 第一行：原始图像和真实标签
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # 为可视化将忽略区域(255)映射为中性灰色，避免看起来像“缺了一块”
    _gt_vis = true_mask.astype(float)
    _gt_vis[true_mask == 255] = 0.5
    axes[0, 1].imshow(_gt_vis, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0, 1].set_title('Ground Truth (gray=ignore)')
    axes[0, 1].axis('off')
    
    # 第二行：两个模型的预测结果
    axes[0, 2].imshow(pred1_prob, cmap='hot')
    axes[0, 2].set_title(f'{model1_name}\nRoad Probability')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(pred1_mask, cmap='gray')
    axes[0, 3].set_title(f'{model1_name}\nBinary Mask')
    axes[0, 3].axis('off')
    
    # 模型1 vs GT 差异（第一行最后一列）
    axes[0, 4].imshow(diff_m1_gt, cmap='Reds')
    axes[0, 4].set_title(f'{model1_name} vs GT\n(Red = Mismatch)')
    axes[0, 4].axis('off')

    axes[1, 0].imshow(pred2_prob, cmap='hot')
    axes[1, 0].set_title(f'{model2_name}\nRoad Probability')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred2_mask, cmap='gray')
    axes[1, 1].set_title(f'{model2_name}\nBinary Mask')
    axes[1, 1].axis('off')

    # 模型2 vs GT 差异（第二行第三列）
    axes[1, 2].imshow(diff_m2_gt, cmap='Reds')
    axes[1, 2].set_title(f'{model2_name} vs GT\n(Red = Mismatch)')
    axes[1, 2].axis('off')

    # 两模型之间差异（第二行第四列）
    diff_mask = np.abs(pred1_mask - pred2_mask)
    axes[1, 3].imshow(diff_mask, cmap='Reds')
    axes[1, 3].set_title('Model1 vs Model2\n(Red = Different)')
    axes[1, 3].axis('off')
    
    # 统计信息：面积与对GT的典型指标（意图：IoU=交并比；Dice≈F1；Prec=查准；Recall=查全）
    area1 = pred1_mask.sum()
    area2 = pred2_mask.sum()
    # 统计信息放在第二行第五列
    ax_stat = axes[1, 4]
    # Model 1 block (top half)
    ax_stat.text(0.06, 0.94, f'Model 1 ({model1_name})', fontsize=10, weight='bold', transform=ax_stat.transAxes)
    ax_stat.text(0.06, 0.86, f'Area: {area1:.0f} px', fontsize=9, transform=ax_stat.transAxes)
    ax_stat.text(0.06, 0.78, f'IoU (overlap): {iou1:.3f}', fontsize=9, transform=ax_stat.transAxes)
    ax_stat.text(0.06, 0.70, f'Dice (F1): {dice1:.3f}', fontsize=9, transform=ax_stat.transAxes)
    ax_stat.text(0.06, 0.62, f'Prec (precision): {prec1:.3f}', fontsize=9, transform=ax_stat.transAxes)
    ax_stat.text(0.06, 0.54, f'Recall (coverage): {rec1:.3f}', fontsize=9, transform=ax_stat.transAxes)

    # Model 2 block (bottom half)
    ax_stat.text(0.06, 0.38, f'Model 2 ({model2_name})', fontsize=10, weight='bold', transform=ax_stat.transAxes)
    ax_stat.text(0.06, 0.30, f'Area: {area2:.0f} px', fontsize=9, transform=ax_stat.transAxes)
    ax_stat.text(0.06, 0.22, f'IoU (overlap): {iou2:.3f}', fontsize=9, transform=ax_stat.transAxes)
    ax_stat.text(0.06, 0.14, f'Dice (F1): {dice2:.3f}', fontsize=9, transform=ax_stat.transAxes)
    ax_stat.text(0.06, 0.06, f'Prec (precision): {prec2:.3f}', fontsize=9, transform=ax_stat.transAxes)
    ax_stat.text(0.06, 0.02, f'Recall (coverage): {rec2:.3f}', fontsize=9, transform=ax_stat.transAxes)
    # 为可读性保留少量底部留白
    ax_stat.set_title('Statistics')
    ax_stat.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数：加载两个模型并进行预测对比"""
    
    # 模型路径
    best_model_path = "runs/best_model_val_iou.pth"
    final_model_path = "runs/freespace_model.pth"
    
    # 加载两个模型
    print("=" * 60)
    print("🔄 加载最佳验证模型...")
    best_model, best_checkpoint = load_model(best_model_path, device)
    
    print("\n" + "=" * 60)
    print("🔄 加载最终训练模型...")
    final_model, final_checkpoint = load_model(final_model_path, device)
    
    if best_model is None or final_model is None:
        print("❌ 无法加载模型，退出")
        return
    
    # 选择测试图像
    image_path = "freespace_dataset/images/0017.png"
    mask_path = "freespace_dataset/masks/0017.png"
    
    if not os.path.exists(image_path):
        print(f"❌ 测试图像不存在: {image_path}")
        return
    
    print(f"\n🖼️ 测试图像: {image_path}")
    
    # 预处理图像
    img_tensor, original_image = preprocess_image(image_path)
    print(f"📐 输入图像尺寸: {img_tensor.shape}")
    
    # 读取真实标签并对齐训练映射，得到 {0,1,255}，其中255为忽略
    gt_np, valid_np = load_gt_mask_binary(mask_path, target_size=(224, 224))
    print(f"🎯 真实标签尺寸: {gt_np.shape}")
    
    # 模型1预测（最佳验证模型）
    print(f"\n🔮 使用最佳验证模型进行预测...")
    with torch.no_grad():
        pred1 = best_model(img_tensor)
        print(f"📊 预测输出尺寸: {pred1.shape}")
        print(f"📊 预测值范围: [{pred1.min():.4f}, {pred1.max():.4f}]")
        
        pred1_prob, pred1_mask = postprocess_prediction(pred1)
        H1, W1 = pred1_mask.shape[-2], pred1_mask.shape[-1]
        area1 = pred1_mask.sum().item()
        ratio1 = area1 / (H1 * W1)
        print(f"🛣️ 道路面积: {area1:.0f} 像素 ({ratio1:.2%})")
    
    # 模型2预测（最终训练模型）
    print(f"\n🔮 使用最终训练模型进行预测...")
    with torch.no_grad():
        pred2 = final_model(img_tensor)
        print(f"📊 预测输出尺寸: {pred2.shape}")
        print(f"📊 预测值范围: [{pred2.min():.4f}, {pred2.max():.4f}]")
        
        pred2_prob, pred2_mask = postprocess_prediction(pred2)
        H2, W2 = pred2_mask.shape[-2], pred2_mask.shape[-1]
        area2 = pred2_mask.sum().item()
        ratio2 = area2 / (H2 * W2)
        print(f"🛣️ 道路面积: {area2:.0f} 像素 ({ratio2:.2%})")
    
    # 可视化对比
    print(f"\n🎨 生成可视化对比...")
    visualize_predictions(
        original_image, gt_np, 
        pred1_prob, pred1_mask, 
        pred2_prob, pred2_mask,
        "Best Val IoU", "Final Training"
    )
    
    # 模型性能对比
    print(f"\n📊 模型性能对比:")
    best_val_iou = None
    if isinstance(best_checkpoint, dict):
        if 'val_mixed' in best_checkpoint:
            best_val_iou = best_checkpoint['val_mixed'].get('iou', None)
        elif 'final_metrics' in best_checkpoint:
            best_val_iou = best_checkpoint['final_metrics'].get('val_mixed_iou', None)
    final_val_iou = None
    if isinstance(final_checkpoint, dict):
        if 'final_metrics' in final_checkpoint:
            final_val_iou = final_checkpoint['final_metrics'].get('val_mixed_iou', None)
        elif 'val_mixed' in final_checkpoint:
            final_val_iou = final_checkpoint['val_mixed'].get('iou', None)
    print(f"最佳验证模型 - 验证IoU: {best_val_iou if best_val_iou is not None else 'N/A'}")
    print(f"最终训练模型 - 验证IoU: {final_val_iou if final_val_iou is not None else 'N/A'}")

if __name__ == "__main__":
    main()
