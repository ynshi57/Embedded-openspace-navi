# predict.py
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from model_mobilenet_unet import MobileNetV2_UNet
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.transforms as T

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
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

def visualize_predictions(image, true_mask, pred1_prob, pred1_mask, pred2_prob, pred2_mask, 
                         model1_name, model2_name):
    """可视化两个模型的预测结果对比"""
    
    # 转换tensor为numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        # 反归一化
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
    
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.squeeze().cpu().numpy()
    
    pred1_prob = pred1_prob.squeeze().cpu().numpy()
    pred1_mask = pred1_mask.squeeze().cpu().numpy()
    pred2_prob = pred2_prob.squeeze().cpu().numpy()
    pred2_mask = pred2_mask.squeeze().cpu().numpy()
    
    # 创建可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 第一行：原始图像和真实标签
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(true_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # 第二行：两个模型的预测结果
    axes[0, 2].imshow(pred1_prob, cmap='hot')
    axes[0, 2].set_title(f'{model1_name}\nRoad Probability')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(pred1_mask, cmap='gray')
    axes[0, 3].set_title(f'{model1_name}\nBinary Mask')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(pred2_prob, cmap='hot')
    axes[1, 0].set_title(f'{model2_name}\nRoad Probability')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred2_mask, cmap='gray')
    axes[1, 1].set_title(f'{model2_name}\nBinary Mask')
    axes[1, 1].axis('off')
    
    # 差异对比
    diff_mask = np.abs(pred1_mask - pred2_mask)
    axes[1, 2].imshow(diff_mask, cmap='Reds')
    axes[1, 2].set_title('Prediction Difference\n(Red = Different)')
    axes[1, 2].axis('off')
    
    # 统计信息
    axes[1, 3].text(0.1, 0.8, f'Model 1 ({model1_name}):\nRoad pixels: {pred1_mask.sum():.0f}\nConfidence: {pred1_prob.mean():.3f}', 
                     transform=axes[1, 3].transAxes, fontsize=10, verticalalignment='top')
    axes[1, 3].text(0.1, 0.4, f'Model 2 ({model2_name}):\nRoad pixels: {pred2_mask.sum():.0f}\nConfidence: {pred2_prob.mean():.3f}', 
                     transform=axes[1, 3].transAxes, fontsize=10, verticalalignment='top')
    axes[1, 3].text(0.1, 0.1, f'Difference:\n{100*diff_mask.sum()/diff_mask.size:.1f}% pixels differ', 
                     transform=axes[1, 3].transAxes, fontsize=10, verticalalignment='top')
    axes[1, 3].set_title('Statistics')
    axes[1, 3].axis('off')
    
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
    image_path = "freespace_dataset/images/0025.png"
    mask_path = "freespace_dataset/masks/0025.png"
    
    if not os.path.exists(image_path):
        print(f"❌ 测试图像不存在: {image_path}")
        return
    
    print(f"\n🖼️ 测试图像: {image_path}")
    
    # 预处理图像
    img_tensor, original_image = preprocess_image(image_path)
    print(f"📐 输入图像尺寸: {img_tensor.shape}")
    
    # 读取真实标签
    mask = Image.open(mask_path).convert("L")
    mask_tensor = ToTensor()(mask)
    print(f"🎯 真实标签尺寸: {mask_tensor.shape}")
    
    # 模型1预测（最佳验证模型）
    print(f"\n🔮 使用最佳验证模型进行预测...")
    with torch.no_grad():
        pred1 = best_model(img_tensor)
        print(f"📊 预测输出尺寸: {pred1.shape}")
        print(f"📊 预测值范围: [{pred1.min():.4f}, {pred1.max():.4f}]")
        
        pred1_prob, pred1_mask = postprocess_prediction(pred1)
        print(f"🛣️ 道路像素数量: {pred1_mask.sum().item():.0f}")
        print(f"🎯 道路概率均值: {pred1_prob.mean().item():.4f}")
    
    # 模型2预测（最终训练模型）
    print(f"\n🔮 使用最终训练模型进行预测...")
    with torch.no_grad():
        pred2 = final_model(img_tensor)
        print(f"📊 预测输出尺寸: {pred2.shape}")
        print(f"📊 预测值范围: [{pred2.min():.4f}, {pred2.max():.4f}]")
        
        pred2_prob, pred2_mask = postprocess_prediction(pred2)
        print(f"🛣️ 道路像素数量: {pred2_mask.sum().item():.0f}")
        print(f"🎯 道路概率均值: {pred2_prob.mean().item():.4f}")
    
    # 可视化对比
    print(f"\n🎨 生成可视化对比...")
    visualize_predictions(
        original_image, mask_tensor, 
        pred1_prob, pred1_mask, 
        pred2_prob, pred2_mask,
        "Best Val IoU", "Final Training"
    )
    
    # 模型性能对比
    print(f"\n📊 模型性能对比:")
    print(f"最佳验证模型 - 验证IoU: {best_checkpoint.get('val_metrics', {}).get('iou', 'N/A')}")
    print(f"最终训练模型 - 验证IoU: {best_checkpoint.get('final_metrics', {}).get('val_iou', 'N/A')}")

if __name__ == "__main__":
    main()
