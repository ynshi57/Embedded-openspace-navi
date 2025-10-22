import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from model_mobilenet_unet import MobileNetV2_UNet

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # # [B,C,H,W] -> [B,C,1,1]
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()           # [B,C,H,W]
        y = self.avg_pool(x).view(b, c) # [B,C,1,1] -> [B,C]
        y = self.fc(y).view(b, c, 1, 1) # [B,C] -> [B,C,1,1]
        return x * y                    # [B,C,H,W] * [B,C,1,1] = [B,C,H,W]

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            ConvRelu(in_channels, mid_channels),
            ConvRelu(mid_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)

class FeatureAlign(nn.Module):
    """特征对齐模块：双线性插值 + 1x1卷积修正"""
    def __init__(self, in_channels, out_channels):
        super(FeatureAlign, self).__init__()
        self.align_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, target_size):
        # 双线性插值
        aligned = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        # 1x1卷积修正
        aligned = self.align_conv(aligned)
        aligned = self.bn(aligned)
        aligned = self.relu(aligned)
        return aligned

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_se=True):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 添加SE模块到跳跃连接处
        self.use_se = use_se
        if use_se:
            self.se_skip = SEBlock(skip_channels, reduction=16)
            self.se_conv = SEBlock(out_channels, reduction=16)

    def forward(self, x, skip):
        x = self.up(x)
        # 假设尺寸严格匹配
        assert x.shape[2:] == skip.shape[2:], f"Spatial mismatch: {x.shape[2:]} vs {skip.shape[2:]}"
        
        # 对跳跃连接应用SE模块
        if self.use_se:
            skip = self.se_skip(skip)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        # 对卷积结果应用SE模块
        if self.use_se:
            x = self.se_conv(x)
            
        return x

class MobileNetV2_UNet_SE(nn.Module):
    def __init__(self, use_se=True, se_reduction=16):
        super(MobileNetV2_UNet_SE, self).__init__()
        self.use_se = use_se
        
        num_classes = 2
        # 加载预训练的MobileNetV2主干网络
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.enc0 = backbone[0]            # Conv+BN+ReLU
        self.enc1 = backbone[1:3]          # 24
        self.enc2 = backbone[3:6]          # 32
        self.enc3 = backbone[6:10]         # 64
        self.enc4 = backbone[10:14]        # 96
        self.enc5 = backbone[14:]          # 1280

        # 在编码器关键层添加SE模块
        if use_se:
            self.se_e3 = SEBlock(64, reduction=se_reduction)
            self.se_e4 = SEBlock(96, reduction=se_reduction)
            self.se_e5 = SEBlock(1280, reduction=se_reduction)

        # 解码器（带SE模块）
        self.up1 = UpBlock(1280, 96, 256, use_se=use_se)  # 7→14 with e4
        self.up2 = UpBlock(256, 32, 128, use_se=use_se)   # 14→28 with e2
        self.up3 = UpBlock(128, 24, 64, use_se=use_se)    # 28→56 with e1
        self.up4 = UpBlock(64, 32, 32, use_se=use_se)     # 56→112 with e0
        
        # final upsample without skip: 112→224
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # 保存输入尺寸用于最终输出
        input_size = x.shape[2:] #[B, 3, 224, 224]
        
        # 编码器部分
        e0 = self.enc0(x)      # [B, 32, 112, 112] - 第一次下采样: 224→112
        e1 = self.enc1(e0)     # [B, 24, 56, 56]   - 第二次下采样: 112→56
        e2 = self.enc2(e1)     # [B, 32, 28, 28]   - 第三次下采样: 56→28
        e3 = self.enc3(e2)     # [B, 64, 14, 14]   - 第四次下采样: 28→14
        e4 = self.enc4(e3)     # [B, 96, 14, 14]   - 保持尺寸: 14×14
        e5 = self.enc5(e4)     # [B, 1280, 7, 7]   - 第五次下采样: 14→7

        # 在关键编码器层应用SE模块
        if self.use_se:
            e3 = self.se_e3(e3)
            e4 = self.se_e4(e4)
            e5 = self.se_e5(e5)

        # 解码器部分
        d1 = self.up1(e5, e4)   # 7→14, skip e4(14)
        d2 = self.up2(d1, e2)   # 14→28, skip e2(28)
        d3 = self.up3(d2, e1)   # 28→56, skip e1(56)
        d4 = self.up4(d3, e0)   # 56→112, skip e0(112)
        d5 = self.final_up(d4)  # 112→224

        out = self.out_conv(d5)  # [B, num_classes, 224, 224]
        
        # Ensure logits match the input spatial size
        assert out.shape[2:] == input_size, f"Spatial mismatch: {out.shape[2:]} vs {input_size}"
        
        return out

def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == "__main__":
    # 测试模型（CPU版本）
    # 原始模型
    model_original = MobileNetV2_UNet()
    total_orig, trainable_orig = count_parameters(model_original)
    
    # SE模型
    model_se = MobileNetV2_UNet_SE(use_se=True)
    total_se, trainable_se = count_parameters(model_se)
    
    print(f"原始模型参数: {total_orig:,} (可训练: {trainable_orig:,})")
    print(f"SE模型参数: {total_se:,} (可训练: {trainable_se:,})")
    print(f"参数增加: {total_se - total_orig:,} ({((total_se - total_orig) / total_orig * 100):.2f}%)")
    
    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        out_orig = model_original(x)
        out_se = model_se(x)
        
    print(f"原始模型输出形状: {out_orig.shape}")
    print(f"SE模型输出形状: {out_se.shape}")
    
    # 计算推理时间（CPU版本）
    import time
    model_original.eval()
    model_se.eval()
    
    for _ in range(10):
        _ = model_original(x)
        _ = model_se(x)

    def benchmark_forward(model, inp, iters=100):
        start_time = time.time()
        for _ in range(iters):
            _ = model(inp)
        return (time.time() - start_time) / iters

    time_orig = benchmark_forward(model_original, x, iters=30)
    time_se = benchmark_forward(model_se, x, iters=30)
    
    print(f"推理时间 - 原始模型: {time_orig*1000:.2f}ms")
    print(f"推理时间 - SE模型: {time_se*1000:.2f}ms")
    print(f"时间增加: {((time_se - time_orig) / time_orig * 100):.2f}%")
