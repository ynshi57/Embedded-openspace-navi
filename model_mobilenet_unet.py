
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

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
    def __init__(self, in_channels, skip_channels, out_channels):
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

    def forward(self, x, skip):
        x = self.up(x)
        # 假设尺寸严格匹配
        assert x.shape[2:] == skip.shape[2:], f"Spatial mismatch: {x.shape[2:]} vs {skip.shape[2:]}"
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MobileNetV2_UNet(nn.Module):
    def __init__(self):
        super(MobileNetV2_UNet, self).__init__()
        # MobileNetV2(
        #     (features): Sequential(
        #         (0): ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1)

        #         (1): InvertedResidual(32, 16, stride=1, expand_ratio=1)
        #         (2): InvertedResidual(16, 24, stride=2, expand_ratio=6)

        #         (3): InvertedResidual(24, 24, stride=1, expand_ratio=6)
        #         (4): InvertedResidual(24, 32, stride=2, expand_ratio=6)
        #         (5): InvertedResidual(32, 32, stride=1, expand_ratio=6)

        #         (6): InvertedResidual(32, 32, stride=1, expand_ratio=6)
        #         (7): InvertedResidual(32, 64, stride=2, expand_ratio=6)
        #         (8): InvertedResidual(64, 64, stride=1, expand_ratio=6)
        #         (9): InvertedResidual(64, 64, stride=1, expand_ratio=6)

        #         (10): InvertedResidual(64, 64, stride=1, expand_ratio=6)
        #         (11): InvertedResidual(64, 96, stride=1, expand_ratio=6)
        #         (12): InvertedResidual(96, 96, stride=1, expand_ratio=6)
        #         (13): InvertedResidual(96, 96, stride=1, expand_ratio=6)

        #         (14): InvertedResidual(96, 160, stride=2, expand_ratio=6)
        #         (15): InvertedResidual(160, 160, stride=1, expand_ratio=6)
        #         (16): InvertedResidual(160, 160, stride=1, expand_ratio=6)
        #         (17): InvertedResidual(160, 320, stride=1, expand_ratio=6)
        #         (18): ConvBNReLU(320, 1280, kernel_size=1)
        #     )
        # )
        num_classes = 2
        # 加载预训练的MobileNetV2主干网络
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.enc0 = backbone[0]            # Conv+BN+ReLU
        self.enc1 = backbone[1:3]          # 24
        self.enc2 = backbone[3:6]          # 32
        self.enc3 = backbone[6:10]         # 64
        self.enc4 = backbone[10:14]        # 96
        self.enc5 = backbone[14:]          # 1280

        # 解码器
        self.up1 = UpBlock(1280, 96, 256)  # 7→14 with e4
        self.up2 = UpBlock(256, 32, 128)   # 14→28 with e2
        self.up3 = UpBlock(128, 24, 64)    # 28→56 with e1
        self.up4 = UpBlock(64, 32, 32)     # 56→112 with e0
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

        # 解码器部分
        d1 = self.up1(e5, e4)   # 7→14, skip e4(14)
        d2 = self.up2(d1, e2)   # 14→28, skip e2(28)
        d3 = self.up3(d2, e1)   # 28→56, skip e1(56)
        d4 = self.up4(d3, e0)   # 56→112, skip e0(112)
        d5 = self.final_up(d4)  # 112→224

        out = self.out_conv(d5)  # [B, num_classes, 224, 224]
        
        # Ensure logits match the input spatial size
        # if out.shape[2:] != input_size:
            # out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        assert out.shape[2:] == input_size, f"Spatial mismatch: {out.shape[2:]} vs {input_size}"
        
        return out
