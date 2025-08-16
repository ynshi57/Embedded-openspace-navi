
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
        # 加载预训练的MobileNetV2主干网络
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.enc0 = backbone[0]            # Conv+BN+ReLU
        self.enc1 = backbone[1:3]          # 24
        self.enc2 = backbone[3:6]          # 32
        self.enc3 = backbone[6:10]         # 64
        self.enc4 = backbone[10:14]        # 96
        self.enc5 = backbone[14:]          # 1280

        # 特征对齐模块
        self.align4 = FeatureAlign(96, 96)   # d4 -> e4
        self.align3 = FeatureAlign(64, 64)   # d3 -> e3
        self.align2 = FeatureAlign(32, 32)   # d2 -> e2
        self.align1 = FeatureAlign(24, 24)   # d1 -> e1
        self.align0 = FeatureAlign(16, 16)   # d0 -> e0

        # 解码器：每一步上采样 + concat + 卷积
        self.up4 = nn.ConvTranspose2d(1280, 96, kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(96 + 96, 96, 64)

        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(64 + 64, 64, 32)

        self.up2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(32 + 32, 32, 24)

        self.up1 = nn.ConvTranspose2d(24, 24, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(24 + 24, 24, 16)

        self.up0 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.dec0 = DecoderBlock(16 + 32, 16, 8)  # enc0输出是32通道

        # 最终上采样：从112x112恢复到224x224
        self.final_upsample = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)

        # 最终输出：通道数为1，sigmoid用于二分类
        self.final = nn.Conv2d(8, 1, kernel_size=1)

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
        d4 = self.up4(e5)
        # 特征对齐：确保d4和e4尺寸匹配
        if d4.shape[2:] != e4.shape[2:]:
            d4 = self.align4(d4, e4.shape[2:])
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        # 特征对齐：确保d3和e3尺寸匹配
        if d3.shape[2:] != e3.shape[2:]:
            d3 = self.align3(d3, e3.shape[2:])
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        # 特征对齐：确保d2和e2尺寸匹配
        if d2.shape[2:] != e2.shape[2:]:
            d2 = self.align2(d2, e2.shape[2:])
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        # 特征对齐：确保d1和e1尺寸匹配
        if d1.shape[2:] != e1.shape[2:]:
            d1 = self.align1(d1, e1.shape[2:])
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        # 特征对齐：确保d0和e0尺寸匹配
        if d0.shape[2:] != e0.shape[2:]:
            d0 = self.align0(d0, e0.shape[2:])
        d0 = torch.cat([d0, e0], dim=1)
        d0 = self.dec0(d0)

        # 最终上采样：从112x112恢复到224x224
        d0 = self.final_upsample(d0)  # [B, 8, 224, 224]

        out = self.final(d0)
        out = torch.sigmoid(out)
        
        # # 确保输出尺寸与输入尺寸匹配
        # if out.shape[2:] != input_size:
        #     raise ValueError(f"输出尺寸不匹配: {out.shape[2:]} vs 输入 {input_size}")
        
        return out
