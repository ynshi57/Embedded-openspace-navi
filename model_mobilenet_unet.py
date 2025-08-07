
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

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

class MobileNetV2_UNet(nn.Module):
    def __init__(self):
        super(MobileNetV2_UNet, self).__init__()
        # 加载预训练的MobileNetV2主干网络
        backbone = mobilenet_v2(pretrained=True).features
        self.enc0 = backbone[0]            # Conv+BN+ReLU
        self.enc1 = backbone[1:3]          # 24
        self.enc2 = backbone[3:6]          # 32
        self.enc3 = backbone[6:10]         # 64
        self.enc4 = backbone[10:14]        # 96
        self.enc5 = backbone[14:]          # 1280

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

        # 最终输出：通道数为1，sigmoid用于二分类
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        e0 = self.enc0(x)      # 32通道
        e1 = self.enc1(e0)     # 24通道
        e2 = self.enc2(e1)     # 32通道
        e3 = self.enc3(e2)     # 64通道
        e4 = self.enc4(e3)     # 96通道
        e5 = self.enc5(e4)     # 1280通道

        # 解码器部分
        d4 = self.up4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        d0 = torch.cat([d0, e0], dim=1)
        d0 = self.dec0(d0)

        out = self.final(d0)
        out = torch.sigmoid(out)
        return out
