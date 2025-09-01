#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model_mobilenet_unet import MobileNetV2_UNet

def analyze_model_parameters():
    """分析模型参数分布"""
    model = MobileNetV2_UNet()
    
    print("MobileNetV2-UNet 模型参数分析")
    print("=" * 60)
    
    # 统计各层参数
    encoder_params = 0
    decoder_params = 0
    total_params = 0
    
    print("\n 编码器部分 (MobileNetV2 Backbone):")
    print("-" * 40)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            
            if 'enc' in name:
                encoder_params += param_count
                print(f"  {name:20s}: {param_count:8d} 参数 ({param.shape})")
            else:
                decoder_params += param_count
                if 'up' in name or 'final_up' in name or 'out_conv' in name:
                    print(f"  {name:20s}: {param_count:8d} 参数 ({param.shape})")
    
    print(" 解码器部分 (UNet Decoder):")
    print("-" * 40)
    for name, param in model.named_parameters():
        if param.requires_grad and ('up' in name or 'final_up' in name or 'out_conv' in name):
            param_count = param.numel()
            print(f"  {name:20s}: {param_count:8d} 参数 ({param.shape})")
    
    print(" 参数统计:")
    print("-" * 40)
    print(f"  编码器参数总数: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"  解码器参数总数: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    print(f"  模型参数总数:   {total_params:,}")
    
    # 详细分析编码器各层
    print(" 编码器详细分析:")
    print("-" * 40)
    enc_layers = {
        'enc0': 0, 'enc1': 0, 'enc2': 0, 
        'enc3': 0, 'enc4': 0, 'enc5': 0
    }
    
    for name, param in model.named_parameters():
        if param.requires_grad and 'enc' in name:
            for layer in enc_layers:
                if layer in name:
                    enc_layers[layer] += param.numel()
                    break
    
    for layer, count in enc_layers.items():
        print(f"  {layer:8s}: {count:8d} 参数")
    
    # 验证参数数量
    print(f"\n 验证: 编码器参数总和 = {sum(enc_layers.values()):,}")

if __name__ == "__main__":
    analyze_model_parameters()