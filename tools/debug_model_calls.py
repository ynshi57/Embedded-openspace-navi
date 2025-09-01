import torch
import torch.nn as nn
from model_mobilenet_unet import MobileNetV2_UNet

# 创建一个简化的模型来演示调用逻辑
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(16, 1, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)  # 只在训练模式下生效
        x = self.conv2(x)
        return torch.sigmoid(x)

# 演示 model.train() 的底层逻辑
def demonstrate_train_mode():
    print("=== model.train() 底层调用演示 ===")
    
    model = SimpleModel()
    print(f"初始状态 - training: {model.training}")
    
    # 检查各个子模块的状态
    print(f"conv1.training: {model.conv1.training}")
    print(f"bn1.training: {model.bn1.training}")
    print(f"dropout.training: {model.dropout.training}")
    
    # 调用 model.train()
    model.train()
    print(f"\n调用 model.train() 后:")
    print(f"model.training: {model.training}")
    print(f"conv1.training: {model.conv1.training}")
    print(f"bn1.training: {model.bn1.training}")
    print(f"dropout.training: {model.dropout.training}")
    
    # 演示 dropout 在训练和评估模式下的差异
    x = torch.randn(1, 3, 64, 64)
    
    print(f"\n=== Dropout 行为演示 ===")
    model.train()
    output_train = model(x)
    print(f"训练模式输出非零元素数量: {(output_train > 0).sum().item()}")
    
    model.eval()
    output_eval = model(x)
    print(f"评估模式输出非零元素数量: {(output_eval > 0).sum().item()}")

# 演示 model(images) 的调用链
def demonstrate_forward_call():
    print("\n=== model(images) 调用链演示 ===")
    
    # 使用我们的实际模型
    model = MobileNetV2_UNet()
    x = torch.randn(1, 3, 256, 256)
    
    print("调用 model(x) 时:")
    print("1. Python 调用 model.__call__(x)")
    print("2. nn.Module.__call__() 被调用")
    print("3. 检查 self.training 状态")
    print("4. 调用 self.forward(x)")
    print("5. 返回 forward() 的结果")
    
    # 实际调用
    with torch.no_grad():  # 避免计算梯度
        output = model(x)
        print(f"\n实际输出形状: {output.shape}")
        print(f"输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")

# 演示 nn.Module 的 __call__ 方法
def demonstrate_module_call():
    print("\n=== nn.Module.__call__ 方法演示 ===")
    
    class DebugModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.called = False
            
        def forward(self, x):
            self.called = True
            print(f"  forward() 被调用，输入形状: {x.shape}")
            return x * 2
    
    model = DebugModule()
    print(f"调用前 - called: {model.called}")
    
    x = torch.randn(2, 3)
    result = model(x)  # 这会调用 __call__
    
    print(f"调用后 - called: {model.called}")
    print(f"结果形状: {result.shape}")

if __name__ == "__main__":
    demonstrate_train_mode()
    demonstrate_forward_call()
    demonstrate_module_call() 