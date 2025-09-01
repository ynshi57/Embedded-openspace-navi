# Python模块导入指南

## 问题描述
在 `tools/` 目录下的脚本需要导入上级目录的 `model_mobilenet_unet.py` 模块。

## 解决方案

### 方案1：使用 `sys.path`（推荐）
**文件**: `tools/analyze_model_param.py`

```python
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_mobilenet_unet import MobileNetV2_UNet
```

**优点**:
- ✅ 简单直接
- ✅ 不依赖环境变量
- ✅ 跨平台兼容
- ✅ 脚本可独立运行

**使用方法**:
```bash
python3 tools/analyze_model_param.py
```

### 方案2：相对导入
**文件**: `tools/analyze_model_param_alt.py`

```python
try:
    from ..model_mobilenet_unet import MobileNetV2_UNet
except ImportError:
    # 回退到绝对导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model_mobilenet_unet import MobileNetV2_UNet
```

**优点**:
- ✅ 符合Python包结构规范
- ✅ 支持包内相对导入

**缺点**:
- ❌ 需要从项目根目录运行
- ❌ 相对导入可能失败

**使用方法**:
```bash
# 从项目根目录运行
python3 -m tools.analyze_model_param_alt
```

### 方案3：环境变量PYTHONPATH
**文件**: `tools/analyze_model_param_env.py`

```python
try:
    from model_mobilenet_unet import MobileNetV2_UNet
except ImportError:
    print("❌ 导入失败！请设置PYTHONPATH环境变量")
    exit(1)
```

**优点**:
- ✅ 代码最简洁
- ✅ 符合Python标准做法

**缺点**:
- ❌ 需要设置环境变量
- ❌ 依赖外部配置

**使用方法**:
```bash
# 设置环境变量
export PYTHONPATH=/path/to/project:$PYTHONPATH
python3 tools/analyze_model_param_env.py
```

## 推荐方案

**推荐使用方案1**，因为：
1. 简单可靠
2. 不依赖外部配置
3. 脚本可独立运行
4. 跨平台兼容性好

## 目录结构
```
project/
├── model_mobilenet_unet.py
├── train.py
├── utils.py
└── tools/
    ├── __init__.py
    ├── analyze_model_param.py          # 方案1
    ├── analyze_model_param_alt.py      # 方案2
    ├── analyze_model_param_env.py      # 方案3
    └── README_import_guide.md
```

## 测试命令
```bash
# 测试方案1
python3 tools/analyze_model_param.py

# 测试方案2
python3 -m tools.analyze_model_param_alt

# 测试方案3
export PYTHONPATH=/home/nan/opensource/Embedded-openspace-navi:$PYTHONPATH
python3 tools/analyze_model_param_env.py
``` 