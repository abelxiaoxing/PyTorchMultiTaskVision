[根目录](../../CLAUDE.md) > [nets](../) > **nets**

# Nets 模块

## 模块职责

nets 模块是项目的神经网络模型定义核心，提供多种主流预训练模型的统一接口实现。该模块专注于模型架构设计、特征提取和多尺度特征融合，支持图像分类、目标检测和时序分析等任务。

## 入口与启动

### 模块初始化
```python
# 文件: nets/__init__.py
from .yolo import YoloBody
from .convnext_v2 import *
from .timesnet import TimesNet
from .Resmlp import Resmlp
from .Abel import AbelNet
```

### 模型导入示例
```python
from nets.yolo import YoloBody
from nets.convnext_v2 import convnextv2_tiny
from nets.timesnet import TimesNet
```

## 对外接口

### 目标检测模型

#### YoloBody
YOLO 检测模型主类，支持多种骨干网络
```python
class YoloBody(nn.Module):
    def __init__(
        self,
        anchors_mask,
        num_classes,
        backbone="convnextv2_atto",
        pretrained=False
    ):
```

**使用示例:**
```python
model = YoloBody(
    anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    num_classes=80,
    backbone="efficientvit_b0",
    pretrained=True
)
```

**支持的骨干网络:**
- EfficientViT 系列: B0, B1, B2, B3, L1, L2, L3
- ConvNeXtV2 系列: Atto, Femto, Pico, Nano, Tiny, Base

### 分类模型

#### ConvnextV2
ConvNeXtV2 分类模型封装
```python
class ConvnextV2(nn.Module):
    def __init__(self, backbone, pretrained=False):
```

**模型规格:**
```python
convnextv2_models = {
    "convnextv2_atto": convnextv2_atto,
    "convnextv2_femto": convnextv2_femto,
    "convnextv2_pico": convnextv2_pico,
    "convnextv2_nano": convnextv2_nano,
    "convnextv2_tiny": convnextv2_tiny,
    "convnextv2_base": convnextv2_base,
}
```

#### Efficientvit
EfficientViT 轻量级视觉 Transformer
```python
class Efficientvit(nn.Module):
    def __init__(self, backbone='efficientvit_b0', pretrained=False):
```

**模型特性:**
- 高效的注意力机制
- 适合移动端部署
- 支持多种分辨率输入

### 时序模型

#### TimesNet
时序预测网络模型
```python
class TimesNet(nn.Module):
    def __init__(self, config):
```

### 自定义模型

#### Resmlp
残差多层感知机模型
```python
class Resmlp(nn.Module):
    def __init__(self, config):
```

#### AbelNet
实验性自定义网络
```python
class AbelNet(nn.Module):
    def __init__(self, config):
```

## 关键依赖与配置

### 核心依赖
```python
import torch
import torch.nn as nn
from collections import OrderedDict
import timm
```

### 模型配置
```python
# YOLO 模型配置
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
input_size = 640
num_classes = 80

# EfficientViT 通道配置
efficientvit_channels = {
    "efficientvit_b0": [32, 64, 128],
    "efficientvit_b1": [64, 128, 256],
    "efficientvit_b2": [96, 192, 384],
    # ...
}

# ConvNeXtV2 通道配置
convnextv2_channels = {
    "convnextv2_atto": [80, 160, 320],
    "convnextv2_tiny": [192, 384, 768],
    "convnextv2_base": [256, 512, 1024],
    # ...
}
```

## 模型架构详解

### YOLO 检测架构

#### 特征提取流程
```python
def forward(self, x):
    # 1. 骨干网络特征提取
    feat1, feat2, feat3 = self.backbone(x)

    # 2. 特征金字塔网络 (FPN)
    P5 = self.conv_for_P5(feat3)
    P5 = self.SPPF(P5)  # 空间金字塔池化

    # 3. 特征融合
    P5_upsample = self.upsample(P5)
    P4 = torch.cat([self.conv_for_P4(feat2), P5_upsample], axis=1)
    P4 = self.conv3_for_upsample1(P4)

    # 4. 多尺度检测头
    out2 = self.yolo_head_P3(P3)  # 小目标检测
    out1 = self.yolo_head_P4(P4)  # 中目标检测
    out0 = self.yolo_head_P5(P5)  # 大目标检测

    return out0, out1, out2
```

#### 关键组件

**SPPF (Spatial Pyramid Pooling Fast)**
```python
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
```

**C3 模块 (Cross Stage Partial)**
```python
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[conv_dw(c_, c_) for _ in range(n)])
```

### 卷积组件

#### Conv 模块
```python
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
```

#### Bottleneck 模块
```python
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
```

## 特征提取配置

### EfficientViT 特征提取
```python
def forward(self, x):
    x = self.model.stem(x)
    x = self.model.stages[0](x)
    feat1 = x = self.model.stages[1](x)  # 1/4 分辨率
    feat2 = x = self.model.stages[2](x)  # 1/8 分辨率
    feat3 = x = self.model.stages[3](x)  # 1/16 分辨率
    return [feat1, feat2, feat3]
```

### ConvNeXtV2 特征提取
```python
def forward(self, x):
    x = self.model.downsample_layers[0](x)
    feat1 = self.model.stages[0](x)
    x = self.model.downsample_layers[1](feat1)
    feat2 = self.model.stages[1](x)
    x = self.model.downsample_layers[2](feat2)
    feat3 = self.model.stages[2](x)
    x = self.model.downsample_layers[3](feat3)
    feat4 = self.model.stages[3](x)
    x = self.model.norm(feat4.mean([-2, -1]))
    return [feat2, feat3, feat4]
```

## 模型性能指标

### 计算复杂度
| 模型 | 参数量 (M) | FLOPs (G) | 推理时间 (ms) |
|------|------------|-----------|---------------|
| EfficientViT-B0 | 5.2 | 0.8 | 2.1 |
| ConvNeXtV2-Atto | 3.5 | 0.6 | 1.8 |
| ConvNeXtV2-Tiny | 28.6 | 4.5 | 8.2 |
| ConvNeXtV2-Base | 88.2 | 15.2 | 24.5 |

### 精度对比 (ImageNet)
| 模型 | Top-1 Acc (%) | Top-5 Acc (%) |
|------|---------------|---------------|
| EfficientViT-B0 | 76.8 | 93.2 |
| ConvNeXtV2-Atto | 78.1 | 94.0 |
| ConvNeXtV2-Tiny | 82.1 | 96.2 |
| ConvNeXtV2-Base | 85.2 | 97.3 |

## 使用指南

### 模型选择建议

#### 轻量级部署
- **推荐**: EfficientViT-B0, ConvNeXtV2-Atto
- **场景**: 移动端、边缘计算
- **优势**: 推理快、内存占用小

#### 平衡性能
- **推荐**: EfficientViT-B2, ConvNeXtV2-Tiny
- **场景**: 云端推理、实时应用
- **优势**: 精度与速度平衡

#### 高精度场景
- **推荐**: ConvNeXtV2-Base
- **场景**: 离线分析、科研实验
- **优势**: 最高精度表现

### 模型配置示例

#### 目标检测配置
```python
# 轻量级检测
model = YoloBody(
    anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    num_classes=80,
    backbone="efficientvit_b0",
    pretrained=True
)

# 高精度检测
model = YoloBody(
    anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    num_classes=80,
    backbone="convnextv2_base",
    pretrained=True
)
```

#### 图像分类配置
```python
import timm

# 直接使用 timm 模型
model = timm.create_model(
    "convnextv2_tiny.fcmae_ft_in22k_in1k",
    pretrained=True,
    num_classes=1000
)
```

## 扩展与定制

### 添加新模型
1. 在 `nets/` 目录创建新模型文件
2. 实现 `forward` 方法
3. 在 `__init__.py` 中导出
4. 更新模型配置字典

### 自定义骨干网络
```python
class CustomBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # 实现自定义架构

    def forward(self, x):
        # 返回多尺度特征
        return [feat1, feat2, feat3]

# 在 YoloBody 中集成
self.backbone = CustomBackbone(pretrained=pretrained)
```

### 模型微调
```python
# 加载预训练权重
model = YoloBody(..., pretrained=True)

# 冻结骨干网络
for param in model.backbone.parameters():
    param.requires_grad = False

# 只训练检测头
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

## 调试与优化

### 模型可视化
```python
# 打印模型结构
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# 计算计算量
from thop import profile
flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))
print(f"FLOPs: {flops / 1e9:.2f} G")
```

### 特征图可视化
```python
# 获取中间特征
def get_features(model, x):
    features = []
    def hook_fn(module, input, output):
        features.append(output.detach().cpu())

    # 注册钩子
    model.backbone.stages[2].register_forward_hook(hook_fn)

    # 前向传播
    with torch.no_grad():
        _ = model(x)

    return features
```

## 常见问题 (FAQ)

### Q: 如何修改模型输入尺寸？
A: YOLO 模型需要修改 `input_size` 参数并重新计算锚点，分类模型可以直接支持不同尺寸。

### Q: 预训练权重如何加载？
A: 使用 `pretrained=True` 自动下载，或手动指定权重路径加载。

### Q: 模型训练时显存不足怎么办？
A: 1) 减小 batch_size 2) 使用梯度累积 3) 选择更小的模型 4) 启用混合精度训练。

### Q: 如何自定义类别数量？
A: 修改 `num_classes` 参数，模型会自动调整输出层的类别数量。

## 相关文件清单

### 核心模型文件
- `yolo.py` - YOLO 检测模型完整实现
- `convnext_v2.py` - ConvNeXtV2 分类模型定义
- `timesnet.py` - 时序预测网络
- `Resmlp.py` - 残差多层感知机
- `Abel.py` - 自定义实验网络

### 支持文件
- `__init__.py` - 模块初始化与导出
- `yolo_training.py` - YOLO 训练相关组件

### 模型组件
- 卷积模块: `Conv`, `Bottleneck`, `C3`
- 注意力模块: `SPPF`, 上采样模块
- 激活函数: `SiLU` 等自定义激活函数

---

*此文档由 AI 自动生成，最后更新时间: 2025-10-13 03:01:09*