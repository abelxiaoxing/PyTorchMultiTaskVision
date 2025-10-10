import torch
import torch.nn as nn
from timm.layers import trunc_normal_, DropPath
import torch.nn.functional as F

# 自定义LayerNorm层
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 权重参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))   # 偏置参数
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            # 如果数据格式是channels_last，直接调用layer_norm函数
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 如果数据格式是channels_first，手动计算均值和方差进行归一化
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# 全局响应归一化（Global Response Normalization）层
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))   # 偏置参数

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)  # 计算输入的L2范数
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)  # 归一化
        return self.gamma * (x * Nx) + self.beta + x  # 应用GRN并返回结果
    

# 基础块（Block）
class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # 深度卷积
        self.norm = LayerNorm(dim, eps=1e-6)  # 归一化层
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 第一个逐点卷积
        self.act = nn.GELU()  # GELU激活函数
        self.grn = GRN(4 * dim)  # GRN层
        self.pwconv2 = nn.Linear(4 * dim, dim)  # 第二个逐点卷积
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # DropPath层

    def forward(self, x):
        input = x
        x = self.dwconv(x)  # 深度卷积
        x = x.permute(0, 2, 3, 1) # 转换维度 (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)  # 归一化
        x = self.pwconv1(x)  # 第一个逐点卷积
        x = self.act(x)  # 激活函数
        x = self.grn(x)  # GRN
        x = self.pwconv2(x)  # 第二个逐点卷积
        x = x.permute(0, 3, 1, 2) # 转换维度 (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)  # 残差连接
        return x

# ConvNeXtV2主干网络
class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        # 构建stem层
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        # 构建下采样层
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        # 计算每个块的drop_path率
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        # 构建每个stage
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # 最后一层的归一化层
        self.head = nn.Linear(dims[-1], num_classes)  # 全连接层，输出类别数
        self.apply(self._init_weights)  # 初始化权重
        self.head.weight.data.mul_(head_init_scale)  # 调整头部权重
        self.head.bias.data.mul_(head_init_scale)  # 调整头部偏置

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)  # 截断正态分布初始化
            nn.init.constant_(m.bias, 0)  # 偏置初始化为0

    def forward(self, x):
        # 前向传播过程，依次通过每层和每个stage
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)
        x = self.downsample_layers[3](x)
        x = self.stages[3](x)
        x = self.norm(x.mean([-2, -1]))  # 平均池化后归一化
        x = self.head(x)  # 全连接层输出
        return x

# 构建不同规模的ConvNeXtV2模型
def convnextv2_atto(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    if pretrained:
        state_dict = torch.load('model_data/convnextv2_atto.pt')['model']
        model.load_state_dict(state_dict, strict=True)
    return model

def convnextv2_femto(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    if pretrained:
        state_dict = torch.load('model_data/convnextv2_femto.pt')['model']
        model.load_state_dict(state_dict, strict=True)
    return model

def convnextv2_pico(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    if pretrained:
        state_dict = torch.load('model_data/convnextv2_pico.pt')['model']
        model.load_state_dict(state_dict, strict=True)
    return model

def convnextv2_nano(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    if pretrained:
        state_dict = torch.load('model_data/convnextv2_nano.pt')['model']
        model.load_state_dict(state_dict, strict=True)
    return model

def convnextv2_tiny(pretrained=False,**kwargs):
    model = ConvNeXtV2(num_classes=2, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        state_dict = torch.load('model_data/convnextv2_tiny.pt')['model']
        model.load_state_dict(state_dict, strict=True)
    return model

def convnextv2_base(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        state_dict = torch.load('model_data/convnextv2_base.pt')['model']
        model.load_state_dict(state_dict, strict=True)
    return model
