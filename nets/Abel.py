# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, GluMlp, SwiGLU, LayerNorm, DropPath, apply_rot_embed_cat, use_fused_attn
from typing import Callable, Optional, Tuple

# 定义自定义的注意力机制模块AbelAttention
class AbelAttention(nn.Module):
    fused_attn: torch.jit.Final[bool]
    
    def __init__(
            self,
            dim: int,  # 输入特征维度
            num_heads: int = 8,  # 注意力头的数量
            qkv_bias: bool = True,  # 是否使用偏置
            qkv_fused: bool = True,  # 是否使用融合的qkv
            attn_drop: float = 0.,  # 注意力dropout的概率
            proj_drop: float = 0.,  # 投影dropout的概率
            attn_head_dim: Optional[int] = None,  # 每个头的维度
            norm_layer: Optional[Callable] = None,  # 归一化层
    ):
        super().__init__()
        self.num_heads = num_heads  # 设置头的数量
        head_dim = dim // num_heads  # 每个头的维度
        if attn_head_dim is not None:
            head_dim = attn_head_dim  # 如果提供了特定头维度，则使用
        all_head_dim = head_dim * self.num_heads  # 总头维度
        self.scale = head_dim ** -0.5  # 缩放因子
        self.fused_attn = use_fused_attn()  # 判断是否使用融合注意力

        # 如果使用融合的qkv
        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)  # 创建融合的qkv线性层
            self.q_proj = self.k_proj = self.v_proj = None  # 单独的q, k, v线性层设为None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))  # q的偏置参数
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)  # k的偏置
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))  # v的偏置参数
            else:
                self.q_bias = self.k_bias = self.v_bias = None  # 如果不使用偏置，设为None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)  # 创建单独的q线性层
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)  # 创建单独的k线性层
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)  # 创建单独的v线性层
            self.qkv = None  # 融合的qkv设为None
            self.q_bias = self.k_bias = self.v_bias = None  # 不使用偏置

        self.attn_drop = nn.Dropout(attn_drop)  # 注意力dropout层
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()  # 归一化层
        self.proj = nn.Linear(all_head_dim, dim)  # 投影层
        self.proj_drop = nn.Dropout(proj_drop)  # 投影dropout层

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,  # 可选的旋转嵌入
            attn_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码
    ):
        B, N, C = x.shape  # 获取输入的批次大小、序列长度和特征维度

        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None  # 合并偏置
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)  # 计算qkv
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 重塑并交换维度
            q, k, v = qkv.unbind(0)  # 拆分q, k, v
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # 计算q
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # 计算k
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # 计算v

        # 如果提供了旋转嵌入
        if rope is not None:
            q = torch.cat([q[:, :, :1, :], apply_rot_embed_cat(q[:, :, 1:, :], rope)], 2).type_as(v)  # 对q应用旋转嵌入
            k = torch.cat([k[:, :, :1, :], apply_rot_embed_cat(k[:, :, 1:, :], rope)], 2).type_as(v)  # 对k应用旋转嵌入

        # 如果使用融合注意力
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,  # 注意力掩码
                dropout_p=self.attn_drop.p if self.training else 0.,  # dropout概率
            )
        else:
            q = q * self.scale  # 缩放q
            attn = (q @ k.transpose(-2, -1))  # 计算注意力权重
            attn = attn.softmax(dim=-1)  # 应用softmax
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))  # 应用掩码
            attn = self.attn_drop(attn)  # 应用dropout
            x = attn @ v  # 计算输出

        x = x.transpose(1, 2).reshape(B, N, C)  # 交换维度并重塑输出
        x = self.norm(x)  # 归一化
        x = self.proj(x)  # 投影
        x = self.proj_drop(x)  # 应用dropout
        return x

# 定义AbelBlock模块
class AbelBlock(nn.Module):

    def __init__(
            self,
            dim: int,  # 输入特征维度
            num_heads: int,  # 注意力头数量
            qkv_bias: bool = True,  # 是否使用偏置
            qkv_fused: bool = True,  # 是否使用融合的qkv
            mlp_ratio: float = 4.,  # MLP隐藏层维度与输入维度的比率
            swiglu_mlp: bool = False,  # 是否使用SwiGLU
            scale_mlp: bool = False,  # 是否缩放MLP
            scale_attn_inner: bool = False,  # 是否缩放内部注意力
            proj_drop: float = 0.,  # 投影dropout概率
            attn_drop: float = 0.,  # 注意力dropout概率
            drop_path: float = 0.,  # drop path概率
            init_values: Optional[float] = None,  # 初始化值
            act_layer: Callable = nn.GELU,  # 激活层
            norm_layer: Callable = LayerNorm,  # 归一化层
            attn_head_dim: Optional[int] = None,  # 每个头的维度
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)  # 第一个归一化层
        self.attn = AbelAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,  # 内部注意力的归一化层
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None  # 初始化gamma_1
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # drop path层

        self.norm2 = norm_layer(dim)  # 第二个归一化层
        hidden_features = int(dim * mlp_ratio)  # 计算隐藏层维度
        if swiglu_mlp:
            if scale_mlp:
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None  # 初始化gamma_2
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # 第二个drop path层

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))  # 第一层计算
            x = x + self.drop_path2(self.mlp(self.norm2(x)))  # 第二层计算
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))  # 带gamma的第一层计算
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))  # 带gamma的第二层计算
        return x

# 定义模型结构
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        hidden_dim = 32  # 隐藏层维度
        depth = 6  # 堆叠层数
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=7, stride=1,padding=3),  # 一维卷积层
            nn.BatchNorm1d(hidden_dim),  # 批归一化层
            nn.GELU()  # 激活函数
        )
        self.blocks = nn.ModuleList([
            AbelBlock(
                dim=hidden_dim,
                num_heads=8,
                qkv_bias=True,
                qkv_fused=True,
                mlp_ratio=2,
                swiglu_mlp=True,
                scale_mlp=False,
                scale_attn_inner=False,
                proj_drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=LayerNorm,
                init_values=None,
            )
            for i in range(depth)])  # 堆叠多个AbelBlock
        self.norm = LayerNorm(hidden_dim)  # 最后的归一化层
        self.head = nn.Linear(hidden_dim, 1)  # 输出层

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加一个维度
        x = self.conv1(x)  # 通过卷积层
        x = x.transpose(1, 2)  # 交换维度
        for blk in self.blocks:
            x = blk(x)  # 通过每个AbelBlock
        x = self.norm(x)  # 最后归一化
        x = x[:, 1:].mean(dim=1)  # 取平均
        x = 100 * F.sigmoid(self.head(x))  # 通过输出层并缩放输出
        return x  # 返回最终输出
