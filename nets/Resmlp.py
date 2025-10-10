import torch.nn as nn
import torch

# 定义多层感知机（MLP）类，继承自nn.ModuleList
class MLP(nn.ModuleList):
    def __init__(self, channels, skips=None, use_bn=True, act=nn.GELU, dropout=0.0):
        super().__init__()
        self.num_layers = len(channels) - 1  # 计算网络层数
        if skips is None:
            skips = {}  # 如果没有指定跳跃连接，则初始化为空字典
        self.skips = skips  # 存储跳跃连接的字典
        self.channels = channels  # 存储各层通道数
        for i in range(1, self.num_layers + 1):
            in_channels = channels[i - 1] + (channels[skips[i]] if i in skips else 0)  # 计算输入通道数
            layers = [nn.Linear(in_channels, channels[i])]  # 添加线性层
            if i < self.num_layers:
                if use_bn:
                    layers.append(nn.BatchNorm1d(channels[i]))  # 可选地添加批量归一化层
                layers.append(act())  # 添加激活函数层
            if i + 1 == self.num_layers and dropout > 0:
                layers.append(nn.Dropout(dropout, inplace=True))  # 可选地添加dropout层
            self.append(nn.Sequential(*layers))  # 将这些层组成一个序列并添加到模块列表中

    def forward(self, x):
        xs = [x]  # 存储各层的输入
        for i in range(self.num_layers):
            if i + 1 in self.skips:
                x = torch.cat([xs[self.skips[i + 1]], x], dim=-1)  # 如果有跳跃连接，则将对应层的输出拼接到当前输入
            x = self[i](x)  # 通过第i层
            xs.append(x)  # 保存当前层的输出
        return x  # 返回最后一层的输出

# 定义回归模型类，继承自nn.Module
class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        self.net = MLP(
            [input_size, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4, output_size],  # 指定各层的通道数
            # skips={3: 1, 5: 3, 7: 5, 9: 7},  # 跳跃连接的设置（已注释）
            act=nn.SELU,  # 激活函数选择SELU
            # dropout=0.1  # dropout的概率（已注释）
        )
        self._initialize_weights()  # 初始化权重

    def forward(self, x):
        x = self.net(x)  # 通过多层感知机网络
        x = 100 * torch.sigmoid(x)  # 输出通过sigmoid激活函数，并乘以100
        return x  # 返回最终输出

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 初始化线性层的权重
                nn.init.kaiming_normal_(m.weight, nonlinearity='selu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 将偏置初始化为0
