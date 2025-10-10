from torch.utils.data import Dataset
from rich.progress import Progress 
import torch
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 定义一个创建学习率调度的函数
def create_lr_scheduler(optimizer,num_step: int,epochs: int,warmup=True,warmup_epochs=1,warmup_factor=1e-3,end_factor=1e-6):
    # 确保num_step和epochs都大于0
    assert num_step > 0 and epochs > 0
    # 如果warmup为False，则将warmup_epochs设置为0
    if warmup is False:
        warmup_epochs = 0

    # 定义一个函数f，用于计算学习率
    def f(x):
        # 如果warmup为True，并且x小于等于warmup_epochs乘以num_step，则根据x计算学习率
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        # 如果warmup为False，则根据x计算学习率
        else:
            current_step = x - warmup_epochs * num_step
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (
                1 - end_factor
            ) + end_factor

    # 返回一个LambdaLR对象，用于设置学习率
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)



class CoalDataset(Dataset):
    def __init__(self, features, labels=None, pca_model=None, n_components=None):
        # 初始化方法，接收特征、标签、PCA模型和PCA组件数
        # 如果PCA模型为空，则创建一个新的PCA模型并对特征进行PCA处理
        if pca_model is None:
            self.pca = PCA(n_components=n_components)
            self.content_data = torch.Tensor(self.pca.fit_transform(features))
        else:
            self.pca = pca_model
            self.content_data = torch.Tensor(self.pca.transform(features))
        self.labels=labels
        # 将标签转换为Tensor
        if labels is not None:
            self.label_data = torch.Tensor(labels)

    def __len__(self):
        # 返回数据集的长度
        return len(self.content_data)

    def __getitem__(self, idx):
        # 获取索引为idx的数据
        content = self.content_data[idx]
        if self.labels is not None:
            label = self.label_data[idx]
            return content, label
        else:
            return content

    def transform(self, data):
        # 对数据进行标准化和PCA变换
        data = self.scaler.transform(data)
        return torch.Tensor(self.pca.transform(data))
    
    
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    # 模型训练函数，每个epoch训练一次
    progress = Progress()  # 训练进度
    model.train()  # 设置模型为训练模式
    train_loss = 0.0  # 训练损失
    for input, label in train_loader:
        input, label = input.to(device), label.to(device)  # 将输入和标签移动到设备上
        output = model(input)  # 获取模型输出
        loss = criterion(output, label)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        train_loss += loss.item()  # 累加损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    avg_train_loss = train_loss / len(train_loader)  # 计算平均训练损失
    return avg_train_loss
        
def val(model, val_loader, criterion, best_val_loss, device):
    # 模型验证函数
    model.eval()  # 设置模型为评估模式
    output_all = []  # 存储所有输出
    label_all = []  # 存储所有标签
    val_loss = 0.  # 验证损失
    with torch.no_grad():
        for input, label in val_loader:
            input, label = input.to(device), label.to(device)  # 将输入和标签移动到设备上
            output = model(input)  # 获取模型输出
            loss = criterion(output, label)  # 计算损失
            val_loss += loss.item()  # 累加损失
            output_all.append(output)
            label_all.append(label)
        output_all = torch.cat(output_all, dim=0)  # 拼接所有输出
        label_all = torch.cat(label_all, dim=0)  # 拼接所有标签
        mean_labels = torch.mean(label_all)  # 计算标签均值
        SSR = torch.sum((output_all - label_all)**2)  # 计算回归平方和
        SST = torch.sum((label_all - mean_labels) ** 2)  # 计算总平方和
        r2_score = (1 - (SSR / SST)).item()  # 计算R^2分数
        avg_val_loss = val_loss / len(val_loader)  # 计算平均验证损失

    return avg_val_loss, r2_score, best_val_loss

def test(model, test_loader, criterion, device):
    # 模型测试函数
    model.eval()  # 设置模型为评估模式
    output_all = []  # 存储所有输出
    label_all = []  # 存储所有标签
    test_loss = 0.  # 测试损失
    with torch.no_grad():
        for input, label in test_loader:
            input, label = input.to(device), label.to(device)  # 将输入和标签移动到设备上
            output = model(input)  # 获取模型输出
            loss = criterion(output, label)  # 计算损失
            test_loss += loss.item()  # 累加损失
            output_all.append(output)
            label_all.append(label)
        output_all = torch.cat(output_all, dim=0)  # 拼接所有输出
        label_all = torch.cat(label_all, dim=0)  # 拼接所有标签
        mean_labels = torch.mean(label_all)  # 计算标签均值
        SSR = torch.sum((output_all - label_all)**2)  # 计算回归平方和
        SST = torch.sum((label_all - mean_labels) ** 2)  # 计算总平方和
        r2_score = (1 - (SSR / SST)).item()  # 计算R^2分数
        avg_test_loss = test_loss / len(test_loader)  # 计算平均测试损失
    return avg_test_loss, r2_score