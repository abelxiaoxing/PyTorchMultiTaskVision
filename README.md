# PyTorchMultiTaskVision 环境同步指南

## 快速开始

### 环境要求
- Python >= 3.12
- CUDA >= 11.8
- Git

### 环境同步步骤

本项目使用uv包管理器进行环境管理，并需要手动安装NVIDIA apex包。请按照以下步骤操作：

#### 第一步：同步基础环境（使用阿里源）
```bash
# 使用uv同步基础环境（会自动使用阿里源）
uv sync
```

#### 第二步：安装NVIDIA apex包
```bash
# 克隆NVIDIA apex仓库
git clone https://github.com/NVIDIA/apex

# 激活虚拟环境并安装apex
source .venv/bin/activate
cd apex
python setup.py install --cpp_ext --cuda_ext

```

### 包源配置
本项目已配置使用以下镜像源：
- **主源**：阿里云镜像 https://mirrors.aliyun.com/pypi/simple/
- **PyTorch源**：PyTorch官方CUDA 13.0源 https://download.pytorch.org/whl/cu130


## 项目简介

PyTorchMultiTaskVision是一个基于PyTorch的深度学习多任务视觉框架，支持图像分类、目标检测等多种视觉任务。本项目采用模块化设计，易于扩展和维护。

### 主要特性
- 🚀 支持多GPU训练
- 📊 集成TensorBoard可视化
- 🔄 自动数据划分和类别统计
- 🎯 支持多种优化器和调度器
- 📈 支持模型EMA（指数移动平均）
- 🔧 支持模型转换（ONNX、TensorRT等）
- 🎨 丰富的数据增强策略
- 📋 详细的训练日志和指标记录

### 项目结构
```
PyTorchMultiTaskVision/
├── configs/          # 配置文件
├── datasets/         # 数据集处理
├── engine/           # 训练引擎
├── models/           # 模型定义
├── utils/            # 工具函数
├── apex/            # NVIDIA apex（需要手动安装）
├── train.py         # 训练脚本
├── val.py           # 验证脚本
├── uv_sync_step1.sh # 环境同步第一步脚本
├── uv_sync_step2.sh # 环境同步第二步脚本
├── pyproject.toml   # 项目配置
└── README.md        # 项目说明
```

### 快捷命令参考（纯uv版本）
```bash
# 环境同步（推荐方式）
./uv_sync_step1.sh  # 第一步：使用uv sync安装基础依赖
./uv_sync_step2.sh  # 第二步：安装NVIDIA apex包

# 或者直接使用uv命令
uv sync              # 安装基础依赖（使用阿里源）
# 然后手动安装apex：git clone + python setup.py install
```

---

## 更新说明
### 20250730
+ 大幅简化了engine和datasets代码结构，移除了重复代码和冗余循环。
+ 删除mmdet依赖
+ 删除冗余数据增强
+ 采用分层采样策略，保证数据划分后的类别分布不变。
+ 优化了划分性能，避免加载完整图像，速度大幅提升。
+ modelchange改为面向对象，代码更加清晰，易于维护。
+ optim_factory区分权重和偏置/归一化参数，不对偏置/归一化参数应用衰减。重构get_parameter_groups部分，用列表推导式构建优化器所需的参数组，提高性能。
+ 删除val.py融合进train.py中，train.py内涵三个mode，分别为train、eval、move。

### 20241217
+ modelchange.py新增model_ema转化为标准模型，权重中"model_ema"改为"model"
+ 修复当model_ema为True时,恢复的权重文件中无model_ema时的权重加载问题
+ 启用mixup时,也可以查看train数据集的准确率
+ 分类任务时模型权重不匹配模型文件时,打印不匹配的层,自动剔除不匹配层(迁移学习等无需手动剔除最后一层)

### 20241127
+ 新增集群训练run_with_submitit.py
+ 新增模型转换model_change.py，实现onnx,pth,trt模型相互转换
+ train.py移除分类任务中num_classes参数，自动读取文件夹内的类别数量
+ 为便于分享减少手动设置参数，修改模型保存方式，只保存权重改为保存整个模型
+ train.py更新model_ema方法，加快了速度
+ val.py新增参数model_ema，决定采用原始模型推理还是采用ema权重推理,如果权重中没有保存model_ema即使推理采用model_ema也将使用原始权重推理
+ 提高optim_factory的自动化程度，非关键参数改为默认


## 基于pytorch的深度学习图像分类框架使用简介
1. 下载好数据集，代码中默认使用的是迷你猫狗分类数据集，下载地址:  [迷你猫狗分类数据集](https://pan.baidu.com/s/16SPmrN_PUUTWQuxtRXZmrA?pwd=abel) 提取码: abel  
### 训练
+ 多显卡训练,此案例为8卡`torchrun --nproc_per_node=8 train.py`
+ 单显卡训练`python train.py`
### train.py
1. 在`train.py`脚本中将`--data-path`设置成解压后的`flower_photos`文件夹绝对路径
2. 去github抄所需要网络的代码，并将其放在`models`文件夹下，然后将`train.py`下导入的网络设置成你要使用的网络
3. 在`train.py`脚本中将`--weights`参数设成下载好的预训练权重路径
4. 设置好数据集的路径`--data-path`以及预训练权重的路径`--weights`就能使用`train.py`脚本开始训练了(训练过程中会自动生成`class_indices.json`文件)
5. 若需要自己划分好数据集并在所填的data_path路径后划分为train和val,则设置data_custom为True。
### val.py
1. 在`val.py`脚本中导入和训练脚本中同样的模型，并将`model_weight_path`设置成训练好的模型权重路径(默认保存在weights文件夹下)
2. 在`val.py`脚本中将`img_path`设置成你自己需要预测的图片文件夹绝对路径
3. 设置好权重路径`model_weight_path`和预测的图片路径`img_path`就能使用`val.py`脚本进行预测了
4. 选择val_move和val_precision执行，val_move为预测该文件夹的图片并进行分类，val_precision为预测包含类别的文件夹并给出精确率
### Others
+ 如果要使用自己的数据集，请按照花分类数据集的文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的`num_classes`设置成你自己数据的类别数
### todo
1. 适配目标检测
2. 适配语义分割
3. 增加权重转换
4. 增加模型可视化
5. 增加剪枝、量化、蒸馏等操作