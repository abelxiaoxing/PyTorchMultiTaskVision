import json
import os
import random
from pathlib import Path

import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms

from config import ClassificationConfig

def _print_transform(transform, name):
    print(f"{name} Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

# 将数据集划分为训练集和验证集，确保验证集每个类别的样本数量相等。
def split_dataset(root, train_ratio=0.5):
    dataset = datasets.ImageFolder(root)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 使用高效的'dataset.samples'按类别对指标进行分组
    class_indices = {i: [] for i in range(len(class_to_idx))}
    for i, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(i)

    train_indices, val_indices = [], []
    train_class_counts = {k: 0 for k in class_to_idx.keys()}
    val_class_counts = {k: 0 for k in class_to_idx.keys()}

    # 分层拆分：保持原始类别分布
    for class_idx, indices in class_indices.items():
        random.shuffle(indices)
        split_point = int(len(indices) * train_ratio)
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])
        train_class_counts[idx_to_class[class_idx]] = len(indices[:split_point])
        val_class_counts[idx_to_class[class_idx]] = len(indices[split_point:])

    print("训练集每个类别的样本数量:", train_class_counts)
    print("验证集每个类别的样本数量:", val_class_counts)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset, class_to_idx


# 构建数据集
def build_dataset(cfg: ClassificationConfig, eval_only: bool = False):
    """Build train/val splits using the config-driven transforms and paths."""
    val_transform = build_transform(False, cfg)
    if eval_only:
        _print_transform(val_transform, "Validation")
        val_dataset = datasets.ImageFolder(cfg.data.data_path, transform=val_transform)
        num_classes = len(val_dataset.classes)
        return None, val_dataset, num_classes

    # 构建训练集和验证集的转换
    train_transform = build_transform(True, cfg)  # 训练集转换

    _print_transform(train_transform, "Train")
    _print_transform(val_transform, "Validation")

    if cfg.data.train_split_ratio == 0:  # 如果数据集是手动设置
        # 手动设置训练集和验证集路径
        train_root = os.path.join(cfg.data.data_path, "train")
        val_root = os.path.join(cfg.data.data_path, "val")

        # 加载训练集和验证集
        train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_root, transform=val_transform)
        
        # 获取类别索引（从训练集获取）
        class_indices = train_dataset.class_to_idx
    else:  # 如果数据集是自动生成
        dataset_root = cfg.data.data_path
        train_ratio = cfg.data.train_split_ratio
        train_dataset, val_dataset, class_indices = split_dataset(dataset_root, train_ratio)

        # 应用转换到训练集和验证集
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
    num_classes = len(class_indices)
    print("Number of the class = %d" % num_classes)  # 打印类别数量

    # 将类索引保存到 JSON 文件中
    json_str = json.dumps({val: key for key, val in class_indices.items()}, indent=4)
    output_dir = Path(cfg.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "class_indices.json", "w") as f:
        f.write(json_str)
    print(f"Number of the class = {num_classes}")  # 打印类别数量
    return train_dataset, val_dataset, num_classes


def build_transform(is_train: bool, cfg: ClassificationConfig):
    if is_train:
        transform = create_transform(
            input_size=cfg.model.input_size,
            scale=(1.0, 1.0),
            ratio=(1.0, 1.0),
            is_training=True,
            vflip=0.5,
            color_jitter=cfg.augmentation.color_jitter,
            auto_augment=cfg.augmentation.aa,
            interpolation="bicubic",
            re_prob=cfg.augmentation.reprob,
        )
        return transform
    else:
        return transforms.Compose([
            transforms.Resize([cfg.model.input_size, cfg.model.input_size]),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
