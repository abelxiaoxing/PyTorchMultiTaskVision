import torch
import random
import os
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import json
from pathlib import Path

# 将数据集划分为训练集和验证集，确保验证集每个类别的样本数量相等。
def split_dataset(root, train_ratio=0.5):
    dataset = datasets.ImageFolder(root)
    class_indices = dataset.class_to_idx
    class_samples = {class_name: [] for class_name in class_indices}

    # 将样本按类别分组
    for idx, (img, label) in enumerate(dataset):
        class_name = list(class_indices.keys())[list(class_indices.values()).index(label)]
        class_samples[class_name].append(idx)

    train_indices, val_indices = [], []

    # 按类别划分训练集和验证集
    min_class_size = min(len(indices) for indices in class_samples.values())  # 找到最小类别样本数量
    val_size_per_class = min_class_size - int(min_class_size * train_ratio)  # 计算每个类别的验证集样本数量

    for class_name, indices in class_samples.items():
        random.shuffle(indices)  # 打乱样本顺序
        train_indices.extend(indices[:-val_size_per_class]) 
        val_indices.extend(indices[-val_size_per_class:])  

    train_dataset = torch.utils.data.Subset(dataset, train_indices) 
    val_dataset = torch.utils.data.Subset(dataset, val_indices)  

    # 打印每个类别在训练集和验证集中的样本数量
    train_class_counts = {class_name: 0 for class_name in class_indices}
    val_class_counts = {class_name: 0 for class_name in class_indices}

    for idx in train_indices:
        _, label = dataset[idx]
        class_name = list(class_indices.keys())[list(class_indices.values()).index(label)]
        train_class_counts[class_name] += 1

    for idx in val_indices:
        _, label = dataset[idx]
        class_name = list(class_indices.keys())[list(class_indices.values()).index(label)]
        val_class_counts[class_name] += 1

    print("训练集每个类别的样本数量:", train_class_counts)
    print("验证集每个类别的样本数量:", val_class_counts)

    return train_dataset, val_dataset, class_indices


# 构建数据集
def build_dataset(args):
    # 构建训练集和验证集的转换
    train_transform = build_transform(True, args)  # 训练集转换
    val_transform = build_transform(False, args)   # 验证集转换

    print("Train Transform = ")
    if isinstance(train_transform, tuple):  # 打印转换信息
        for trans in train_transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in train_transform.transforms:
            print(t)
    print("---------------------------")

    print("Validation Transform = ")
    if isinstance(val_transform, tuple):
        for trans in val_transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in val_transform.transforms:
            print(t)
    print("---------------------------")

    if args.train_split_rato == 0:  # 如果数据集是手动设置
        # 手动设置训练集和验证集路径
        train_root = os.path.join(args.data_path, "train")
        val_root = os.path.join(args.data_path, "val")

        # 加载训练集和验证集
        train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_root, transform=val_transform)
        
        # 获取类别索引（从训练集获取）
        class_indices = train_dataset.class_to_idx
        # 保存类别索引为JSON文件
        json_str = json.dumps(
            dict((val, key) for key, val in class_indices.items()), indent=4
        )
        with open(Path("./train_cls/output/class_indices.json"), "w") as f:
            f.write(json_str)
        num_classes = len(train_dataset.class_to_idx)
    else:  # 如果数据集是自动生成
        dataset_root = args.data_path
        train_ratio = args.train_split_rato
        train_dataset, val_dataset, class_indices = split_dataset(dataset_root, train_ratio)

        # 应用转换到训练集和验证集
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
        # 保存类别索引为JSON文件
        json_str = json.dumps(
            dict((val, key) for key, val in class_indices.items()), indent=4
        )
        with open(Path("./train_cls/output/class_indices.json"), "w") as f:
            f.write(json_str)
        num_classes = len(class_indices)
    print("Number of the class = %d" % num_classes)  # 打印类别数量
    return train_dataset, val_dataset, num_classes


def build_transform(is_train, args):
    if is_train:
        transform = []
        transform = create_transform(
            input_size=args.input_size,
            scale=(1.0, 1.0),
            ratio=(1.0, 1.0),
            is_training=True,
            vflip=0.5,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        return transform
    else :
        t = []
        size = args.input_size
        t.append(transforms.Resize([size, size]))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)
