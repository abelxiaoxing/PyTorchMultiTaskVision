import torch
import random
import os
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import json

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
    else:  # 如果数据集是自动生成
        dataset_root = args.data_path
        train_ratio = args.train_split_rato
        train_dataset, val_dataset, class_indices = split_dataset(dataset_root, train_ratio)

        # 应用转换到训练集和验证集
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
    num_classes = len(class_indices)
    print("Number of the class = %d" % num_classes)  # 打印类别数量

    # 将类索引保存到 JSON 文件中
    json_str = json.dumps({val: key for key, val in class_indices.items()}, indent=4)
    with open("./train_cls/output/class_indices.json", "w") as f:
        f.write(json_str)
    print(f"Number of the class = {num_classes}")  # 打印类别数量
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
        )
        return transform
    else :
        t = []
        size = args.input_size
        t.append(transforms.Resize([size, size]))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)

