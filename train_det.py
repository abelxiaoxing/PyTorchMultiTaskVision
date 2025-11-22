import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.yolo import YoloBody
from nets.yolo_training import (
    YOLOLoss,
    get_lr_scheduler,
    set_optimizer_lr,
    weights_init,
)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.coco_dataloader import CocoYoloDataset, coco_dataset_collate
from utils.utils import get_anchors, get_classes, show_config
from utils.utils_fit import fit_one_epoch
from utils.coco_utils import validate_coco_annotations, get_coco_statistics, get_coco_class_mapping
from pathlib import Path
import utils.utils as utils


def train_detection(
    device: str = "cuda",
    resume="",
    input_size=224,
    backbone="efficientvit_b0",
    mosaic=True,
    mixup=True,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    opt="adamw",
    save_ckpt_freq=10,
    eval_freq=10,
    focal_loss=False,
    data_path="",
):

    lr_scheduler = "cosine"
    pretrained = True
    focal_alpha = 0.25
    focal_gamma = 2
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    mosaic_prob = 0.5
    mixup_prob = 0.5
    special_aug_ratio = 0.7
    label_smoothing = 0.01
    weight_decay = 5e-4
    Freeze_Epoch = int(epochs / 2)
    start_epoch = 0
    device = torch.device(device)
    Freeze_Train = True
    momentum = 0.937
    save_dir = Path("train_det/output")
    os.makedirs(save_dir, exist_ok=True)
    figure_dir = Path("train_det/figure")
    eval_flag = True
    num_workers = 1

    # 使用COCO API的新方式
    print("使用COCO API加载数据集... (哈雷酱大小姐推荐！)")

    # 验证COCO标注文件
    train_annotation_path = Path(data_path) / "annotations" / "instances_train2017.json"
    val_annotation_path = Path(data_path) / "annotations" / "instances_val2017.json"
    train_image_path = Path(data_path) / "train2017"
    val_image_path = Path(data_path) / "val2017"

    # 验证标注文件
    for ann_file in [train_annotation_path, val_annotation_path]:
        is_valid, msg = validate_coco_annotations(str(ann_file))
        if not is_valid:
            raise ValueError(f"COCO标注文件验证失败: {ann_file}, 错误: {msg}")

    # 获取COCO统计信息
    train_stats = get_coco_statistics(str(train_annotation_path))
    val_stats = get_coco_statistics(str(val_annotation_path))

    print(f"训练集统计: {train_stats['num_images']} 张图片, {train_stats['num_annotations']} 个标注")
    print(f"验证集统计: {val_stats['num_images']} 张图片, {val_stats['num_annotations']} 个标注")

    # 获取类别信息
    class_mapping = get_coco_class_mapping(str(train_annotation_path))
    class_names = list(class_mapping.values())
    num_classes = len(class_names)

    print(f"类别数量: {num_classes}")
    print(f"类别列表: {class_names[:5]}...")  # 只显示前5个类别

    num_train = train_stats['num_images']
    num_val = val_stats['num_images']

    anchors, num_anchors = get_anchors(
        "12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401"
    )

    model = YoloBody(
        anchors_mask, num_classes, backbone=backbone, pretrained=pretrained
    )
    if not pretrained:
        weights_init(model)

    yolo_loss = YOLOLoss(
        anchors,
        num_classes,
        input_size,
        anchors_mask,
        label_smoothing,
        focal_loss,
        focal_alpha,
        focal_gamma,
    )
    loss_history = LossHistory(figure_dir, model, input_size=input_size)
    model_train = model.train()

    if device.type == "cuda":
        cudnn.benchmark = True
    model_train = model_train.to(device)

    # 显示配置信息
    show_config(
        classes_path="COCO API",
        resume=resume,
        input_size=input_size,
        start_epoch=start_epoch,
        Freeze_Epoch=Freeze_Epoch,
        epochs=epochs,
        batch_size=batch_size,
        Freeze_Train=Freeze_Train,
        lr=lr,
        opt=opt,
        momentum=momentum,
        lr_scheduler=lr_scheduler,
        save_ckpt_freq=save_ckpt_freq,
        save_dir=save_dir,
        num_workers=num_workers,
        num_train=num_train,
        num_val=num_val,
    )


    UnFreeze_flag = False
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = {
        "adam": optim.Adam(pg0, lr, betas=(momentum, 0.999)),
        "adamw": optim.AdamW(pg0, lr, betas=(momentum, 0.999)),
        "sgd": optim.SGD(pg0, lr, momentum=momentum, nesterov=True),
    }[opt]
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})

    # 创建一个简单的参数对象用于auto_load_model
    class SimpleArgs:
        def __init__(self):
            self.auto_resume = False
            self.resume = resume
            self.model_ema = False
            self.start_epoch = 0

    args_obj = SimpleArgs()

    # 尝试加载模型（如果有的话）
    if resume:
        try:
            utils.auto_load_model(
                args=args_obj,
                model_without_ddp=model,
                optimizer=optimizer,
                loss_scaler=None,
                model_ema=None,
            )
            start_epoch = args_obj.start_epoch
        except Exception as e:
            print(f"模型加载失败，使用从头训练: {e}")
            start_epoch = 0
    else:
        start_epoch = 0
    lr_scheduler_func = get_lr_scheduler(lr_scheduler, lr, epochs)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    # 使用COCO API创建数据集
    print("创建COCO YOLO数据集...")
    train_dataset = CocoYoloDataset(
        root=str(train_image_path),
        annFile=str(train_annotation_path),
        input_size=input_size,
        num_classes=num_classes,
        epoch_length=epochs,
        mosaic=mosaic,
        mixup=mixup,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
        train=True,
        special_aug_ratio=special_aug_ratio,
    )
    val_dataset = CocoYoloDataset(
        root=str(val_image_path),
        annFile=str(val_annotation_path),
        input_size=input_size,
        num_classes=num_classes,
        epoch_length=epochs,
        mosaic=False,
        mixup=False,
        mosaic_prob=0,
        mixup_prob=0,
        train=False,
        special_aug_ratio=0,
    )
    collate_fn = coco_dataset_collate

    train_sampler = None
    val_sampler = None
    shuffle = True

    gen = DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )
    gen_val = DataLoader(
        val_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=val_sampler,
    )

    # 评估回调
    # 对于COCO API，我们使用图像ID列表
    val_image_ids = val_dataset.image_ids if hasattr(val_dataset, 'image_ids') else []
    eval_callback = EvalCallback(
        model,
        input_size,
        anchors,
        anchors_mask,
        class_names,
        num_classes,
        val_image_ids,  # 使用图像ID列表
        figure_dir,
        device,
        eval_flag=eval_flag,
        period=eval_freq,
        num_epochs=epochs,
    )

    temp_map = 0
    last_map = temp_map
    for epoch in range(start_epoch, epochs):

        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            nbs = 64
            lr_limit_max = 1e-3 if opt in ["adam", "adamw"] else 5e-2
            lr_limit_min = 3e-4 if opt in ["adam", "adamw"] else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(
                max(batch_size / nbs * lr, lr_limit_min * 1e-2),
                lr_limit_max * 1e-2,
            )
            lr_scheduler_func = get_lr_scheduler(
                lr_scheduler, Init_lr_fit, Min_lr_fit, epochs
            )
            for param in model.backbone.parameters():
                param.requires_grad = True
            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size
            UnFreeze_flag = True

        gen.dataset.epoch_now = epoch
        gen_val.dataset.epoch_now = epoch

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        train_loss_avg, val_loss_avg, temp_map = fit_one_epoch(
            model_train,
            model,
            yolo_loss,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            gen,
            gen_val,
            epochs,
            device,
            save_ckpt_freq,
            save_dir,
            temp_map,
        )
        if temp_map is None:
            temp_map = last_map
        else:
            last_map = temp_map



if __name__ == "__main__":
    train_detection(
        data_path='/home/abelxiaoxing/datas/COCO2017',
        input_size=640,     # YOLO标准输入尺寸
        batch_size=16,      # 根据GPU内存调整
        epochs=10,          # 测试用较少的轮数
        save_ckpt_freq=5,   # 更频繁保存检查点
        eval_freq=5,        # 更频繁评估
    )
