from pathlib import Path
from typing import Optional

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import DetectionConfig, load_detection_config
from nets.yolo import YoloBody
from nets.yolo_training import (
    YOLOLoss,
    get_lr_scheduler,
    set_optimizer_lr,
    weights_init,
)
from utils.callbacks import EvalCallback, LossHistory
from utils.coco_dataloader import CocoYoloDataset
from utils.coco_utils import get_coco_class_mapping, get_coco_statistics, validate_coco_annotations
from utils.checkpoint import auto_load_model
from utils.loggers import show_config
from utils.runtime import setup_runtime
from utils.utils_fit import fit_one_epoch
from utils.vision import get_anchors
from utils.yolo_transforms import collate_yolo_batch





def train_detection(
    config_path: Optional[str] = None,
):
    """
    目标检测训练入口，使用 TOML 配置。
    """
    cfg = load_detection_config(config_path)

    if not cfg.data.data_path:
        raise ValueError("data_path 不能为空，请提供 COCO 数据路径。")

    device = setup_runtime(cfg.runtime)
    lr_scheduler_name = cfg.training.lr_scheduler
    anchors_mask = [list(mask) for mask in cfg.model.anchor_mask]
    mosaic_prob = cfg.augmentation.mosaic_prob
    mixup_prob = cfg.augmentation.mixup_prob
    special_aug_ratio = cfg.augmentation.special_aug_ratio
    label_smoothing = cfg.augmentation.label_smoothing
    weight_decay = cfg.optim.weight_decay
    freeze_epoch = cfg.training.freeze_epoch
    freeze_train = cfg.training.freeze_train
    momentum = cfg.optim.momentum
    save_ckpt_freq = cfg.checkpoint.save_ckpt_freq
    save_dir = Path(cfg.logging.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(cfg.logging.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    eval_flag = True

    data_root = Path(cfg.data.data_path)
    train_annotation_path = data_root / cfg.data.train_annotation
    val_annotation_path = data_root / cfg.data.val_annotation
    train_image_path = data_root / cfg.data.train_split
    val_image_path = data_root / cfg.data.val_split

    print("使用COCO API加载数据集...")
    for ann_file in [train_annotation_path, val_annotation_path]:
        is_valid, msg = validate_coco_annotations(str(ann_file))
        if not is_valid:
            raise ValueError(f"COCO标注文件验证失败: {ann_file}, 错误: {msg}")

    train_stats = get_coco_statistics(str(train_annotation_path))
    val_stats = get_coco_statistics(str(val_annotation_path))
    print(f"训练集统计: {train_stats['num_images']} 张图片, {train_stats['num_annotations']} 个标注")
    print(f"验证集统计: {val_stats['num_images']} 张图片, {val_stats['num_annotations']} 个标注")

    class_mapping = get_coco_class_mapping(str(train_annotation_path))
    class_names = list(class_mapping.values())
    num_classes = len(class_names)
    print(f"类别数量: {num_classes}")
    print(f"类别列表: {class_names[:5]}...")

    num_train = train_stats['num_images']
    num_val = val_stats['num_images']

    anchors_str = ", ".join(str(x) for x in cfg.model.anchors)
    anchors, _ = get_anchors(anchors_str)

    model = YoloBody(anchors_mask, num_classes, backbone=cfg.model.backbone, pretrained=cfg.model.pretrained)
    if not cfg.model.pretrained:
        weights_init(model)

    yolo_loss = YOLOLoss(
        anchors,
        num_classes,
        cfg.model.input_size,
        anchors_mask,
        label_smoothing,
        cfg.training.focal_loss,
        cfg.model.focal_alpha,
        cfg.model.focal_gamma,
    )
    loss_history = LossHistory(figure_dir, model, input_size=cfg.model.input_size)
    model_train = model.train().to(device)

    unfreeze_flag = False
    if freeze_train:
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
        "adam": optim.Adam(pg0, cfg.optim.lr, betas=(momentum, 0.999)),
        "adamw": optim.AdamW(pg0, cfg.optim.lr, betas=(momentum, 0.999)),
        "sgd": optim.SGD(pg0, cfg.optim.lr, momentum=momentum, nesterov=True),
    }[cfg.optim.opt]
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})

    auto_load_model(
        checkpoint_cfg=cfg.checkpoint,
        model_without_ddp=model,
        optimizer=optimizer,
        loss_scaler=None,
        model_ema=None,
        model_ema_enabled=False,
        output_dir=cfg.logging.output_dir,
    )
    start_epoch = getattr(cfg.checkpoint, "start_epoch", 0)

    show_config(
        classes_path="COCO API",
        resume=cfg.checkpoint.resume,
        input_size=cfg.model.input_size,
        start_epoch=start_epoch,
        Freeze_Epoch=freeze_epoch,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        Freeze_Train=freeze_train,
        lr=cfg.optim.lr,
        opt=cfg.optim.opt,
        momentum=momentum,
        lr_scheduler=lr_scheduler_name,
        save_ckpt_freq=save_ckpt_freq,
        save_dir=save_dir,
        num_workers=cfg.data.num_workers,
        num_train=num_train,
        num_val=num_val,
    )

    lr_scheduler_func = get_lr_scheduler(lr_scheduler_name, cfg.optim.lr, cfg.training.epochs)
    epoch_step = num_train // cfg.training.batch_size
    epoch_step_val = (num_val + cfg.training.batch_size - 1) // cfg.training.batch_size

    print("创建COCO YOLO数据集...")
    train_dataset = CocoYoloDataset(
        root=str(train_image_path),
        annFile=str(train_annotation_path),
        input_size=cfg.model.input_size,
        epoch_length=cfg.training.epochs,
        mosaic=cfg.augmentation.mosaic,
        mixup=cfg.augmentation.mixup,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
        train=True,
        special_aug_ratio=special_aug_ratio,
    )
    val_dataset = CocoYoloDataset(
        root=str(val_image_path),
        annFile=str(val_annotation_path),
        input_size=cfg.model.input_size,
        epoch_length=cfg.training.epochs,
        mosaic=False,
        mixup=False,
        mosaic_prob=0,
        mixup_prob=0,
        train=False,
        special_aug_ratio=0,
    )

    gen: DataLoader[CocoYoloDataset] = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_yolo_batch,
    )
    gen_val: DataLoader[CocoYoloDataset] = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_yolo_batch,
    )

    val_image_ids = val_dataset.image_ids if hasattr(val_dataset, 'image_ids') else []
    eval_callback = EvalCallback(
        model,
        cfg.model.input_size,
        anchors,
        anchors_mask,
        class_names,
        num_classes,
        val_image_ids,
        figure_dir,
        device,
        val_dataset=val_dataset,
        eval_flag=eval_flag,
        period=cfg.training.eval_freq,
        num_epochs=cfg.training.epochs,
    )

    temp_map = 0
    last_map = temp_map
    for epoch in range(start_epoch, cfg.training.epochs):
        if epoch >= freeze_epoch and not unfreeze_flag and freeze_train:
            nbs = 64
            lr_limit_max = 1e-3 if cfg.optim.opt in ["adam", "adamw"] else 5e-2
            lr_limit_min = 3e-4 if cfg.optim.opt in ["adam", "adamw"] else 5e-4
            init_lr_fit = min(max(cfg.training.batch_size / nbs * cfg.optim.lr, lr_limit_min), lr_limit_max)
            lr_scheduler_func = get_lr_scheduler(lr_scheduler_name, init_lr_fit, cfg.training.epochs)
            for param in model.backbone.parameters():
                param.requires_grad = True
            epoch_step = num_train // cfg.training.batch_size
            epoch_step_val = num_val // cfg.training.batch_size
            unfreeze_flag = True

        train_dataset.epoch_now = epoch
        val_dataset.epoch_now = epoch

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
            cfg.training.epochs,
            device,
            save_ckpt_freq,
            save_dir,
            temp_map,
        )
        if temp_map is None:
            temp_map = last_map
        else:
            last_map = temp_map


def main(config_path: Optional[str] = None):
    """主函数入口 - 启动目标检测训练流程。"""
    train_detection(config_path=config_path)


if __name__ == "__main__":
    main()
