import datetime
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

from config import DetectionConfig, load_detection_config
from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.coco_dataloader import CocoYoloDataset
from utils.coco_utils import get_coco_class_mapping, get_coco_statistics, validate_coco_annotations
from utils.checkpoint import auto_load_model
from utils.loggers import show_config
from utils.runtime import get_rank, get_world_size, is_main_process, setup_runtime
from utils.utils_fit import fit_one_epoch
from utils.vision import get_anchors
from utils.yolo_transforms import collate_yolo_batch


@dataclass
class DetectionComponents:
    model: nn.Module
    model_train: nn.Module
    optimizer: optim.Optimizer
    loss_fn: YOLOLoss
    lr_scheduler_func: Any
    loss_history: LossHistory


@dataclass
class DetectionDataLoaders:
    train: DataLoader
    val: DataLoader
    num_classes: int
    train_size: int
    val_size: int
    class_names: list[str]
    anchors: Any
    anchors_mask: list[list[int]]
    val_image_ids: list[Any]
    val_dataset: CocoYoloDataset


@dataclass
class DetectionTrainState:
    lr_scheduler_func: Any
    unfreeze_flag: bool
    epoch_step: int
    epoch_step_val: int


def setup_detection_logging(cfg: DetectionConfig):
    output_dir = Path(cfg.logging.output_dir)
    figure_dir = Path(cfg.logging.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir.parent / "log_det.json"
    return output_dir, figure_dir, log_file



def prepare_detection_data(cfg: DetectionConfig) -> DetectionDataLoaders:
    """准备 COCO 数据集、校验标注并构建 DataLoader。"""
    if not cfg.data.data_path:
        raise ValueError("data_path 不能为空，请提供 COCO 数据路径。")

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

    anchors_mask = [list(mask) for mask in cfg.model.anchor_mask]
    anchors_str = ", ".join(str(x) for x in cfg.model.anchors)
    anchors, _ = get_anchors(anchors_str)

    print("创建COCO YOLO数据集...")
    train_dataset = CocoYoloDataset(
        root=str(train_image_path),
        annFile=str(train_annotation_path),
        input_size=cfg.model.input_size,
        epoch_length=cfg.training.epochs,
        mosaic=cfg.augmentation.mosaic,
        mixup=cfg.augmentation.mixup,
        mosaic_prob=cfg.augmentation.mosaic_prob,
        mixup_prob=cfg.augmentation.mixup_prob,
        train=True,
        special_aug_ratio=cfg.augmentation.special_aug_ratio,
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

  
    sampler_train = DistributedSampler(
        train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True
    ) if cfg.runtime.distributed else None
    sampler_val = SequentialSampler(val_dataset)

    drop_last_train = len(train_dataset) > cfg.training.batch_size
    train_loader: DataLoader[CocoYoloDataset] = DataLoader(
        train_dataset,
        shuffle=sampler_train is None,
        sampler=sampler_train,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=drop_last_train,
        collate_fn=collate_yolo_batch,
    )
    val_loader: DataLoader[CocoYoloDataset] = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=sampler_val,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_yolo_batch,
    )

    val_image_ids = val_dataset.image_ids if hasattr(val_dataset, "image_ids") else []
    return DetectionDataLoaders(
        train=train_loader,
        val=val_loader,
        num_classes=num_classes,
        train_size=train_stats["num_images"],
        val_size=val_stats["num_images"],
        class_names=class_names,
        anchors=anchors,
        anchors_mask=anchors_mask,
        val_image_ids=val_image_ids,
        val_dataset=val_dataset,
    )


def build_detection_optimizer(cfg: DetectionConfig, model: nn.Module):
    """创建与分类端风格一致的优化器构建函数。"""
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    optimizer = {
        "adam": optim.Adam(pg0, cfg.optim.lr, betas=(cfg.optim.momentum, 0.999)),
        "adamw": optim.AdamW(pg0, cfg.optim.lr, betas=(cfg.optim.momentum, 0.999)),
        "sgd": optim.SGD(pg0, cfg.optim.lr, momentum=cfg.optim.momentum, nesterov=True),
    }[cfg.optim.opt]
    optimizer.add_param_group({"params": pg1, "weight_decay": cfg.optim.weight_decay})
    optimizer.add_param_group({"params": pg2})
    return optimizer


def build_detection_scheduler(cfg: DetectionConfig, base_lr: float):
    return get_lr_scheduler(cfg.training.lr_scheduler, base_lr, cfg.training.epochs)


def prepare_detection_model(
    cfg: DetectionConfig,
    num_classes: int,
    device,
    anchors,
    anchors_mask,
    figure_dir: Path,
) -> tuple[DetectionComponents, DetectionTrainState]:
    """构建模型、损失、优化器与调度器。"""
    model = YoloBody(anchors_mask, num_classes, backbone=cfg.model.backbone, pretrained=cfg.model.pretrained)
    if not cfg.model.pretrained:
        weights_init(model)

    yolo_loss = YOLOLoss(
        anchors,
        num_classes,
        cfg.model.input_size,
        anchors_mask,
        cfg.augmentation.label_smoothing,
        cfg.training.focal_loss,
        cfg.model.focal_alpha,
        cfg.model.focal_gamma,
    )
    loss_history = LossHistory(figure_dir, model, input_size=cfg.model.input_size)
    model_train = model.train().to(device)

    if cfg.training.freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    optimizer = build_detection_optimizer(cfg, model)
    lr_scheduler_func = build_detection_scheduler(cfg, cfg.optim.lr)

    components = DetectionComponents(
        model=model,
        model_train=model_train,
        optimizer=optimizer,
        loss_fn=yolo_loss,
        lr_scheduler_func=lr_scheduler_func,
        loss_history=loss_history,
    )
    state = DetectionTrainState(
        lr_scheduler_func=lr_scheduler_func,
        unfreeze_flag=not cfg.training.freeze_train,
        epoch_step=0,
        epoch_step_val=0,
    )
    return components, state


def build_eval_callback(
    cfg: DetectionConfig,
    components: DetectionComponents,
    data: DetectionDataLoaders,
    figure_dir: Path,
    device,
    eval_flag: bool,
):
    return EvalCallback(
        components.model,
        cfg.model.input_size,
        data.anchors,
        data.anchors_mask,
        data.class_names,
        data.num_classes,
        data.val_image_ids,
        figure_dir,
        device,
        val_dataset=data.val_dataset,
        eval_flag=eval_flag,
        period=cfg.training.eval_freq,
        num_epochs=cfg.training.epochs,
    )


def maybe_unfreeze_backbone(
    cfg: DetectionConfig,
    model: nn.Module,
    epoch: int,
    train_steps: int,
    val_steps: int,
    state: DetectionTrainState,
) -> DetectionTrainState:
    """在到达指定 epoch 后解冻 backbone，并更新调度与步数。"""
    if state.unfreeze_flag or not cfg.training.freeze_train or epoch < cfg.training.freeze_epoch:
        return state

    nbs = 64
    lr_limit_max = 1e-3 if cfg.optim.opt in ["adam", "adamw"] else 5e-2
    lr_limit_min = 3e-4 if cfg.optim.opt in ["adam", "adamw"] else 5e-4
    init_lr_fit = min(max(cfg.training.batch_size / nbs * cfg.optim.lr, lr_limit_min), lr_limit_max)
    for param in model.backbone.parameters():
        param.requires_grad = True

    return DetectionTrainState(
        lr_scheduler_func=build_detection_scheduler(cfg, init_lr_fit),
        unfreeze_flag=True,
        epoch_step=train_steps,
        epoch_step_val=val_steps,
    )


def run_detection_training_loop(
    cfg: DetectionConfig,
    components: DetectionComponents,
    data: DetectionDataLoaders,
    eval_callback: EvalCallback,
    device,
    output_dir: Path,
    log_file: Path,
    initial_state: DetectionTrainState,
    is_main: bool,
):
    """与分类端一致的训练主循环。"""
    train_steps = max(1, len(data.train))
    val_steps = max(1, len(data.val))
    total_batch_size = cfg.training.batch_size * get_world_size()
    state = DetectionTrainState(
        lr_scheduler_func=initial_state.lr_scheduler_func,
        unfreeze_flag=initial_state.unfreeze_flag,
        epoch_step=train_steps,
        epoch_step_val=val_steps,
    )

    start_epoch = getattr(cfg.checkpoint, "start_epoch", 0)
    print(f"学习率 = {cfg.optim.lr:.6f}, 总批次大小 = {total_batch_size}")
    print(
        f"训练样本数 = {data.train_size}, 验证样本数 = {data.val_size}, "
        f"每轮训练步数 = {state.epoch_step}"
    )

    temp_map = 0
    best_map = 0
    start_time = time.time()
    for epoch in range(start_epoch, cfg.training.epochs):
        state = maybe_unfreeze_backbone(cfg, components.model, epoch, train_steps, val_steps, state)

        if isinstance(data.train.sampler, DistributedSampler):
            data.train.sampler.set_epoch(epoch)
        if hasattr(data.train.dataset, "epoch_now"):
            data.train.dataset.epoch_now = epoch  # type: ignore[attr-defined]
        if hasattr(data.val.dataset, "epoch_now"):
            data.val.dataset.epoch_now = epoch  # type: ignore[attr-defined]

        set_optimizer_lr(components.optimizer, state.lr_scheduler_func, epoch)

        eval_cb = eval_callback if is_main else None
        run_validation = (not cfg.runtime.distributed) or is_main
        train_loss_avg, val_loss_avg, temp_map = fit_one_epoch(
            components.model_train,
            components.model,
            components.loss_fn,
            components.loss_history,
            eval_cb,
            components.optimizer,
            epoch,
            state.epoch_step,
            state.epoch_step_val,
            data.train,
            data.val,
            cfg.training.epochs,
            device,
            cfg.checkpoint.save_ckpt_freq,
            output_dir,
            temp_map,
            is_main_process=is_main,
            run_validation=run_validation,
        )
        if is_main:
            if temp_map is None:
                temp_map = best_map
            best_map = max(best_map, temp_map)

        if is_main:
            log_stats = {
                "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "train_loss": train_loss_avg,
                "val_loss": val_loss_avg,
                "map": temp_map,
                "best_map": best_map,
                "n_parameters": f"{sum(p.numel() for p in components.model.parameters() if p.requires_grad) / 1e6:.2f}M",
            }
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"训练时间 {total_time_str}")


def train_detection(config_path: Optional[str] = None):
    """
    目标检测训练入口，使用 TOML 配置。
    """
    cfg = load_detection_config(config_path)
    output_dir, figure_dir, log_file = setup_detection_logging(cfg)
    device = setup_runtime(cfg.runtime)
    main_process = is_main_process()

    data = prepare_detection_data(cfg)
    components, state = prepare_detection_model(
        cfg, data.num_classes, device, data.anchors, data.anchors_mask, figure_dir
    )

    auto_load_model(
        checkpoint_cfg=cfg.checkpoint,
        model_without_ddp=components.model,
        optimizer=components.optimizer,
        loss_scaler=None,
        model_ema=None,
        model_ema_enabled=False,
        output_dir=cfg.logging.output_dir,
    )

    if cfg.runtime.distributed:
        components.model_train = DDP(
            components.model,
            device_ids=[cfg.runtime.gpu] if cfg.runtime.gpu is not None else None,
        )

    show_config(
        classes_path="COCO API",
        resume=cfg.checkpoint.resume,
        input_size=cfg.model.input_size,
        start_epoch=getattr(cfg.checkpoint, "start_epoch", 0),
        Freeze_Epoch=cfg.training.freeze_epoch,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        Freeze_Train=cfg.training.freeze_train,
        lr=cfg.optim.lr,
        opt=cfg.optim.opt,
        momentum=cfg.optim.momentum,
        lr_scheduler=cfg.training.lr_scheduler,
        save_ckpt_freq=cfg.checkpoint.save_ckpt_freq,
        save_dir=output_dir,
        num_workers=cfg.data.num_workers,
        num_train=data.train_size,
        num_val=data.val_size,
    )

    eval_callback = build_eval_callback(cfg, components, data, figure_dir, device, eval_flag=main_process)
    run_detection_training_loop(
        cfg,
        components,
        data,
        eval_callback,
        device,
        output_dir,
        log_file,
        state,
        is_main=main_process,
    )


def main(config_path: Optional[str] = None):
    """主函数入口 - 启动目标检测训练流程。"""
    train_detection(config_path=config_path)


if __name__ == "__main__":
    main()
