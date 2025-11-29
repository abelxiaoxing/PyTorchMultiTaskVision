import datetime
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from timm.data.mixup import Mixup
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils.model_ema import ModelEmaV3

from config import ClassificationConfig, load_classification_config
from datasets import build_dataset, build_transform
from engine import evaluate, train_one_epoch
from optim_factory import create_optimizer
from utils.checkpoint import auto_load_model, save_model
from utils.loggers import TensorboardLogger, WandbLogger
from utils.metrics import NativeScalerWithGradNormCount as NativeScaler
from utils.runtime import get_rank, get_world_size, is_main_process, setup_runtime
from utils.sampler import RASampler
from utils.scheduler import cosine_scheduler


@dataclass
class TrainingComponents:
    model: torch.nn.Module
    model_without_ddp: torch.nn.Module
    model_ema: Optional[ModelEmaV3]
    optimizer: torch.optim.Optimizer
    loss_scaler: NativeScaler
    criterion: torch.nn.Module
    mixup_fn: Optional[Mixup]
    n_parameters: int


@dataclass
class DataLoaders:
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader
    num_classes: int
    train_size: int
    val_size: int


def create_val_loader(dataset, cfg: ClassificationConfig):
    """构建验证/评估 DataLoader"""
    assert dataset is not None, "验证数据集不能为None"
    sampler_val = torch.utils.data.SequentialSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler_val,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )


def create_train_sampler(dataset_train, cfg: ClassificationConfig):
    """构建训练采样器"""
    num_tasks = get_world_size()
    global_rank = get_rank()
    if cfg.augmentation.ra_sampler:
        return RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    return torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=cfg.runtime.seed
    )


def build_schedulers(cfg: ClassificationConfig, steps_per_epoch: int):
    """返回学习率和权重衰减调度表"""
    lr_schedule_values = cosine_scheduler(
        cfg.optim.lr, cfg.optim.min_lr, cfg.training.epochs, steps_per_epoch, warmup_epochs=cfg.optim.warmup_epochs
    )
    weight_decay_end = cfg.optim.weight_decay_end if cfg.optim.weight_decay_end is not None else cfg.optim.weight_decay
    wd_schedule_values = cosine_scheduler(
        cfg.optim.weight_decay, weight_decay_end, cfg.training.epochs, steps_per_epoch
    )
    return lr_schedule_values, wd_schedule_values


def prepare_data(cfg: ClassificationConfig):
    """准备数据集和数据加载器"""
    dataset_train, dataset_val, num_classes = build_dataset(cfg=cfg)
    assert dataset_train is not None, "训练数据集不能为None"
    assert dataset_val is not None, "验证数据集不能为None"

    sampler_train = create_train_sampler(dataset_train, cfg)
    print(f"训练采样器 = {sampler_train}")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True,
    )
    data_loader_val = create_val_loader(dataset_val, cfg)
    return DataLoaders(
        train=data_loader_train,
        val=data_loader_val,
        num_classes=num_classes,
        train_size=len(dataset_train),
        val_size=len(dataset_val),
    )


def prepare_model(cfg: ClassificationConfig, runtime, num_classes: int, device):
    """准备模型、EMA、优化器和损失函数"""
    model_kwargs = {"pretrained": cfg.model.pretrained, "num_classes": num_classes}
    if cfg.model.name.startswith("efficientvit"):
        model_kwargs["drop_rate"] = cfg.model.drop_path
    elif cfg.model.name.startswith("convnext"):
        model_kwargs["drop_path_rate"] = cfg.model.drop_path

    model = create_model(cfg.model.name, **model_kwargs)
    model.to(device)

    model_ema = ModelEmaV3(model, decay=0.9995, device=device) if cfg.training.model_ema else None
    model_without_ddp = model

    if runtime.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[runtime.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数数量: {n_parameters / 1e6:.2f}M")

    optimizer = create_optimizer(
        opt=cfg.optim.opt,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        model=model_without_ddp,
    )
    loss_scaler = NativeScaler()

    mixup_fn = None
    if cfg.augmentation.mixup > 0 or cfg.augmentation.cutmix > 0.0:
        print("Mixup已激活!")
        mixup_fn = Mixup(mixup_alpha=cfg.augmentation.mixup, cutmix_alpha=cfg.augmentation.cutmix, num_classes=num_classes)

    criterion = SoftTargetCrossEntropy() if mixup_fn else LabelSmoothingCrossEntropy()
    print(f"损失函数 = {criterion}")

    return TrainingComponents(
        model=model,
        model_without_ddp=model_without_ddp,
        model_ema=model_ema,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        criterion=criterion,
        mixup_fn=mixup_fn,
        n_parameters=n_parameters,
    )


def setup_logging(cfg: ClassificationConfig):
    if is_main_process():
        Path(cfg.logging.log_dir).mkdir(parents=True, exist_ok=True)
        log_writer = TensorboardLogger(log_dir=cfg.logging.log_dir)
        wandb_logger = WandbLogger(cfg.logging) if cfg.logging.enable_wandb else None
    else:
        log_writer, wandb_logger = None, None
    return log_writer, wandb_logger


def save_checkpoint(cfg: ClassificationConfig, epoch, model_without_ddp, optimizer, loss_scaler, model_ema, num_classes, input_shape):
    """保存模型检查点"""
    if not cfg.checkpoint.save_ckpt:
        return
    save_model(
        output_dir=Path(cfg.logging.output_dir),
        input_shape=input_shape,
        model=model_without_ddp,
        optimizer=optimizer,
        epoch=epoch,
        loss_scaler=loss_scaler,
        model_ema=model_ema,
        num_classes=num_classes,
        save_ckpt_num=cfg.checkpoint.save_ckpt_num,
        save_ckpt_freq=cfg.checkpoint.save_ckpt_freq,
    )


def update_log_stats(test_stats, test_stats_ema=None):
    """更新日志统计信息"""
    log_stats = {f"test_{k}": v for k, v in test_stats.items()}

    if test_stats_ema:
        log_stats.update({f"test_{k}_ema": v for k, v in test_stats_ema.items()})

    return log_stats


def run_evaluation(cfg: ClassificationConfig, components: TrainingComponents, data: DataLoaders, device, epoch, log_writer, wandb_logger, max_accuracy, max_accuracy_ema):
    """运行评估并记录结果"""
    test_stats = evaluate(data.val, components.model, device, num_classes=data.num_classes, use_amp=cfg.training.use_amp)
    print(f"模型在{data.val_size}张测试图像上的准确率: {test_stats['acc1']:.3f}%")
    if max_accuracy < test_stats["acc1"]:
        max_accuracy = test_stats["acc1"]
        save_checkpoint(
            cfg, "best", components.model_without_ddp, components.optimizer, components.loss_scaler,
            components.model_ema, data.num_classes, (1, 3, cfg.model.input_size, cfg.model.input_size)
        )
    print(f"最高准确率: {max_accuracy:.3f}%")

    if log_writer is not None:
        log_writer.update(test_acc1=test_stats["acc1"], head="perf", step=epoch)
        log_writer.update(test_loss=test_stats["loss"], head="perf", step=epoch)

    log_stats = update_log_stats(test_stats)

    if cfg.training.model_ema and components.model_ema:
        test_stats_ema = evaluate(data.val, components.model_ema.module, device, data.num_classes, use_amp=cfg.training.use_amp)
        print(f"EMA模型在{data.val_size}张测试图像上的准确率: {test_stats_ema['acc1']:.1f}%")
        if max_accuracy_ema < test_stats_ema["acc1"]:
            max_accuracy_ema = test_stats_ema["acc1"]
            save_checkpoint(
                cfg, "best-ema", components.model_without_ddp, components.optimizer, components.loss_scaler,
                components.model_ema, data.num_classes, (1, 3, cfg.model.input_size, cfg.model.input_size)
            )
        print(f"最高EMA准确率: {max_accuracy_ema:.2f}%")
        if log_writer is not None:
            log_writer.update(test_acc1_ema=test_stats_ema["acc1"], head="perf", step=epoch)
        log_stats = update_log_stats(test_stats, test_stats_ema)

    return log_stats, max_accuracy, max_accuracy_ema


def run_training_loop(cfg: ClassificationConfig, runtime, components: TrainingComponents, data: DataLoaders, device, log_writer, wandb_logger):
    max_accuracy, max_accuracy_ema = 0.0, 0.0
    print(f"开始训练 {cfg.training.epochs} 轮")
    start_time = time.time()

    total_batch_size = cfg.training.batch_size * cfg.training.update_freq * get_world_size()
    num_training_steps_per_epoch = data.train_size // total_batch_size

    lr_schedule_values, wd_schedule_values = build_schedulers(cfg, num_training_steps_per_epoch)
    log_file = Path(cfg.logging.output_dir).parent / "log.txt"

    for epoch in range(cfg.checkpoint.start_epoch, cfg.training.epochs):
        if runtime.distributed:
            data.train.sampler.set_epoch(epoch)  # type: ignore
        if log_writer:
            log_writer.set_step(epoch * num_training_steps_per_epoch * cfg.training.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()

        train_stats = train_one_epoch(
            components.model, components.criterion, data.train, components.optimizer, device, epoch, components.loss_scaler,
            num_training_steps_per_epoch, cfg.training.update_freq, cfg.training.clip_grad, components.model_ema,
            components.mixup_fn, log_writer=log_writer, wandb_logger=wandb_logger,
            start_steps=epoch * num_training_steps_per_epoch, lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values, use_amp=cfg.training.use_amp, num_classes=data.num_classes
        )

        if (epoch + 1) % cfg.checkpoint.save_ckpt_freq == 0 or epoch + 1 == cfg.training.epochs:
            save_checkpoint(
                cfg, epoch, components.model_without_ddp, components.optimizer, components.loss_scaler,
                components.model_ema, data.num_classes, (1, 3, cfg.model.input_size, cfg.model.input_size)
            )

        eval_log_stats, max_accuracy, max_accuracy_ema = run_evaluation(
            cfg, components, data, device, epoch,
            log_writer, wandb_logger, max_accuracy, max_accuracy_ema
        )

        log_stats = {
            "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **{f"train_{k}": v for k, v in train_stats.items()},
            **eval_log_stats,
            "epoch": epoch,
            "n_parameters": f"{components.n_parameters / 1e6:.2f}M",
        }

        if is_main_process():
            if log_writer:
                log_writer.flush()
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

    if wandb_logger and cfg.logging.wandb_ckpt:
        wandb_logger.log_checkpoints()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"训练时间 {total_time_str}")


def load_model_for_inference(model_path, use_ema, device):
    """为推理（评估或移动）加载模型。"""
    print(f"从 {model_path} 加载模型...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = checkpoint['num_classes']
    if use_ema:
        model = checkpoint['model']
        model.to(device)
        model_ema_obj = ModelEmaV3(model, decay=0.9995, device=device)
        if 'model_ema' in checkpoint:
            model_ema_obj.module.load_state_dict(checkpoint['model_ema'])
            print("成功加载 EMA 模型权重。" )
        else:
            print("警告: 在检查点中未找到 'model_ema' 权重。使用基础模型权重代替。" )
            model_ema_obj.module.load_state_dict(checkpoint['model'].state_dict())
        model_to_eval = model_ema_obj.module
    else:
        model_to_eval = checkpoint['model']
        print("成功加载基础模型权重。" )

    model_to_eval.to(device)
    model_to_eval.eval()
    return model_to_eval, num_classes


def train_mode(cfg: ClassificationConfig, runtime, device):
    """训练模式入口"""
    data = prepare_data(cfg)
    components = prepare_model(cfg, runtime, data.num_classes, device)
    auto_load_model(
        checkpoint_cfg=cfg.checkpoint,
        model_without_ddp=components.model_without_ddp,
        optimizer=components.optimizer,
        loss_scaler=components.loss_scaler,
        model_ema=components.model_ema,
        model_ema_enabled=cfg.training.model_ema,
        output_dir=cfg.logging.output_dir,
    )

    log_writer, wandb_logger = setup_logging(cfg)
    total_batch_size = cfg.training.batch_size * cfg.training.update_freq * get_world_size()
    num_training_steps_per_epoch = data.train_size // total_batch_size
    print(f"学习率 = {cfg.optim.lr:.8f}, 批处理大小 = {total_batch_size}, 更新频率 = {cfg.training.update_freq}")
    print(f"训练样本数 = {data.train_size}, 验证样本数 = {data.val_size},每轮训练步数 = {num_training_steps_per_epoch}")

    run_training_loop(cfg, runtime, components, data, device, log_writer, wandb_logger)


def eval_mode(cfg: ClassificationConfig, device):
    """仅评估模式"""
    print("仅评估模式")
    model_to_eval, num_classes = load_model_for_inference(cfg.checkpoint.resume, cfg.training.model_ema, device)
    _, dataset_val, _ = build_dataset(cfg, eval_only=True)
    data_loader_val = create_val_loader(dataset_val, cfg)
    test_stats = evaluate(data_loader_val, model_to_eval, device, num_classes=num_classes, use_amp=cfg.training.use_amp)
    print(f"网络在{len(dataset_val)}张测试图像上的准确率: {test_stats['acc1']:.5f}%")


def move_mode(cfg: ClassificationConfig, device):
    """移动分类图片模式"""
    if not cfg.data.move_dir:
        raise ValueError("move_dir 未设置，请在配置中提供要移动的图片目录。")
    model, _ = load_model_for_inference(cfg.checkpoint.resume, cfg.training.model_ema, device)
    print(f"开始从 '{cfg.data.move_dir}' 移动图片...")
    empty_path = os.path.join(os.path.dirname(cfg.data.move_dir), "Empty")
    non_empty_path = os.path.join(os.path.dirname(cfg.data.move_dir), "NonEmpty")
    os.makedirs(empty_path, exist_ok=True)
    os.makedirs(non_empty_path, exist_ok=True)

    data_transform = build_transform(is_train=False, cfg=cfg)

    for file_name in os.listdir(cfg.data.move_dir):
        file_path = os.path.join(cfg.data.move_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        img = Image.open(file_path).convert("RGB")
        img = data_transform(img).unsqueeze(0).to(device)  # type: ignore

        with torch.no_grad():
            output = torch.squeeze(model(img)).cpu()
            predict = torch.softmax(output, dim=0)

        predicted_class_index = torch.argmax(predict).item()
        target_path = empty_path if predicted_class_index == 0 else non_empty_path
        shutil.move(file_path, os.path.join(target_path, file_name))
    print("图片移动完成。" )


def main(config_path: Optional[str] = None):
    cfg = load_classification_config(config_path)
    Path(cfg.logging.output_dir).mkdir(parents=True, exist_ok=True)
    device = setup_runtime(cfg.runtime)
    print(cfg)
    mode_runner = {
        "train": lambda: train_mode(cfg, cfg.runtime, device),
        "eval": lambda: eval_mode(cfg, device),
        "move": lambda: move_mode(cfg, device),
    }
    if cfg.training.mode not in mode_runner:
        raise ValueError(f"未知模式: {cfg.training.mode}")
    mode_runner[cfg.training.mode]()


if __name__ == "__main__":
    main()
