import argparse
import datetime
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import ModelEmaV3

from datasets import build_dataset, build_transform
from engine import evaluate, train_one_epoch
from optim_factory import create_optimizer
from utils.utils import (
    NativeScalerWithGradNormCount as NativeScaler,
    TensorboardLogger,
    WandbLogger,
    RASampler,
    auto_load_model,
    cosine_scheduler,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_main_process,
    save_model,
)


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


def get_args_parser():
    """
    获取命令行参数解析器
    """
    parser = argparse.ArgumentParser(
        "图像分类的训练和评估脚本", add_help=False
    )
    parser.add_argument("--batch_size", default=16, type=int, help="批处理大小")
    parser.add_argument("--epochs", default=30, type=int, help="训练轮数")
    parser.add_argument("--update_freq", default=1, type=int, help="更新频率")
    # 模型参数
    # 模型参数
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained', help="不使用预训练模型 (默认: 使用)")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--model", default="convnextv2_tiny.fcmae_ft_in22k_in1k", type=str, metavar="MODEL", help="模型名称")
    parser.add_argument("--drop_path", type=float, default=0.05, metavar="PCT", help="drop path概率")
    parser.add_argument("--input_size", default=224, type=int, help="输入图像大小")
    # EMA相关参数
    parser.add_argument('--model_ema', action='store_true', help="启用模型EMA (默认: 关闭)")

    # 优化参数
    parser.add_argument("--opt", default="lion", type=str, help="优化器类型")
    parser.add_argument("--clip_grad", type=float, default=None,help='梯度裁剪')
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="权重衰减")
    parser.add_argument("--weight_decay_end", type=float, default=5e-6, help="权重衰减终点值")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="最小学习率")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="预热轮数")

    # 数据增强参数
    parser.add_argument('--RASampler', action='store_true', help="启用RASampler (默认: 关闭)")
    parser.add_argument("--color_jitter", type=float, default=0.1, help="颜色抖动")
    parser.add_argument("--aa",type=str,default="",help='"rand-m9-mstd0.5-inc1","augmix"'),
    parser.add_argument("--reprob", type=float, default=0., metavar="PCT", help="随机擦除概率")
    parser.add_argument("--mixup", type=float, default=0., help="mixup系数")
    parser.add_argument("--cutmix", type=float, default=0., help="cutmix系数")

    # 数据集参数
    parser.add_argument("--data_path",default="../../datas/flower_photos",type=str, help="数据路径")
    parser.add_argument("--train_split_ratio",default=0.8,type=float,help="0为手动分割，其他0到1的浮点数为训练集自动分割的比例")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--seed", default=88, type=int, help="随机种子")
    parser.add_argument("--resume", default="", help="恢复训练的检查点路径")
    parser.add_argument('--auto_resume', action='store_true', help="自动从最新的检查点恢复训练 (默认: 关闭)")
    parser.add_argument('--no-save_ckpt', action='store_false', dest='save_ckpt', help="不保存检查点 (默认: 保存)")
    parser.set_defaults(save_ckpt=True)
    parser.add_argument("--save_ckpt_freq", default=1, type=int, help="保存检查点的频率")
    parser.add_argument("--save_ckpt_num", default=100, type=int, help="保存检查点的数量")
    parser.add_argument("--start_epoch", default=0, type=int, help="起始轮数")
    parser.add_argument("--mode", default="train", choices=["train", "eval", "move"], type=str, help="运行模式: train(训练), eval(评估), move(移动分类图片)")
    parser.add_argument("--move_dir", default="", type=str, help="需要移动分类的图片路径(仅move模式有效)")
    parser.add_argument("--num_workers", default=8, type=int, help="数据加载器的工作进程数")
    parser.add_argument('--no-use_amp', action='store_false', dest='use_amp', help="不使用PyTorch的AMP (默认: 使用)")
    parser.set_defaults(use_amp=True)

    # 分布式训练参数
    parser.add_argument("--world_size", default=1, type=int, help="分布式进程数")
    parser.add_argument("--local_rank", default=-1, type=int, help="本地rank")
    parser.add_argument('--dist_on_itp', action='store_true', help="在ITP上进行分布式训练 (默认: 关闭)")
    parser.add_argument("--dist_url", default="env://", help="用于设置分布式训练的URL")


    # Weights and Biases参数
    parser.add_argument('--enable_wandb', action='store_true', help="启用Weights and Biases日志记录 (默认: 关闭)")
    parser.add_argument("--project",default="classification",type=str,help="发送新运行的W&B项目的名称")
    parser.add_argument('--wandb_ckpt', action='store_true', help="将模型检查点保存为W&B工件 (默认: 关闭)")

    return parser

def setup(args):
    """初始化分布式模式和随机种子"""
    init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    return device


def create_val_loader(dataset, args):
    """构建验证/评估 DataLoader"""
    sampler_val = torch.utils.data.SequentialSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def create_train_sampler(dataset_train, args):
    """构建训练采样器"""
    num_tasks = get_world_size()
    global_rank = get_rank()
    if args.RASampler:
        return RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    return torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed
    )


def build_schedulers(args, steps_per_epoch):
    """返回学习率和权重衰减调度表"""
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, steps_per_epoch, warmup_epochs=args.warmup_epochs
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, steps_per_epoch
    )
    return lr_schedule_values, wd_schedule_values

def prepare_data(args):
    """准备数据集和数据加载器"""
    dataset_train, dataset_val, num_classes = build_dataset(args=args)
    sampler_train = create_train_sampler(dataset_train, args)
    print(f"训练采样器 = {sampler_train}")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    data_loader_val = create_val_loader(dataset_val, args)
    return DataLoaders(
        train=data_loader_train,
        val=data_loader_val,
        num_classes=num_classes,
        train_size=len(dataset_train),
        val_size=len(dataset_val),
    )

def prepare_model(args, num_classes, device):
    """准备模型、EMA、优化器和损失函数"""
    model_kwargs = {"pretrained": args.pretrained, "num_classes": num_classes}
    if args.model.startswith("efficientvit"):
        model_kwargs["drop_rate"] = args.drop_path
    elif args.model.startswith("convnext"):
        model_kwargs["drop_path_rate"] = args.drop_path

    model = create_model(args.model, **model_kwargs)
    model.to(device)

    model_ema = ModelEmaV3(model, decay=0.9995, device=device) if args.model_ema else None
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数数量: {n_parameters / 1e6:.2f}M")

    optimizer = create_optimizer(opt=args.opt, lr=args.lr, weight_decay=args.weight_decay, model=model_without_ddp)
    loss_scaler = NativeScaler()

    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0.0:
        print("Mixup已激活!")
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, num_classes=num_classes)

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

def setup_logging(args):
    if is_main_process():
        os.makedirs("train_cls/log_dir", exist_ok=True)
        log_writer = TensorboardLogger(log_dir="train_cls/log_dir")
        wandb_logger = WandbLogger(args) if args.enable_wandb else None
    else:
        log_writer, wandb_logger = None, None
    return log_writer, wandb_logger

def save_checkpoint(args, epoch, model_without_ddp, optimizer, loss_scaler, model_ema, num_classes, input_shape):
    """保存模型检查点"""
    if not args.save_ckpt:
        return
    save_model(
        output_dir=Path("train_cls/output"),
        input_shape=input_shape,
        model=model_without_ddp,
        optimizer=optimizer,
        epoch=epoch,
        loss_scaler=loss_scaler,
        model_ema=model_ema,
        num_classes=num_classes,
        save_ckpt_num=args.save_ckpt_num,
        save_ckpt_freq=args.save_ckpt_freq,
    )

def update_log_stats(test_stats, test_stats_ema=None):
    """更新日志统计信息"""
    log_stats = {f"test_{k}": v for k, v in test_stats.items()}
    
    if test_stats_ema:
        log_stats.update({f"test_{k}_ema": v for k, v in test_stats_ema.items()})
    
    return log_stats

def run_evaluation(args, components: TrainingComponents, data: DataLoaders, device, epoch, log_writer, wandb_logger, max_accuracy, max_accuracy_ema):
    """运行评估并记录结果"""
    test_stats = evaluate(data.val, components.model, device, num_classes=data.num_classes, use_amp=args.use_amp)
    print(f"模型在{data.val_size}张测试图像上的准确率: {test_stats['acc1']:.3f}%")
    if max_accuracy < test_stats["acc1"]:
        max_accuracy = test_stats["acc1"]
        save_checkpoint(
            args, "best", components.model_without_ddp, components.optimizer, components.loss_scaler,
            components.model_ema, data.num_classes, (1, 3, args.input_size, args.input_size)
        )
    print(f"最高准确率: {max_accuracy:.3f}%")

    if log_writer is not None:
        log_writer.update(test_acc1=test_stats["acc1"], head="perf", step=epoch)
        log_writer.update(test_loss=test_stats["loss"], head="perf", step=epoch)

    log_stats = update_log_stats(test_stats)

    if args.model_ema and components.model_ema:
        test_stats_ema = evaluate(data.val, components.model_ema.module, device, data.num_classes, use_amp=args.use_amp)
        print(f"EMA模型在{data.val_size}张测试图像上的准确率: {test_stats_ema['acc1']:.1f}%")
        if max_accuracy_ema < test_stats_ema["acc1"]:
            max_accuracy_ema = test_stats_ema["acc1"]
            save_checkpoint(
                args, "best-ema", components.model_without_ddp, components.optimizer, components.loss_scaler,
                components.model_ema, data.num_classes, (1, 3, args.input_size, args.input_size)
            )
        print(f"最高EMA准确率: {max_accuracy_ema:.2f}%")
        if log_writer is not None:
            log_writer.update(test_acc1_ema=test_stats_ema["acc1"], head="perf", step=epoch)
        log_stats = update_log_stats(test_stats, test_stats_ema)

    return log_stats, max_accuracy, max_accuracy_ema

def run_training_loop(args, components: TrainingComponents, data: DataLoaders, device, log_writer, wandb_logger):
    max_accuracy, max_accuracy_ema = 0.0, 0.0
    print(f"开始训练 {args.epochs} 轮")
    start_time = time.time()

    total_batch_size = args.batch_size * args.update_freq * get_world_size()
    num_training_steps_per_epoch = data.train_size // total_batch_size

    lr_schedule_values, wd_schedule_values = build_schedulers(args, num_training_steps_per_epoch)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data.train.sampler.set_epoch(epoch)
        if log_writer:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()

        train_stats = train_one_epoch(
            components.model, components.criterion, data.train, components.optimizer, device, epoch, components.loss_scaler,
            args.clip_grad, components.model_ema, components.mixup_fn, log_writer=log_writer, wandb_logger=wandb_logger,
            start_steps=epoch * num_training_steps_per_epoch, lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values, num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq, use_amp=args.use_amp, num_classes=data.num_classes
        )

        if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            save_checkpoint(
                args, epoch, components.model_without_ddp, components.optimizer, components.loss_scaler,
                components.model_ema, data.num_classes, (1, 3, args.input_size, args.input_size)
            )

        eval_log_stats, max_accuracy, max_accuracy_ema = run_evaluation(
            args, components, data, device, epoch,
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
            with open(os.path.join("train_cls/log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

    if wandb_logger and args.wandb_ckpt:
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


def train_mode(args, device):
    """训练模式入口"""
    data = prepare_data(args)
    components = prepare_model(args, data.num_classes, device)
    auto_load_model(
        args=args,
        model_without_ddp=components.model_without_ddp,
        optimizer=components.optimizer,
        loss_scaler=components.loss_scaler,
        model_ema=components.model_ema,
    )

    log_writer, wandb_logger = setup_logging(args)
    total_batch_size = args.batch_size * args.update_freq * get_world_size()
    num_training_steps_per_epoch = data.train_size // total_batch_size
    print(f"学习率 = {args.lr:.8f}, 批处理大小 = {total_batch_size}, 更新频率 = {args.update_freq}")
    print(f"训练样本数 = {data.train_size}, 验证样本数 = {data.val_size},每轮训练步数 = {num_training_steps_per_epoch}")

    run_training_loop(args, components, data, device, log_writer, wandb_logger)


def eval_mode(args, device):
    """仅评估模式"""
    print("仅评估模式")
    model_to_eval, num_classes = load_model_for_inference(args.resume, args.model_ema, device)
    _, dataset_val, _ = build_dataset(args, eval_only=True)
    data_loader_val = create_val_loader(dataset_val, args)
    test_stats = evaluate(data_loader_val, model_to_eval, device, num_classes=num_classes, use_amp=args.use_amp)
    print(f"网络在{len(dataset_val)}张测试图像上的准确率: {test_stats['acc1']:.5f}%")


def move_mode(args, device):
    """移动分类图片模式"""
    model, _ = load_model_for_inference(args.resume, args.model_ema, device)
    print(f"开始从 '{args.move_dir}' 移动图片...")
    empty_path = os.path.join(os.path.dirname(args.move_dir), "Empty")
    non_empty_path = os.path.join(os.path.dirname(args.move_dir), "NonEmpty")
    os.makedirs(empty_path, exist_ok=True)
    os.makedirs(non_empty_path, exist_ok=True)

    data_transform = build_transform(is_train=False, args=args)

    for file_name in os.listdir(args.move_dir):
        file_path = os.path.join(args.move_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        img = Image.open(file_path).convert("RGB")
        img = data_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = torch.squeeze(model(img)).cpu()
            predict = torch.softmax(output, dim=0)

        predicted_class_index = torch.argmax(predict).item()
        target_path = empty_path if predicted_class_index == 0 else non_empty_path
        shutil.move(file_path, os.path.join(target_path, file_name))
    print("图片移动完成。" )


def main(args):
    """模式分发主入口"""
    device = setup(args)
    print(args)
    mode_runner = {
        "train": train_mode,
        "eval": eval_mode,
        "move": move_mode,
    }
    mode_runner[args.mode](args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("分类训练和评估脚本", parents=[get_args_parser()])
    args = parser.parse_args()
    Path("./train_cls/output").mkdir(parents=True, exist_ok=True)
    main(args)
