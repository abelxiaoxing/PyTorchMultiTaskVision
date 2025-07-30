import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEmaV3
from optim_factory import create_optimizer
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils


def str2bool(v):
    """
    将字符串转换为布尔值
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("期望布尔值")


def get_args_parser():
    """
    获取命令行参数解析器
    """
    parser = argparse.ArgumentParser(
        "图像分类的训练和评估脚本", add_help=False
    )
    parser.add_argument("--batch_size", default=16, type=int, help="批处理大小")
    parser.add_argument("--epochs", default=100, type=int, help="训练轮数")
    parser.add_argument("--update_freq", default=1, type=int, help="更新频率")
    # 模型参数
    parser.add_argument("--pretrained", default=True, type=bool, help="是否使用预训练模型")
    parser.add_argument("--model", default="efficientvit_m0", type=str, metavar="MODEL", help="模型名称")
    parser.add_argument("--drop_path", type=float, default=0.05, metavar="PCT", help="drop path概率")
    parser.add_argument("--input_size", default=224, type=int, help="输入图像大小")
    # EMA相关参数
    parser.add_argument("--model_ema", type=str2bool, default=False, help="是否使用EMA")

    # 优化参数
    parser.add_argument("--opt", default="lion", type=str, help="优化器类型")
    parser.add_argument("--clip_grad", type=float, default=None,help='梯度裁剪')
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="权重衰减")
    parser.add_argument("--weight_decay_end", type=float, default=5e-6, help="权重衰减终点值")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="最小学习率")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="预热轮数")

    # 数据增强参数
    parser.add_argument("--RASampler", default=False, type=bool, help="是否使用RASampler")
    parser.add_argument("--color_jitter", type=float, default=0.3, help="颜色抖动")
    parser.add_argument("--aa",type=str,default="",help='"rand-m9-mstd0.5-inc1","augmix"'),
    parser.add_argument("--reprob", type=float, default=0., metavar="PCT", help="随机擦除概率")
    parser.add_argument("--mixup", type=float, default=0., help="mixup系数")
    parser.add_argument("--cutmix", type=float, default=0., help="cutmix系数")

    # 数据集参数
    parser.add_argument("--data_path",default="../../datas/CatsDogs_mini",type=str, help="数据路径")
    parser.add_argument("--train_split_rato",default=0.,type=float,help="0为手动分割，其他0到1的浮点数为训练集自动分割的比例")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--seed", default=88, type=int, help="随机种子")
    parser.add_argument("--resume", default="", help="恢复训练的检查点路径")
    parser.add_argument("--auto_resume", type=str2bool, default=False, help="是否自动恢复训练")
    parser.add_argument("--save_ckpt", type=str2bool, default=True, help="是否保存检查点")
    parser.add_argument("--save_ckpt_freq", default=5, type=int, help="保存检查点的频率")
    parser.add_argument("--save_ckpt_num", default=20, type=int, help="保存检查点的数量")
    parser.add_argument("--start_epoch", default=0, type=int, help="起始轮数")
    parser.add_argument("--eval", type=str2bool, default=False, help="仅评估模式")
    parser.add_argument("--num_workers", default=8, type=int, help="数据加载器的工作进程数")
    parser.add_argument("--use_amp",type=str2bool,default=True,help="是否使用PyTorch的AMP")

    # 分布式训练参数
    parser.add_argument("--world_size", default=1, type=int, help="分布式进程数")
    parser.add_argument("--local_rank", default=-1, type=int, help="本地rank")
    parser.add_argument("--dist_on_itp", type=str2bool, default=False, help="是否在ITP上进行分布式训练")
    parser.add_argument("--dist_url", default="env://", help="用于设置分布式训练的URL")


    # Weights and Biases参数
    parser.add_argument("--enable_wandb",type=str2bool,default=False,help="启用Weights and Biases日志记录")
    parser.add_argument("--project",default="classification",type=str,help="发送新运行的W&B项目的名称")
    parser.add_argument("--wandb_ckpt",type=str2bool,default=False,help="将模型检查点保存为W&B工件")

    return parser


def main(args):
    """
    主函数
    """
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # 设置随机种子
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 构建数据集
    dataset_train,dataset_val, num_classes = build_dataset(args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if args.RASampler:
        sampler_train = utils.RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
            seed=args.seed,
        )
    print("训练采样器 = %s" % str(sampler_train))

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 初始化日志写入器
    if global_rank == 0:
        os.makedirs("train_cls/log_dir", exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir="train_cls/log_dir")
    else:
        log_writer = None

    # 初始化wandb日志记录器
    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    # 创建训练数据加载器
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    input_shape = [
        1,
    ] + list(next(iter(data_loader_train))[0].shape[1:])

    # 创建验证数据加载器
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 配置mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0
    if mixup_active:
        print("Mixup已激活!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            num_classes=num_classes,
        )

    # 模型参数配置
    model_kwargs = {"pretrained": args.pretrained, "num_classes": num_classes}

    if args.model.startswith("efficientvit"):
        model_kwargs["drop_rate"] = args.drop_path
    elif args.model.startswith("convnext"):
        model_kwargs["drop_path_rate"] = args.drop_path

    # 创建模型
    model = create_model(args.model, **model_kwargs)

    model.to(device)

    # EMA模型
    model_ema = None
    if args.model_ema:
        # 重要：在cuda()、DP包装器和AMP之后但在SyncBN和DDP包装器之前创建EMA模型
        model_ema = ModelEmaV3(model,decay=0.9995,device=device,)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(f"Model = {str(model_without_ddp)}")
    print("参数数量:", n_parameters)

    # 计算训练相关参数
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("学习率 = %.8f" % args.lr)
    print("批处理大小 = %d" % total_batch_size)
    print("更新频率 = %d" % args.update_freq)
    print("训练样本数 = %d" % len(dataset_train))
    print("每轮训练步数 = %d" % num_training_steps_per_epoch)


    # 分布式训练设置
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    # 创建优化器
    optimizer = create_optimizer(
        opt = args.opt,
        lr = args.lr,
        weight_decay = args.weight_decay,
        model = model_without_ddp,
    )

    # 损失缩放器
    loss_scaler = NativeScaler()  # 如果args.use_amp为False，则不会使用

    print("使用余弦学习率调度器")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )

    # 权重衰减调度
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch,
    )
    print("最大权重衰减 = %.7f, 最小权重衰减 = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # 损失函数
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = LabelSmoothingCrossEntropy()
        # criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1,5],dtype=torch.float,device=args.device))
    print(f"损失函数 = {criterion}")

    # 自动加载模型
    utils.auto_load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=model_ema,
    )

    # 仅评估模式
    if args.eval:
        print(f"仅评估模式")
        if args.model_ema:
            test_stats = evaluate(data_loader_val, model_ema.module, device, num_classes=num_classes,use_amp=args.use_amp)
        else:
            test_stats = evaluate(data_loader_val, model, device, num_classes=num_classes,use_amp=args.use_amp)
        print(f"网络在{len(dataset_val)}张测试图像上的准确率: {test_stats['acc1']:.5f}%")
        return

    max_accuracy = 0.0
    if args.model_ema:
        max_accuracy_ema = 0.0

    print("开始训练 %d 轮" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()
        # 训练一轮
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            log_writer=log_writer,
            wandb_logger=wandb_logger,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            use_amp=args.use_amp,
            num_classes=num_classes,
        )
        # 保存检查点
        if args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args,
                    input_shape=input_shape,
                    model=model_without_ddp,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss_scaler=loss_scaler,
                    model_ema=model_ema,
                    num_classes=num_classes,
                )
        # 验证
        test_stats = evaluate(
            data_loader_val,
            model,
            device,
            num_classes=num_classes,
            use_amp=args.use_amp,
        )
        print(
            f"模型在{len(dataset_val)}张测试图像上的准确率: {test_stats['acc1']:.3f}%"
        )
        # 保存最佳模型
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.save_ckpt:
                utils.save_model(
                    args=args,
                    input_shape=input_shape,
                    model=model_without_ddp,
                    optimizer=optimizer,
                    epoch="best",
                    loss_scaler=loss_scaler,
                    model_ema=model_ema,
                    num_classes=num_classes,
                )
        print(f"最高准确率: {max_accuracy:.3f}%")

        # 更新日志
        if log_writer is not None:
            log_writer.update(test_acc1=test_stats["acc1"], head="perf", step=epoch)
            # log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
            log_writer.update(test_loss=test_stats["loss"], head="perf", step=epoch)
        log_stats = {
            "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": f"{n_parameters / 1e6:.2f}M",
        }

        # 如果启用了EMA评估，重复测试EMA模型
        if args.model_ema:
            test_stats_ema = evaluate(data_loader_val, model_ema.module, device,num_classes, use_amp=args.use_amp)
            print(f"EMA模型在{len(dataset_val)}张测试图像上的准确率: {test_stats_ema['acc1']:.1f}%")
            if max_accuracy_ema < test_stats_ema["acc1"]:
                max_accuracy_ema = test_stats_ema["acc1"]
                if args.save_ckpt:
                    utils.save_model(
                        args=args,
                        input_shape=input_shape,
                        model=model_without_ddp,
                        optimizer=optimizer,
                        epoch="best-ema",
                        loss_scaler=loss_scaler,
                        model_ema=model_ema,
                        num_classes=num_classes,
                    )
                print(f"最高EMA准确率: {max_accuracy_ema:.2f}%")
            if log_writer is not None:
                log_writer.update(
                    test_acc1_ema=test_stats_ema["acc1"], head="perf", step=epoch
                )
            log_stats.update(
                {**{f"test_{k}_ema": v for k, v in test_stats_ema.items()}}
            )

        # 写入日志文件
        if utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join("train_cls/log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        # 记录wandb日志
        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

    # 保存wandb检查点
    if wandb_logger and args.wandb_ckpt and args.save_ckpt:
        wandb_logger.log_checkpoints()

    # 计算训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("训练时间 {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "分类训练和评估脚本", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    Path("./train_cls/output").mkdir(parents=True, exist_ok=True)
    main(args)
