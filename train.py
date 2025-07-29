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
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Training and evaluation script for image classification", add_help=False
    )
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--update_freq", default=1, type=int)
    # Model parameters
    parser.add_argument("--pretrained", default=True, type=bool)
    parser.add_argument("--model", default="efficientvit_m0", type=str, metavar="MODEL")
    parser.add_argument("--drop_path", type=float, default=0.05, metavar="PCT")
    parser.add_argument("--input_size", default=224, type=int)
    # EMA related parameters
    parser.add_argument("--model_ema", type=str2bool, default=False)

    # Optimization parameters
    parser.add_argument("--opt", default="lion", type=str)
    parser.add_argument("--clip_grad", type=float, default=None,help='梯度裁剪')
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--weight_decay_end", type=float, default=5e-6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    # Augmentation parameters
    parser.add_argument("--RASampler", default=False, type=bool)
    parser.add_argument("--color_jitter", type=float, default=0.3)
    parser.add_argument("--aa",type=str,default="",help='"rand-m9-mstd0.5-inc1","augmix"'),
    parser.add_argument("--reprob", type=float, default=0., metavar="PCT")
    parser.add_argument("--mixup", type=float, default=0.)
    parser.add_argument("--cutmix", type=float, default=0.)

    # Dataset parameters
    parser.add_argument("--data_path",default="../../datas/CatsDogs_mini",type=str)
    parser.add_argument("--train_split_rato",default=0.,type=float,help="0为手动分割，其他0到1的浮点数为训练集自动分割的比例")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=88, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto_resume", type=str2bool, default=False)
    parser.add_argument("--save_ckpt", type=str2bool, default=True)
    parser.add_argument("--save_ckpt_freq", default=5, type=int)
    parser.add_argument("--save_ckpt_num", default=20, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", type=str2bool, default=False)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--use_amp",type=str2bool,default=True,help="Use PyTorch's AMP or not")

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", type=str2bool, default=False)
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")


    # Weights and Biases arguments
    parser.add_argument("--enable_wandb",type=str2bool,default=False,help="enable logging to Weights and Biases")
    parser.add_argument("--project",default="classification",type=str,help="The name of the W&B project where you're sending the new run.")
    parser.add_argument("--wandb_ckpt",type=str2bool,default=False,help="Save model checkpoints as W&B Artifacts.")

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # 设置随机种子
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

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
    print("Sampler_train = %s" % str(sampler_train))

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0:
        os.makedirs("train_cls/log_dir", exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir="train_cls/log_dir")
    else:
        log_writer = None

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

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

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            num_classes=num_classes,
        )

    model_kwargs = {"pretrained": args.pretrained, "num_classes": num_classes}

    if args.model.startswith("efficientvit"):
        model_kwargs["drop_rate"] = args.drop_path
    elif args.model.startswith("convnext"):
        model_kwargs["drop_path_rate"] = args.drop_path

    model = create_model(args.model, **model_kwargs)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV3(model,decay=0.9995,device=device,)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(f"Model = {str(model_without_ddp)}")
    print("number of params:", n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    optimizer = create_optimizer(
        opt = args.opt,
        lr = args.lr,
        weight_decay = args.weight_decay,
        model = model_without_ddp,
    )

    loss_scaler = NativeScaler()  # if args.use_amp is False, this won't be used

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch,
    )
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = LabelSmoothingCrossEntropy()
        # criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1,5],dtype=torch.float,device=args.device))
    print(f"criterion = {criterion}")

    utils.auto_load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=model_ema,
    )

    if args.eval:
        print(f"Eval only mode")
        if args.model_ema:
            test_stats = evaluate(data_loader_val, model_ema.module, device, num_classes=num_classes,use_amp=args.use_amp)
        else:
            test_stats = evaluate(data_loader_val, model, device, num_classes=num_classes,use_amp=args.use_amp)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    max_accuracy = 0.0
    if args.model_ema:
        max_accuracy_ema = 0.0

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()
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
        test_stats = evaluate(
            data_loader_val,
            model,
            device,
            num_classes=num_classes,
            use_amp=args.use_amp,
        )
        print(
            f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.3f}%"
        )
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
        print(f"Max accuracy: {max_accuracy:.3f}%")

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

        # repeat testing routines for EMA, if ema eval is turned on
        if args.model_ema:
            test_stats_ema = evaluate(data_loader_val, model_ema.module, device,num_classes, use_amp=args.use_amp)
            print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
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
                print(f"Max EMA accuracy: {max_accuracy_ema:.2f}%")
            if log_writer is not None:
                log_writer.update(
                    test_acc1_ema=test_stats_ema["acc1"], head="perf", step=epoch
                )
            log_stats.update(
                {**{f"test_{k}_ema": v for k, v in test_stats_ema.items()}}
            )

        if utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join("train_cls/log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

    if wandb_logger and args.wandb_ckpt and args.save_ckpt:
        wandb_logger.log_checkpoints()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Classification training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    Path("./train_cls/output").mkdir(parents=True, exist_ok=True)
    main(args)
