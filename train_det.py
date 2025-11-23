import argparse
from pathlib import Path
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
from utils.coco_dataloader import CocoYoloDataset
from utils.yolo_transforms import collate_yolo_batch
from utils.coco_utils import get_coco_class_mapping, get_coco_statistics, validate_coco_annotations
from utils.checkpoint import auto_load_model
from utils.loggers import show_config
from utils.runtime import setup_runtime
from utils.utils_fit import fit_one_epoch
from utils.vision import get_anchors

DEFAULT_ANCHORS = "12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401"
DEFAULT_ANCHOR_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]


def get_args_parser():
    parser = argparse.ArgumentParser("检测训练脚本", add_help=False)
    parser.add_argument("--device", default="cuda", help="训练/评估设备")
    parser.add_argument("--data_path", type=str, required=True, help="COCO 数据根目录")
    parser.add_argument("--resume", default="", help="checkpoint 路径")
    parser.add_argument("--auto_resume", action="store_true", help="自动从 output_dir 中最新权重恢复")
    parser.add_argument("--input_size", type=int, default=640, help="输入尺寸")
    parser.add_argument("--backbone", type=str, default="efficientvit_b0", help="YOLO 骨干网络")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="优化器类型")
    parser.add_argument("--save_ckpt_freq", type=int, default=10, help="保存检查点频率")
    parser.add_argument("--eval_freq", type=int, default=2, help="评估频率")
    parser.add_argument("--focal_loss", action="store_true", help="启用 focal loss")
    parser.add_argument("--mosaic", dest="mosaic", action="store_true", default=True, help="启用 mosaic 增强")
    parser.add_argument("--no-mosaic", dest="mosaic", action="store_false", help="禁用 mosaic 增强")
    parser.add_argument("--mixup", dest="mixup", action="store_true", default=True, help="启用 mixup 增强")
    parser.add_argument("--no-mixup", dest="mixup", action="store_false", help="禁用 mixup 增强")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers 数量")
    parser.add_argument("--seed", type=int, default=88, help="随机种子")
    parser.add_argument("--output_dir", default="train_det/output", help="模型与日志输出目录")
    parser.add_argument("--figure_dir", default="train_det/figure", help="曲线/可视化输出目录")
    parser.add_argument("--dist_url", default="env://", help="分布式 URL")
    parser.add_argument("--dist_on_itp", action="store_true", help="ITP 分布式模式")
    return parser


RUNTIME_DEFAULTS = {
    "world_size": 1,
    "local_rank": -1,
    "model_ema": False,
    "start_epoch": 0,
}


def train_detection(args=None, **overrides):
    """
    直接运行目标检测训练。支持传入 argparse.Namespace 或以关键字参数调用。
    """
    parser = get_args_parser()
    defaults = {action.dest: action.default for action in parser._actions if action.dest != "help"}
    valid_keys = set(defaults.keys()) | set(RUNTIME_DEFAULTS.keys())

    if args is None:
        args = argparse.Namespace(**{**defaults, **RUNTIME_DEFAULTS})
    elif isinstance(args, argparse.Namespace):
        # 补齐运行时缺省值
        for key, value in RUNTIME_DEFAULTS.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    else:
        raise TypeError("train_detection 期望 argparse.Namespace 或关键字参数调用。")

    unknown = set(overrides.keys()) - valid_keys
    if unknown:
        raise ValueError(f"收到未知参数: {sorted(unknown)}")

    for key, value in overrides.items():
        setattr(args, key, value)

    if not getattr(args, "data_path", ""):
        raise ValueError("data_path 不能为空，请提供 COCO 数据路径。")

    device = setup_runtime(args)
    lr_scheduler = "cosine"
    pretrained = True
    focal_alpha = 0.25
    focal_gamma = 2
    anchors_mask = DEFAULT_ANCHOR_MASK
    mosaic_prob = 0.5
    mixup_prob = 0.5
    special_aug_ratio = 0.7
    label_smoothing = 0.01
    weight_decay = 5e-4
    freeze_epoch = int(args.epochs / 2)
    freeze_train = True
    momentum = 0.937
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    eval_flag = True

    data_root = Path(args.data_path)
    train_annotation_path = data_root / "annotations" / "instances_val2017.json"
    val_annotation_path = data_root / "annotations" / "instances_val2017.json"
    train_image_path = data_root / "val2017"
    val_image_path = data_root / "val2017"

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

    anchors, _ = get_anchors(DEFAULT_ANCHORS)

    model = YoloBody(anchors_mask, num_classes, backbone=args.backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)

    yolo_loss = YOLOLoss(
        anchors,
        num_classes,
        args.input_size,
        anchors_mask,
        label_smoothing,
        args.focal_loss,
        focal_alpha,
        focal_gamma,
    )
    loss_history = LossHistory(figure_dir, model, input_size=args.input_size)
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
        "adam": optim.Adam(pg0, args.lr, betas=(momentum, 0.999)),
        "adamw": optim.AdamW(pg0, args.lr, betas=(momentum, 0.999)),
        "sgd": optim.SGD(pg0, args.lr, momentum=momentum, nesterov=True),
    }[args.opt]
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})

    if args.resume or args.auto_resume:
        auto_load_model(
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=None,
            model_ema=None,
        )
    start_epoch = getattr(args, "start_epoch", 0)

    show_config(
        classes_path="COCO API",
        resume=args.resume,
        input_size=args.input_size,
        start_epoch=start_epoch,
        Freeze_Epoch=freeze_epoch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        Freeze_Train=freeze_train,
        lr=args.lr,
        opt=args.opt,
        momentum=momentum,
        lr_scheduler=lr_scheduler,
        save_ckpt_freq=args.save_ckpt_freq,
        save_dir=save_dir,
        num_workers=args.num_workers,
        num_train=num_train,
        num_val=num_val,
    )

    lr_scheduler_func = get_lr_scheduler(lr_scheduler, args.lr, args.epochs)
    epoch_step = num_train // args.batch_size
    epoch_step_val = (num_val + args.batch_size - 1) // args.batch_size

    print("创建COCO YOLO数据集...")
    train_dataset = CocoYoloDataset(
        root=str(train_image_path),
        annFile=str(train_annotation_path),
        input_size=args.input_size,
        epoch_length=args.epochs,
        mosaic=args.mosaic,
        mixup=args.mixup,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
        train=True,
        special_aug_ratio=special_aug_ratio,
    )
    val_dataset = CocoYoloDataset(
        root=str(val_image_path),
        annFile=str(val_annotation_path),
        input_size=args.input_size,
        epoch_length=args.epochs,
        mosaic=False,
        mixup=False,
        mosaic_prob=0,
        mixup_prob=0,
        train=False,
        special_aug_ratio=0,
    )
    collate_fn = collate_yolo_batch

    gen = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    gen_val = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    val_image_ids = val_dataset.image_ids if hasattr(val_dataset, 'image_ids') else []
    eval_callback = EvalCallback(
        model,
        args.input_size,
        anchors,
        anchors_mask,
        class_names,
        num_classes,
        val_image_ids,
        figure_dir,
        device,
        val_dataset=val_dataset,
        eval_flag=eval_flag,
        period=args.eval_freq,
        num_epochs=args.epochs,
    )

    temp_map = 0
    last_map = temp_map
    for epoch in range(start_epoch, args.epochs):
        if epoch >= freeze_epoch and not unfreeze_flag and freeze_train:
            nbs = 64
            lr_limit_max = 1e-3 if args.opt in ["adam", "adamw"] else 5e-2
            lr_limit_min = 3e-4 if args.opt in ["adam", "adamw"] else 5e-4
            init_lr_fit = min(max(args.batch_size / nbs * args.lr, lr_limit_min), lr_limit_max)
            min_lr_fit = min(
                max(args.batch_size / nbs * args.lr, lr_limit_min * 1e-2),
                lr_limit_max * 1e-2,
            )
            lr_scheduler_func = get_lr_scheduler(lr_scheduler, init_lr_fit, min_lr_fit, args.epochs)
            for param in model.backbone.parameters():
                param.requires_grad = True
            epoch_step = num_train // args.batch_size
            epoch_step_val = num_val // args.batch_size
            unfreeze_flag = True

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
            args.epochs,
            device,
            args.save_ckpt_freq,
            save_dir,
            temp_map,
        )
        if temp_map is None:
            temp_map = last_map
        else:
            last_map = temp_map


def main():
    parser = argparse.ArgumentParser("检测训练脚本", parents=[get_args_parser()])
    args = parser.parse_args()
    train_detection(args)


if __name__ == "__main__":
    main()
