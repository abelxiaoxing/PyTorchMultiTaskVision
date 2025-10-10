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
from utils.utils import get_anchors, get_classes, show_config
from utils.utils_fit import fit_one_epoch
from pathlib import Path
from utils_coco.coco_annotation import coco_annotation
import utils.utils as utils


def train_detection(
    mid=11,
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
    coco_annotation(data_path=data_path, mid=mid)

    lr_scheduler = "cosine"
    pretrained = True
    focal_alpha = 0.25
    focal_gamma = 2
    classes_path = Path("train_det/" + str(mid) + "/classes.txt")
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
    save_dir = Path("train_det/" + str(mid) + "/output")
    os.makedirs(save_dir, exist_ok=True)
    figure_dir = Path("train_det/" + str(mid) + "/figure")
    eval_flag = True
    num_workers = 1
    train_annotation_path = Path("train_det/" + str(mid) + "/train.txt")
    val_annotation_path = Path("train_det/" + str(mid) + "/val.txt")

    class_names, num_classes = get_classes(classes_path)
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
    with open(train_annotation_path, encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding="utf-8") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        classes_path=classes_path,
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

    model, start_epoch = utils.auto_load_model(
        resume=resume,
        model=model,
        optimizer=optimizer,
        model_ema=None,
        device=device,
    )
    lr_scheduler_func = get_lr_scheduler(lr_scheduler, lr, epochs)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    train_dataset = YoloDataset(
        train_lines,
        input_size,
        num_classes,
        epoch_length=epochs,
        mosaic=mosaic,
        mixup=mixup,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
        train=True,
        special_aug_ratio=special_aug_ratio,
    )
    val_dataset = YoloDataset(
        val_lines,
        input_size,
        num_classes,
        epoch_length=epochs,
        mosaic=False,
        mixup=False,
        mosaic_prob=0,
        mixup_prob=0,
        train=False,
        special_aug_ratio=0,
    )

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
        collate_fn=yolo_dataset_collate,
        sampler=train_sampler,
    )
    gen_val = DataLoader(
        val_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=yolo_dataset_collate,
        sampler=val_sampler,
    )

    eval_callback = EvalCallback(
        model,
        input_size,
        anchors,
        anchors_mask,
        class_names,
        num_classes,
        val_lines,
        figure_dir,
        device,
        eval_flag=eval_flag,
        period=eval_freq,
        num_epochs=epochs,
    )
    config = {
        "user": "root",
        "password": "123456",
        "host": "192.168.100.123",
        "database": "datacenter",
        "raise_on_warnings": True,
        "port": 3306,
    }
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
    train_detection(data_path='C:/datas/COCO2017')
