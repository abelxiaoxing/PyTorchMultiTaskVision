import math
import time
from typing import Iterable, Optional

import torch
from timm.data.mixup import Mixup
from timm.utils.model_ema import ModelEmaV3
from timm.utils.metrics import accuracy
from rich.progress import Progress

from utils.metrics import MetricLogger, SmoothedValue, calculate_precision_recall

def update_metrics(preds, targets, true_positives, false_positives, false_negatives):
    num_classes = len(true_positives)
    for i in range(num_classes):
        true_positives[i] += torch.sum((preds == i) & (targets == i)).item()
        false_positives[i] += torch.sum((preds == i) & (targets != i)).item()
        false_negatives[i] += torch.sum((preds != i) & (targets == i)).item()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEmaV3] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False,num_classes=2):

    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    optimizer.zero_grad()
    start_time = time.time()
    true_positives = [0] * num_classes
    false_positives = [0] * num_classes
    false_negatives = [0] * num_classes
    with Progress() as progress:
        task = progress.add_task(f"[green]Epoch {epoch} ", total=len(data_loader))

        for data_iter_step, (samples, targets) in enumerate(data_loader):
            progress.update(task, advance=1)
            step = data_iter_step // update_freq
            if step >= num_training_steps_per_epoch:
                continue
            it = start_steps + step
            if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            original_samples = samples
            original_targets = targets

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    output = model(samples)
                    loss = criterion(output, targets)
            else:
                output = model(samples)
                loss = criterion(output, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                optimizer.zero_grad()
                continue

            if use_amp: 
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss /= update_freq
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,parameters=model.parameters(), create_graph=is_second_order,update_grad=(data_iter_step + 1) % update_freq == 0)
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
                        
            else: 
                loss /= update_freq
                loss.backward()
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)

            torch.cuda.synchronize()

            if mixup_fn is not None:
                with torch.no_grad():
                    eval_output = model(original_samples)
                eval_targets = original_targets
            else:
                eval_output = output
                eval_targets = targets

            _, preds = torch.max(eval_output, 1)
            update_metrics(preds, eval_targets, true_positives, false_positives, false_negatives)
            
            class_acc = (eval_output.max(-1)[-1] == eval_targets).float().mean()
                    
            metric_logger.update(loss=loss_value)
            metric_logger.update(class_acc=class_acc)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(class_acc=class_acc, head="loss")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                if use_amp:
                    log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.set_step()

            if wandb_logger:
                wandb_logger._wandb.log({
                    'Rank-0 Batch Wise/train_loss': loss_value,
                    'Rank-0 Batch Wise/train_max_lr': max_lr,
                    'Rank-0 Batch Wise/train_min_lr': min_lr
                }, commit=False)
                if class_acc:
                    wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
                if use_amp:
                    wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
                wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
    end_time = time.time() 
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats:{metric_logger},Time:{end_time - start_time}")

    # 计算并打印每个类别的精确率和召回率
    calculate_precision_recall(true_positives, false_positives, false_negatives, num_classes)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, num_classes, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    # 初始化用于存储每类的真正例、假正例和假反例的计数器
    true_positives = [0] * num_classes
    false_positives = [0] * num_classes
    false_negatives = [0] * num_classes

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Val:'

    # 添加平均精确率和召回率的 Meter 对象
    metric_logger.add_meter('avg_precision', SmoothedValue(window_size=1, fmt='{value:.5f}'))
    metric_logger.add_meter('avg_recall', SmoothedValue(window_size=1, fmt='{value:.5f}'))

    # 切换到评估模式
    model.eval()

    for batch in metric_logger.log_every(data_loader, 0, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        output = model(images)
        loss = criterion(output, target)

        # 转换输出为预测类别
        _, preds = torch.max(output, 1)

        # 更新每类的真正例、假正例和假反例计数
        update_metrics(preds, target, true_positives, false_positives, false_negatives)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    # 计算并打印每类的精确率和召回率
    precision_recall_results = calculate_precision_recall(true_positives, false_positives, false_negatives, num_classes)
    
    total_precision = sum([pr[0] for pr in precision_recall_results])
    total_recall = sum([pr[1] for pr in precision_recall_results])
    
    avg_precision = total_precision / num_classes if num_classes > 0 else 0
    avg_recall = total_recall / num_classes if num_classes > 0 else 0
    
    # 更新平均精确率和召回率的 Meter 对象
    metric_logger.meters['avg_precision'].update(avg_precision)
    metric_logger.meters['avg_recall'].update(avg_recall)

    print(f'Average Precision: {avg_precision:.5f}, Average Recall: {avg_recall:.5f}')

    # 返回所有度量的全局平均值
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
