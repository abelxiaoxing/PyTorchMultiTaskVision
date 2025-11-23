import glob
import os
from pathlib import Path
from timm.utils import get_state_dict
import torch

from .runtime import save_on_master


def save_model(
    output_dir,
    input_shape=None,
    epoch=None,
    model=None,
    optimizer=None,
    loss_scaler=None,
    model_ema=None,
    num_classes=None,
    save_ckpt_num=None,
    save_ckpt_freq=None,
):
    """
    保存模型权重，兼容分类与检测。要求显式传入 output_dir，避免默认路径歧义。
    """
    epoch_name = str(epoch)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / ("checkpoint-%s.pth" % epoch_name)

    to_save = {
        "model": model,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "input_shape": input_shape,
    }
    if loss_scaler is not None:
        to_save["scaler"] = loss_scaler.state_dict()
    if num_classes is not None:
        to_save["num_classes"] = num_classes
    if model_ema is not None:
        to_save["model_ema"] = get_state_dict(model_ema)
    save_on_master(to_save, checkpoint_path)

    if isinstance(epoch, int) and save_ckpt_num and save_ckpt_freq:
        to_del = epoch - save_ckpt_num * save_ckpt_freq
        old_ckpt = output_dir / ("checkpoint-%s.pth" % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def auto_load_model(args, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(getattr(args, "output_dir", "./train_cls/output"))
    if getattr(args, "auto_resume", False) and len(getattr(args, "resume", "")) == 0:
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if getattr(args, "resume", ""):
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
            state_dict = checkpoint["model"]
        else:
            print(args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            raw_model = checkpoint["model"]
            if isinstance(raw_model, torch.nn.Module):
                state_dict = raw_model.state_dict()
            elif isinstance(raw_model, dict):
                state_dict = raw_model
            else:
                raise TypeError(f"Unsupported checkpoint['model'] type: {type(raw_model)}")

        model_state_dict = model_without_ddp.state_dict()
        new_state_dict = {}
        missing_nums = 0
        for k, v in state_dict.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                new_state_dict[k] = v
            else:
                print(f"Skipping mismatched key: {k}")
                missing_nums += 1

        model_without_ddp.load_state_dict(new_state_dict, strict=False)
        print("Resume checkpoint %s" % args.resume)

        if getattr(args, "model_ema", False):
            if 'model_ema' in checkpoint.keys() and missing_nums == 0:
                model_ema.module.load_state_dict(checkpoint['model_ema'])
            else:
                model_ema.set(model_without_ddp)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and missing_nums == 0:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not isinstance(checkpoint['epoch'], str):
                args.start_epoch = checkpoint['epoch'] + 1
            else:
                assert getattr(args, "eval", False), 'Does not support resuming with checkpoint-best'

            if 'scaler' in checkpoint and loss_scaler is not None:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
    return


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))
