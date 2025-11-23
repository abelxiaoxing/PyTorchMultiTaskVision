import torch
from tqdm import tqdm

from utils.scheduler import get_lr
from utils.checkpoint import save_model

def fit_one_epoch(
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
):
    loss = 0  # 总损失
    val_loss = 0  # 验证集损失
    input_shape = [1,] + list(next(iter(gen))[0].shape[1:])
    print("开始训练")
    pbar = tqdm(
        total=epoch_step,
        desc=f"Epoch {epoch}/{epochs}",  # 显示进度条描述
        postfix=dict,
        mininterval=0.3,
    )
    model_train.train()  # 设置模型为训练模式
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]  # 读取图像和目标
        with torch.no_grad():
            images = images.to(device)  # 将图像移动到指定设备上
            targets = [ann.to(device) for ann in targets]  # 将目标移动到指定设备上
        optimizer.zero_grad()  # 梯度清零
        outputs = model_train(images)  # 前向传播
        outputs = outputs[0:3]  # 只保留部分输出
        loss_value_all = 0
        for l in range(len(outputs)):
            loss_item = yolo_loss(l, outputs[l], targets)  # 计算损失
            loss_value_all += loss_item
        loss_value = loss_value_all  # 总损失为所有层损失之和
        loss_value.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        loss += loss_value.item()  # 累加损失

        pbar.set_postfix(
            **{"loss": loss / (iteration + 1), "lr": get_lr(optimizer)}
        )  # 更新进度条显示
        pbar.update(1)
    train_loss_avg = loss / (iteration + 1)  # 计算平均训练损失
    pbar.close()
    print("完成训练")
    print("开始验证")
    pbar = tqdm(
        total=epoch_step_val,
        desc=f"Epoch {epoch}/{epochs}",
        postfix=dict,
        mininterval=0.3,
    )

    model_train.eval()  # 设置模型为评估模式
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            images = images.to(device)
            targets = [ann.to(device) for ann in targets]
            optimizer.zero_grad()
            outputs = model_train(images)
            outputs = outputs[0:3]
            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

        val_loss += loss_value.item()  # 累加验证集损失
        pbar.set_postfix(**{"val_loss": val_loss / (iteration + 1)})  # 更新进度条显示
        pbar.update(1)
    val_loss_avg = val_loss / (iteration + 1)  # 计算平均验证集损失

    pbar.close()
    print("完成验证")
    loss_history.append_loss(
        epoch, loss / epoch_step, val_loss / epoch_step_val
    )  # 记录损失历史
    temp_map = eval_callback.on_epoch_end(epoch, model_train)  # 执行回调函数
    print("Epoch:" + str(epoch) + "/" + str(epochs))
    print(
        "总损失: %.3f || 验证损失: %.3f "
        % (loss / epoch_step, val_loss / epoch_step_val)
    )

    # 保存模型权重
    if (epoch) % save_ckpt_freq == 0 or epoch == epochs:
        save_model(
            output_dir=save_dir,
            input_shape=input_shape,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
        )

    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(
        loss_history.val_loss
    ):
        print("保存最佳模型至 checkpoint-best.pth")
        save_model(
            output_dir=save_dir,
            input_shape=input_shape,
            model=model,
            optimizer=optimizer,
            epoch="best",
        )

    return train_loss_avg, val_loss_avg, temp_map
