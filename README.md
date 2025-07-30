## 更新说明
### 20250730
+ 大幅简化了engine和datasets代码结构，移除了重复代码和冗余循环。
+ 删除mmdet依赖
+ 删除冗余数据增强
+ 采用分层采样策略，保证数据划分后的类别分布不变。
+ 优化了划分性能，避免加载完整图像，速度大幅提升。
+ modelchange改为面向对象，代码更加清晰，易于维护。
+ optim_factory区分权重和偏置/归一化参数，不对偏置/归一化参数应用衰减。重构get_parameter_groups部分，用列表推导式构建优化器所需的参数组，提高性能。

### 20241217
+ modelchange.py新增model_ema转化为标准模型，权重中"model_ema"改为"model"
+ 修复当model_ema为True时,恢复的权重文件中无model_ema时的权重加载问题
+ 启用mixup时,也可以查看train数据集的准确率
+ 分类任务时模型权重不匹配模型文件时,打印不匹配的层,自动剔除不匹配层(迁移学习等无需手动剔除最后一层)

### 20241127
+ 新增集群训练run_with_submitit.py
+ 新增模型转换model_change.py，实现onnx,pth,trt模型相互转换
+ train.py移除分类任务中num_classes参数，自动读取文件夹内的类别数量
+ 为便于分享减少手动设置参数，修改模型保存方式，只保存权重改为保存整个模型
+ train.py更新model_ema方法，加快了速度
+ val.py新增参数model_ema，决定采用原始模型推理还是采用ema权重推理,如果权重中没有保存model_ema即使推理采用model_ema也将使用原始权重推理
+ 提高optim_factory的自动化程度，非关键参数改为默认


## 基于pytorch的深度学习图像分类框架使用简介
1. 下载好数据集，代码中默认使用的是迷你猫狗分类数据集，下载地址:  [迷你猫狗分类数据集](https://pan.baidu.com/s/16SPmrN_PUUTWQuxtRXZmrA?pwd=abel) 提取码: abel  
### 训练
+ 多显卡训练,此案例为8卡`torchrun --nproc_per_node=8 train.py`
+ 单显卡训练`python train.py`
### train.py
1. 在`train.py`脚本中将`--data-path`设置成解压后的`flower_photos`文件夹绝对路径
2. 去github抄所需要网络的代码，并将其放在`models`文件夹下，然后将`train.py`下导入的网络设置成你要使用的网络
3. 在`train.py`脚本中将`--weights`参数设成下载好的预训练权重路径
4. 设置好数据集的路径`--data-path`以及预训练权重的路径`--weights`就能使用`train.py`脚本开始训练了(训练过程中会自动生成`class_indices.json`文件)
5. 若需要自己划分好数据集并在所填的data_path路径后划分为train和val,则设置data_custom为True。
### val.py
1. 在`val.py`脚本中导入和训练脚本中同样的模型，并将`model_weight_path`设置成训练好的模型权重路径(默认保存在weights文件夹下)
2. 在`val.py`脚本中将`img_path`设置成你自己需要预测的图片文件夹绝对路径
3. 设置好权重路径`model_weight_path`和预测的图片路径`img_path`就能使用`val.py`脚本进行预测了
4. 选择val_move和val_precision执行，val_move为预测该文件夹的图片并进行分类，val_precision为预测包含类别的文件夹并给出精确率
### Others
+ 如果要使用自己的数据集，请按照花分类数据集的文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的`num_classes`设置成你自己数据的类别数
### todo
1. 适配目标检测
2. 适配语义分割
3. 增加权重转换
4. 增加模型可视化
5. 增加剪枝、量化、蒸馏等操作