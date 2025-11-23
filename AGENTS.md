# Repository Guidelines

## 项目结构与模块
- 入口：`train_cls.py`（分类，timm 模型、EMA、mixup/cutmix）、`train_det.py`（YOLO 风格 COCO 检测）；`modelchange.py` 负责 ONNX/TRT/EMA 权重互转。
- `datasets.py`（ImageFolder 划分与增强）、`engine.py`（训练/评估循环）、`optim_factory.py`（参数组与优化器），`utils/`（日志、分布式、dataloader、回调）、`utils_coco/`（COCO 评估）。
- 模型位于 `nets/`（ConvNeXt V2、TimesNet、YOLO 等）；资源在 `model_data/`（`simhei.ttf`、`classes.txt`）。分类输出位于 `train_cls/output` 与 `train_cls/log_dir`，检测产物在 `train_det/`（含 checkpoint、figure）。

## 构建、测试与开发命令
- 环境：Python 3.12+，按 `pyproject.toml` 安装，推荐 `uv sync` 或 `pip install -e .`（必要时用阿里云镜像与 pytorch-cuda 索引）。
- 单卡分类：`python train_cls.py --data_path /path/to/data --epochs 30 --model convnextv2_tiny.fcmae_ft_in22k_in1k`。
- 多卡分类：`torchrun --nproc_per_node=8 train_cls.py --data_path /path/to/data --model_ema --batch_size 32`。
- 仅评估：`python train_cls.py --mode eval --resume /path/to/checkpoint --data_path /path/to/val`。
- 检测：`python -c "from train_det import train_detection; train_detection(data_path='/data/COCO2017', input_size=640, batch_size=16)"`。
- 集群：`python run_with_submitit.py ... --job_dir /path/to/job --dist_url tcp://...`，提前确认分区、节点数。

## 代码风格与命名
- 遵循 PEP8，4 空格缩进；函数/变量用 snake_case，类用 PascalCase，参数尽量显式，避免隐藏状态。
- 复用日志、分布式、优化器工具；新增公共函数补充简短 docstring 与类型标注；日志建议中英文简短可搜索。
- argparse 默认值贴合常用数据集，保持可复现；必要时设置合理随机种子。

## 测试与验证
- 暂无单测，先做冒烟：小子集或 `--mode eval` 试跑；关注 `train_cls/log_dir`（TensorBoard）与 `train_cls/output/class_indices.json`。
- 检测训练前确保 COCO 标注可通过校验；关注每轮 mAP 与 `train_det/figure` 图表，体积较大图表或中间权重提交前请清理。
- 调整 anchors、数据增强或输入尺寸时记录命令、数据切分方式与 seed，便于复现。

## Commit 与 PR 指南
- Commit 标题用简短祈使句（中英皆可），相关改动聚合提交。
- PR 描述需包含目的、关键改动、运行命令、数据集、前后指标或日志片段；若关联 Issue/任务请引用编号，截图仅在有帮助时附上。
- 避免提交大体积权重或原始数据，外部存储路径写明；保留最小必要的 class_indices、日志与可视化文件。

## 安全与配置提示
- 训练路径避免空格或非 ASCII 字符，长路径使用绝对路径；检查 `PYTHONHASHSEED`、`OMP_NUM_THREADS` 等是否符合集群限制。
- 不要在仓库或日志中写入 API 密钥/证书，W&B 等登录信息放在环境变量并遮蔽输出。
- GPU 显存不足时下调 `input_size`、`batch_size` 或关闭混合增强；多卡需确保 `--dist_url`、`--world_size` 与拓扑一致；长跑任务定期轮转或压缩日志与权重，确保磁盘充足。
