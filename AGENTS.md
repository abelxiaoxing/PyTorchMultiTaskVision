# Repository Guidelines
Always respond in Chinese-simplified
## 项目结构与模块
- 入口：`train_cls.py`（分类，timm 模型、EMA、mixup/cutmix），`train_det.py`（YOLO 风格 COCO 检测）；`modelchange.py` 负责 ONNX/TRT/EMA 权重互转。
- 核心工具：`datasets.py`（ImageFolder 划分与增强）、`engine.py`（训练/评估循环）、`optim_factory.py`（参数组与优化器）、`utils/`（日志、分布式、dataloader、回调）、`utils_coco/`（COCO 评估）。
- 模型位于 `nets/`（ConvNeXt V2、TimesNet、YOLO 等）；资源与类别文件在 `model_data/`。分类输出存放 `train_cls/output` 与 `train_cls/log_dir`，检测训练输出在 `train_det/`（包含 checkpoint、图表）。
- 数据期望：分类遵循 ImageFolder，可自动按 `train_split_ratio` 分层切分或手动提供 train/val 目录；检测采用 COCO2017 标准目录与 annotation json。
- 其他资产：`model_data/simhei.ttf` 用于可视化中文字；`model_data/classes.txt` 供检测读取类别，更新时保持与数据一致。

## 构建、测试与开发命令
- 环境：Python 3.12+，按 `pyproject.toml` 使用 CUDA 版 PyTorch；推荐 `uv sync`，或 `pip install -e .`，必要时使用阿里云镜像与 pytorch-cuda 索引。
- 分类单卡：`python train_cls.py --data_path /path/to/data --epochs 30 --model convnextv2_tiny.fcmae_ft_in22k_in1k`。
- 分类多卡：`torchrun --nproc_per_node=8 train_cls.py --data_path /path/to/data --model_ema --batch_size 32`。
- 仅评估：`python train_cls.py --mode eval --resume /path/to/checkpoint --data_path /path/to/val`。
- 检测（COCO 目录）：更新文件底部示例或直接调用 `train_detection`，示例 `python -c "from train_det import train_detection; train_detection(data_path='/data/COCO2017', input_size=640, batch_size=16)"`。
- 集群提交：使用 `run_with_submitit.py` 包装分类训练，提前确认分区、节点数与 `--job_dir`，产物会写入 job 目录；必要时自定义 `--dist_url`。

## 代码风格与命名
- 遵循 PEP8：4 空格缩进，函数/变量用 snake_case，类用 PascalCase。优先显式参数，避免隐藏状态。
- 复用已有日志、分布式、优化器工具，新增公共函数补充简短 docstring。
- 保障可复现：合理默认种子，必要时启用确定性加载，argparse 默认值贴合常用数据集；推荐补充类型标注与断言以减少运行时错误。日志输出建议中英文一致、简短可搜索。

## 测试指引
- 暂无线下单测，使用冒烟验证。分类先在小子集 `--mode eval` 试跑，关注 `train_cls/log_dir`（TensorBoard）与 `train_cls/output/class_indices.json`。
- 检测训练前确保 COCO 标注可通过校验（脚本内已校验），关注每轮打印的 mAP；`train_det/figure` 图表或中间权重若体积较大，请清理后再提交。
- 汇报结果或回归时，请附运行命令、数据切分方式与 seed；若使用 EMA、AMP、混合增强（mixup/cutmix/ mosaic）请注明，以便复现。
- 调整 anchors、数据增强或输入尺寸时记录变更点，方便重新对齐指标。

## 安全与配置提示
- 训练路径中避免包含空格或非 ASCII 字符，以免 dataloader 解析失败；长路径建议使用绝对路径。
- 不要将 API 密钥、证书或私有数据写入仓库或日志，W&B 登录信息请置于环境变量，并在日志中屏蔽。
- GPU 显存不足时可下调 `input_size`、`batch_size` 或关闭混合增强；多卡训练需确保 `--dist_url`、`--world_size` 与实际拓扑一致。
- 运行前检查 `PYTHONHASHSEED`、`OMP_NUM_THREADS` 等环境变量是否符合集群限制；提交作业前清理旧的 checkpoints/figure 目录，避免冗余占用。
- 保证输出目录磁盘空间充足，长跑任务建议定期轮转或压缩日志与权重。

## Commit 与 PR 规范
- Commit 标题用简短祈使句（中英皆可），如 “修复 EMA 转常规模型导出”；相关改动聚合提交。
- PR 应包含：目的、关键改动、运行命令、数据集、前后指标或日志片段；曲线截图仅在有帮助时附上，注明数据集版本与超参。
- 避免提交大体积权重或原始数据，请存储在外部并在文档中注明路径；对生成的 class_indices、日志、可视化文件保持最小必要集。
- 如 PR 关联 Issue/任务，请在描述中引用编号并列出评审关注点，便于复核。
