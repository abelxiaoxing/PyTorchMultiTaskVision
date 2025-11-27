from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, get_args, get_origin, get_type_hints

import tomllib

T = TypeVar("T")


class ConfigError(Exception):
    """当配置文件无法解析为预期结构时抛出。"""


@dataclass
class RuntimeConfig:
    """
    运行时配置类

    包含训练过程中的运行时参数，如设备设置、分布式训练配置等

    Attributes:
        device: 训练设备，默认为 "cuda"
        seed: 随机种子，默认为 88
        dist_url: 分布式训练URL，默认为 "env://"
        dist_on_itp: 是否在ITP上运行分布式训练，默认为 False
        world_size: 分布式训练的世界大小，默认为 1
        local_rank: 本地进程排名，默认为 -1
        rank: 进程全局排名，默认为 0
        gpu: GPU编号，默认为 0
        distributed: 是否启用分布式训练，默认为 False
    """
    device: str = "cuda"
    seed: int = 88
    dist_url: str = "env://"
    dist_on_itp: bool = False
    world_size: int = 1
    local_rank: int = -1
    rank: int = 0
    gpu: int = 0
    distributed: bool = False


@dataclass
class CheckpointConfig:
    """
    检查点配置类

    管理模型检查点的保存、恢复和加载相关参数

    Attributes:
        resume: 恢复训练的检查点路径，默认为空字符串
        auto_resume: 是否自动恢复训练，默认为 False
        save_ckpt: 是否保存检查点，默认为 True
        save_ckpt_freq: 检查点保存频率（按epoch），默认为 1
        save_ckpt_num: 最大保存检查点数量，默认为 100
        start_epoch: 起始训练轮次，默认为 0
    """
    resume: str = ""
    auto_resume: bool = False
    save_ckpt: bool = True
    save_ckpt_freq: int = 1
    save_ckpt_num: int = 100
    start_epoch: int = 0


@dataclass
class ClassificationLoggingConfig:
    """
    分类任务日志配置类

    配置分类任务的日志记录、实验跟踪和输出目录

    Attributes:
        enable_wandb: 是否启用Weights & Biases实验跟踪，默认为 False
        project: WandB项目名称，默认为 "classification"
        wandb_ckpt: 是否将检查点保存到WandB，默认为 False
        log_dir: 日志文件目录，默认为 "train_cls/log_dir"
        output_dir: 模型输出目录，默认为 "train_cls/output"
    """
    enable_wandb: bool = False
    project: str = "classification"
    wandb_ckpt: bool = False
    log_dir: str = "train_cls/log_dir"
    output_dir: str = "train_cls/output"


@dataclass
class DetectionLoggingConfig:
    """
    目标检测任务日志配置类

    配置目标检测任务的输出目录和可视化结果目录

    Attributes:
        output_dir: 模型输出目录，默认为 "train_det/output"
        figure_dir: 可视化图表目录，默认为 "train_det/figure"
    """
    output_dir: str = "train_det/output"
    figure_dir: str = "train_det/figure"


@dataclass
class ClassificationTrainingConfig:
    """
    分类任务训练配置类

    配置图像分类任务的训练参数和优化策略

    Attributes:
        mode: 训练模式，默认为 "train"
        batch_size: 批次大小，默认为 16
        epochs: 训练轮次，默认为 30
        update_freq: 梯度累积频率，默认为 1
        model_ema: 是否使用指数移动平均模型，默认为 False
        use_amp: 是否使用自动混合精度训练，默认为 True
        clip_grad: 梯度裁剪阈值，None表示不裁剪，默认为 None
    """
    mode: str = "train"
    batch_size: int = 16
    epochs: int = 30
    update_freq: int = 1
    model_ema: bool = False
    use_amp: bool = True
    clip_grad: Optional[float] = None


@dataclass
class ClassificationModelConfig:
    """
    分类任务模型配置类

    配置图像分类任务的模型架构和参数

    Attributes:
        name: 模型名称，默认为 "convnextv2_tiny.fcmae_ft_in22k_in1k"
        pretrained: 是否使用预训练权重，默认为 True
        drop_path: DropPath比率，用于正则化，默认为 0.05
        input_size: 输入图像尺寸，默认为 224
    """
    name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    pretrained: bool = True
    drop_path: float = 0.05
    input_size: int = 224


@dataclass
class OptimizerConfig:
    """
    优化器配置类

    配置模型训练的优化器参数和学习率调度策略

    Attributes:
        opt: 优化器类型，默认为 "lion"
        lr: 初始学习率，默认为 1e-4
        min_lr: 最小学习率，默认为 1e-7
        weight_decay: 权重衰减系数，默认为 5e-4
        weight_decay_end: 最终权重衰减系数，默认为 5e-6
        warmup_epochs: 学习率预热轮次，默认为 1
    """
    opt: str = "lion"
    lr: float = 1e-4
    min_lr: float = 1e-7
    weight_decay: float = 5e-4
    weight_decay_end: float = 5e-6
    warmup_epochs: int = 1


@dataclass
class ClassificationAugmentationConfig:
    """
    分类任务数据增强配置类

    配置图像分类任务的数据增强策略和参数

    Attributes:
        ra_sampler: 是否使用随机增强采样器，默认为 False
        color_jitter: 颜色抖动强度，默认为 0.1
        aa: 自动增强策略，默认为空字符串
        reprob: 随机擦除概率，默认为 0.0
        mixup: Mixup混合比例，默认为 0.0
        cutmix: CutMix混合比例，默认为 0.0
    """
    ra_sampler: bool = False
    color_jitter: float = 0.1
    aa: str = ""
    reprob: float = 0.0
    mixup: float = 0.0
    cutmix: float = 0.0


@dataclass
class ClassificationDataConfig:
    """
    分类任务数据配置类

    配置图像分类任务的数据集路径、数据加载参数等

    Attributes:
        data_path: 数据集根目录路径，默认为 "../../datas/flower_photos"
        train_split_ratio: 训练集划分比例，默认为 0.8
        num_workers: 数据加载器工作进程数，默认为 8
        move_dir: 数据移动目录（用于数据预处理），默认为空字符串
    """
    data_path: str = "../../datas/flower_photos"
    train_split_ratio: float = 0.8
    num_workers: int = 8
    move_dir: str = ""


@dataclass
class DetectionTrainingConfig:
    """
    目标检测任务训练配置类

    配置目标检测任务的训练参数和优化策略

    Attributes:
        epochs: 训练轮次，默认为 100
        batch_size: 批次大小，默认为 64
        eval_freq: 评估频率（按epoch），默认为 2
        lr_scheduler: 学习率调度器类型，默认为 "cosine"
        freeze_epoch: 冻结训练轮次，默认为 50
        freeze_train: 是否启用冻结训练，默认为 True
        focal_loss: 是否使用Focal Loss，默认为 False
    """
    epochs: int = 100
    batch_size: int = 64
    eval_freq: int = 2
    lr_scheduler: str = "cosine"
    freeze_epoch: int = 50
    freeze_train: bool = True
    focal_loss: bool = False


@dataclass
class DetectionOptimConfig:
    """
    目标检测优化器配置类

    配置目标检测任务的优化器参数

    Attributes:
        opt: 优化器类型，默认为 "adamw"
        lr: 学习率，默认为 1e-3
        momentum: 动量系数，默认为 0.937
        weight_decay: 权重衰减系数，默认为 5e-4
    """
    opt: str = "adamw"
    lr: float = 1e-3
    momentum: float = 0.937
    weight_decay: float = 5e-4


@dataclass
class DetectionModelConfig:
    """
    目标检测模型配置类

    配置目标检测任务的模型架构、锚框参数和损失函数参数

    Attributes:
        backbone: 骨干网络类型，默认为 "efficientvit_b0"
        pretrained: 是否使用预训练权重，默认为 True
        input_size: 输入图像尺寸，默认为 640
        anchors: 锚框尺寸列表，包含18个预设锚框尺寸
        anchor_mask: 锚框掩码，用于不同特征层级的锚框分配
        focal_alpha: Focal Loss的alpha参数，默认为 0.25
        focal_gamma: Focal Loss的gamma参数，默认为 2.0
    """
    backbone: str = "efficientvit_b0"
    pretrained: bool = True
    input_size: int = 640
    anchors: tuple[int, ...] = (
        12,
        16,
        19,
        36,
        40,
        28,
        36,
        75,
        76,
        55,
        72,
        146,
        142,
        110,
        192,
        243,
        459,
        401,
    )
    anchor_mask: tuple[tuple[int, int, int], ...] = ((6, 7, 8), (3, 4, 5), (0, 1, 2))
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class DetectionAugmentationConfig:
    """
    目标检测数据增强配置类

    配置目标检测任务的数据增强策略和参数

    Attributes:
        mosaic: 是否启用Mosaic数据增强，默认为 True
        mixup: 是否启用Mixup数据增强，默认为 True
        mosaic_prob: Mosaic增强概率，默认为 0.5
        mixup_prob: Mixup增强概率，默认为 0.5
        special_aug_ratio: 特殊增强比例，默认为 0.7
        label_smoothing: 标签平滑系数，默认为 0.01
    """
    mosaic: bool = True
    mixup: bool = True
    mosaic_prob: float = 0.5
    mixup_prob: float = 0.5
    special_aug_ratio: float = 0.7
    label_smoothing: float = 0.01


@dataclass
class DetectionDataConfig:
    """
    目标检测数据配置类

    配置目标检测任务的数据集路径、标注文件和加载参数

    Attributes:
        data_path: 数据集根目录路径，默认为 "/data/COCO2017"
        train_annotation: 训练集标注文件路径，默认为 "annotations/instances_val2017.json"
        val_annotation: 验证集标注文件路径，默认为 "annotations/instances_val2017.json"
        train_split: 训练集划分名称，默认为 "val2017"
        val_split: 验证集划分名称，默认为 "val2017"
        num_workers: 数据加载器工作进程数，默认为 4
    """
    data_path: str = "/data/COCO2017"
    train_annotation: str = "annotations/instances_val2017.json"
    val_annotation: str = "annotations/instances_val2017.json"
    train_split: str = "val2017"
    val_split: str = "val2017"
    num_workers: int = 4


@dataclass
class ClassificationConfig:
    """
    分类任务主配置类

    图像分类任务的完整配置容器，包含所有子配置模块

    Attributes:
        runtime: 运行时配置模块
        checkpoint: 检查点配置模块
        logging: 日志配置模块
        training: 训练参数配置模块
        model: 模型配置模块
        optim: 优化器配置模块
        augmentation: 数据增强配置模块
        data: 数据配置模块
    """
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: ClassificationLoggingConfig = field(default_factory=ClassificationLoggingConfig)
    training: ClassificationTrainingConfig = field(default_factory=ClassificationTrainingConfig)
    model: ClassificationModelConfig = field(default_factory=ClassificationModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    augmentation: ClassificationAugmentationConfig = field(default_factory=ClassificationAugmentationConfig)
    data: ClassificationDataConfig = field(default_factory=ClassificationDataConfig)


@dataclass
class DetectionConfig:
    """
    目标检测任务主配置类

    目标检测任务的完整配置容器，包含所有子配置模块

    Attributes:
        runtime: 运行时配置模块
        checkpoint: 检查点配置模块
        logging: 日志配置模块
        training: 训练参数配置模块
        model: 模型配置模块
        optim: 优化器配置模块
        augmentation: 数据增强配置模块
        data: 数据配置模块
    """
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: DetectionLoggingConfig = field(default_factory=DetectionLoggingConfig)
    training: DetectionTrainingConfig = field(default_factory=DetectionTrainingConfig)
    model: DetectionModelConfig = field(default_factory=DetectionModelConfig)
    optim: DetectionOptimConfig = field(default_factory=DetectionOptimConfig)
    augmentation: DetectionAugmentationConfig = field(default_factory=DetectionAugmentationConfig)
    data: DetectionDataConfig = field(default_factory=DetectionDataConfig)


def _merge_dicts(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并两个字典

    深度合并两个字典，支持嵌套字典的递归合并。如果两个字典有相同的键且值都是字典，
    则递归合并这两个字典，否则用other字典的值覆盖base字典的值。

    Args:
        base: 基础字典，作为合并的基准
        other: 要合并到基础字典的字典

    Returns:
        合并后的新字典

    Examples:
        >>> base = {"a": 1, "b": {"x": 10}}
        >>> other = {"b": {"y": 20}, "c": 3}
        >>> _merge_dicts(base, other)
        {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}
    """
    merged = dict(base)
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(config_path: Path, visited: set[Path]) -> Dict[str, Any]:
    """
    加载原始配置文件并处理继承关系

    从TOML文件加载配置数据，支持配置文件的继承机制。通过"extends"字段可以指定
    父配置文件，实现配置的复用和覆盖。支持防止循环引用检测。

    Args:
        config_path: 配置文件路径
        visited: 已访问的配置文件路径集合，用于检测循环引用

    Returns:
        合并了所有父配置文件后的配置数据字典

    Raises:
        ConfigError: 当检测到循环引用或配置文件不存在时抛出

    Examples:
        # base.toml
        [training]
        batch_size = 16

        # classification.toml
        extends = ["base.toml"]
        [training]
        batch_size = 8  # 覆盖基础配置
    """
    if config_path in visited:
        raise ConfigError(f"Circular config reference detected at {config_path}")
    visited.add(config_path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    extends = data.pop("extends", [])
    if isinstance(extends, (str, Path)):
        extends = [extends]

    merged: Dict[str, Any] = {}
    for parent in extends:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = (config_path.parent / parent_path).resolve()
        merged = _merge_dicts(merged, _load_raw_config(parent_path, visited))
    return _merge_dicts(merged, data)


def _is_dataclass_type(type_hint: Any) -> bool:
    """
    检查类型提示是否为数据类类型

    判断给定的类型提示是否是一个数据类（dataclass）类型。

    Args:
        type_hint: 要检查的类型提示

    Returns:
        如果type_hint是一个数据类类型则返回True，否则返回False

    Examples:
        >>> _is_dataclass_type(RuntimeConfig)
        True
        >>> _is_dataclass_type(int)
        False
    """
    return isinstance(type_hint, type) and is_dataclass(type_hint)


def _build_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """
    从字典数据构建数据类实例

    递归地将字典数据转换为数据类实例，支持嵌套数据类的构建。
    自动处理字段的默认值、默认工厂函数和类型检查。

    Args:
        cls: 要构建的数据类类型
        data: 包含配置数据的字典

    Returns:
        构建完成的数据类实例

    Raises:
        ConfigError: 当缺少必需字段时抛出

    Examples:
        >>> data = {"device": "cuda", "seed": 88}
        >>> _build_dataclass(RuntimeConfig, data)
        RuntimeConfig(device='cuda', seed=88)
    """
    # 类型断言：确保cls是一个数据类类型
    assert is_dataclass(cls), f"{cls} must be a dataclass"
    
    kwargs: Dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for field_def in fields(cls):
        value = data.get(field_def.name, MISSING)
        field_type = type_hints.get(field_def.name, field_def.type)
        origin = get_origin(field_type)
        args = get_args(field_type)

        if value is MISSING:
            if field_def.default is not MISSING:
                continue
            if field_def.default_factory is not MISSING:  # type: ignore[attr-defined]
                kwargs[field_def.name] = field_def.default_factory()  # type: ignore[misc]
                continue
            raise ConfigError(f"Missing required config field: {field_def.name}")

        if _is_dataclass_type(field_type):
            kwargs[field_def.name] = _build_dataclass(field_type, value or {})
        elif origin is Optional and args and _is_dataclass_type(args[0]):
            kwargs[field_def.name] = _build_dataclass(args[0], value or {})
        else:
            kwargs[field_def.name] = value

    return cls(**kwargs)


def load_config(path: str | Path, cls: Type[T]) -> T:
    """
    从TOML文件加载配置数据类

    支持的核心特性：
    - 配置文件继承：通过"extends"字段实现配置复用
    - 类型安全：自动将配置数据转换为类型安全的数据类实例

    Args:
        path: 配置文件路径，可以是字符串或Path对象
        cls: 目标配置数据类类型

    Returns:
        构建完成的配置数据类实例

    Raises:
        ConfigError: 当配置文件不存在或解析失败时抛出

    Examples:
        # 加载分类任务配置
        config = load_config("configs/classification.toml", ClassificationConfig)
    """
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    raw = _load_raw_config(config_path.resolve(), set())
    return _build_dataclass(cls, raw)


def load_classification_config(path: str | Path | None = None) -> ClassificationConfig:
    """
    加载图像分类任务配置

    这是分类任务配置的便捷加载函数，封装了默认配置路径。

    Args:
        path: 自定义配置文件路径，如果不提供则使用默认路径

    Returns:
        分类任务配置实例

    Examples:
        # 使用默认配置路径
        config = load_classification_config()

        # 使用自定义配置路径
        config = load_classification_config("my_config.toml")
    """
    config_path = path or Path("configs/classification.toml")
    return load_config(config_path, ClassificationConfig)


def load_detection_config(path: str | Path | None = None) -> DetectionConfig:
    """
    加载目标检测任务配置

    这是目标检测任务配置的便捷加载函数，封装了默认配置路径。

    Args:
        path: 自定义配置文件路径，如果不提供则使用默认路径

    Returns:
        目标检测任务配置实例

    Examples:
        # 使用默认配置路径
        config = load_detection_config()

        # 使用自定义配置路径
        config = load_detection_config("my_detection_config.toml")
    """
    config_path = path or Path("configs/detection.toml")
    return load_config(config_path, DetectionConfig)
