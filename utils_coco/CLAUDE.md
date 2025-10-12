[根目录](../../CLAUDE.md) > [utils_coco](../) > **utils_coco**

# Utils COCO 模块

## 模块职责

utils_coco 模块是项目的 COCO 数据集专用工具，提供完整的 COCO 格式数据处理、验证、评估和转换功能。该模块专注于 COCO 数据集的标准化处理，确保数据格式正确性和评估指标计算的准确性。

## 入口与启动

### 模块功能概览
```python
# 主要功能模块
from .coco_annotation import coco_annotation, process_annotations
from .coco_evaluation import evaluate_coco, calculate_map
from .coco_utils import validate_coco_annotations, get_coco_statistics
from .get_map_coco import get_map
```

### 基本使用流程
```python
# 1. COCO 数据处理
coco_annotation(data_path="coco_dataset")

# 2. 数据验证
is_valid, msg = validate_coco_annotations("annotations.json")

# 3. 统计信息
stats = get_coco_statistics("annotations.json")

# 4. 评估计算
mAP = evaluate_coco(predictions, ground_truth)
```

## 对外接口

### 标注处理接口

#### COCO 标注转换
```python
def coco_annotation(data_path="coco_dataset"):
    """将 COCO 格式标注转换为 YOLO 训练格式"""
    train_output_path = Path("train_det/train.txt")
    val_output_path = Path("train_det/val.txt")
    classes_output_path = Path("train_det/classes.txt")

def process_annotations(image_annotations, data_path, output_path, images):
    """处理图像标注信息，转换为 YOLO 格式"""
    name_box_id = defaultdict(list)
    for image_id, annotations in image_annotations.items():
        image_entry = next(image for image in images if image['id'] == image_id)
        file_name = image_entry['file_name']
        name = os.path.join(data_path, file_name)
        for ant in annotations:
            cat = ant["category_id"] - 1  # COCO ID 从 1 开始，转为 0 开始
            name_box_id[name].append([ant["bbox"], cat])
```

#### 类别信息生成
```python
def generate_classes(categories, output_path):
    """生成类别文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for category in categories:
            f.write(f"{category['name']}\n")
```

### 数据验证接口

#### 标注文件验证
```python
def validate_coco_annotations(annotation_file):
    """验证 COCO 标注文件的完整性和正确性"""
    required_fields = ['images', 'annotations', 'categories']

    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 验证必需字段
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"

        # 验证数据结构
        if not validate_images(data['images']):
            return False, "Invalid images data structure"

        if not validate_annotations(data['annotations']):
            return False, "Invalid annotations data structure"

        if not validate_categories(data['categories']):
            return False, "Invalid categories data structure"

        return True, "Validation passed"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
```

#### 数据集统计
```python
def get_coco_statistics(annotation_file):
    """获取 COCO 数据集统计信息"""
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {
        'num_images': len(data['images']),
        'num_annotations': len(data['annotations']),
        'num_categories': len(data['categories']),
        'category_distribution': {},
        'image_sizes': [],
        'annotation_density': []
    }

    # 计算类别分布
    category_counts = defaultdict(int)
    for ann in data['annotations']:
        category_counts[ann['category_id']] += 1
    stats['category_distribution'] = dict(category_counts)

    return stats
```

#### 类别映射
```python
def get_coco_class_mapping(annotation_file):
    """获取 COCO 类别 ID 到名称的映射"""
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    class_mapping = {}
    for category in data['categories']:
        class_mapping[category['id']] = category['name']

    return class_mapping
```

### 评估计算接口

#### mAP 计算
```python
def evaluate_coco(predictions, ground_truth, iou_threshold=0.5):
    """计算 COCO 格式的 mAP 指标"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # 加载真实标注
    coco_gt = COCO(ground_truth)

    # 加载预测结果
    coco_dt = coco_gt.loadRes(predictions)

    # 初始化评估器
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    # 运行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        'mAP': coco_eval.stats[0],
        'mAP_50': coco_eval.stats[1],
        'mAP_75': coco_eval.stats[2],
        'mAP_s': coco_eval.stats[3],
        'mAP_m': coco_eval.stats[4],
        'mAP_l': coco_eval.stats[5]
    }
```

#### 精确率召回率计算
```python
def calculate_precision_recall(predictions, ground_truth, iou_threshold=0.5):
    """计算精确率和召回率"""
    tp, fp, fn = 0, 0, 0

    for pred_boxes, gt_boxes in zip(predictions, ground_truth):
        matched_gt = set()

        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue

                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall
```

### 工具函数接口

#### IoU 计算
```python
def calculate_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union
```

#### 坐标转换
```python
def coco_to_yolo(bbox, image_width, image_height):
    """将 COCO 格式边界框转换为 YOLO 格式"""
    x_min, y_min, width, height = bbox

    # 计算中心点坐标和相对尺寸
    x_center = x_min + width / 2
    y_center = y_min + height / 2

    # 归一化
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return x_center, y_center, width, height

def yolo_to_coco(bbox, image_width, image_height):
    """将 YOLO 格式边界框转换为 COCO 格式"""
    x_center, y_center, width, height = bbox

    # 反归一化
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    # 计算左上角坐标
    x_min = x_center - width / 2
    y_min = y_center - height / 2

    return [x_min, y_min, width, height]
```

## 关键依赖与配置

### 核心依赖
```python
import json
import os
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
```

### 可选依赖
```python
# COCO 评估工具
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 图像处理
from PIL import Image
import cv2
```

### 配置参数
```python
# 数据分割比例
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

# 验证配置
MIN_OBJECT_SIZE = 10  # 最小目标尺寸
MAX_OBJECTS_PER_IMAGE = 100  # 每张图像最大目标数

# IoU 阈值
DEFAULT_IOU_THRESHOLD = 0.5
EVAL_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
```

## 数据处理流程

### COCO 到 YOLO 转换
```python
def convert_coco_to_yolo(coco_annotation_file, output_dir):
    """完整的 COCO 到 YOLO 格式转换流程"""

    # 1. 加载 COCO 数据
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 3. 处理图像和标注
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)

    # 4. 分割训练集和验证集
    all_image_ids = list(image_annotations.keys())
    random.shuffle(all_image_ids)
    split_index = int(TRAIN_RATIO * len(all_image_ids))

    train_ids = all_image_ids[:split_index]
    val_ids = all_image_ids[split_index:]

    # 5. 转换并保存
    convert_and_save(image_annotations, train_ids, coco_data['images'],
                    f"{output_dir}/train.txt")
    convert_and_save(image_annotations, val_ids, coco_data['images'],
                    f"{output_dir}/val.txt")

    # 6. 保存类别文件
    save_classes_file(coco_data['categories'], f"{output_dir}/classes.txt")
```

### 数据质量检查
```python
def check_data_quality(coco_annotation_file, image_dir):
    """检查数据集质量"""
    issues = []

    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # 检查图像文件是否存在
    for image_info in coco_data['images']:
        image_path = os.path.join(image_dir, image_info['file_name'])
        if not os.path.exists(image_path):
            issues.append(f"Missing image: {image_path}")

    # 检查标注边界框是否有效
    for ann in coco_data['annotations']:
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0:
            issues.append(f"Invalid bbox in annotation {ann['id']}")

        if w < MIN_OBJECT_SIZE or h < MIN_OBJECT_SIZE:
            issues.append(f"Too small object in annotation {ann['id']}")

    # 检查类别一致性
    category_ids = set(ann['category_id'] for ann in coco_data['annotations'])
    defined_category_ids = set(cat['id'] for cat in coco_data['categories'])

    undefined_categories = category_ids - defined_category_ids
    if undefined_categories:
        issues.append(f"Undefined categories: {undefined_categories}")

    return issues
```

## 评估与指标

### COCO 评估指标
```python
def comprehensive_evaluation(predictions_file, ground_truth_file):
    """全面的 COCO 格式评估"""

    # 标准 mAP 评估
    results = evaluate_coco(predictions_file, ground_truth_file)

    # 按类别评估
    per_class_results = evaluate_per_class(predictions_file, ground_truth_file)

    # 按面积评估
    area_results = evaluate_by_area(predictions_file, ground_truth_file)

    return {
        'overall': results,
        'per_class': per_class_results,
        'by_area': area_results
    }

def evaluate_per_class(predictions, ground_truth):
    """按类别评估性能"""
    coco_gt = COCO(ground_truth)
    coco_dt = coco_gt.loadRes(predictions)

    class_aps = {}

    for cat_id in coco_gt.getCatIds():
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        class_aps[cat_id] = coco_eval.stats[0]  # mAP@0.5

    return class_aps
```

### 可视化工具
```python
def visualize_annotations(image_path, annotations, output_path):
    """可视化标注结果"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']

        # 创建矩形框
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

        # 添加类别标签
        ax.text(bbox[0], bbox[1], f'Class {category_id}',
                bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
```

## 性能优化

### 内存优化
```python
def process_large_dataset(annotation_file, batch_size=1000):
    """分批处理大型数据集"""
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    total_annotations = len(annotations)

    for i in range(0, total_annotations, batch_size):
        batch = annotations[i:i+batch_size]
        process_annotation_batch(batch)

        # 清理内存
        del batch
        if i % (batch_size * 10) == 0:
            import gc
            gc.collect()
```

### 并行处理
```python
from multiprocessing import Pool
import multiprocessing

def parallel_image_processing(image_list, num_processes=None):
    """并行处理图像数据"""
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with Pool(num_processes) as pool:
        results = pool.map(process_single_image, image_list)

    return results
```

## 使用示例

### 基础使用
```python
# 1. 转换 COCO 数据集
from utils_coco.coco_annotation import coco_annotation
coco_annotation(data_path="./coco_dataset")

# 2. 验证数据集
from utils_coco.coco_utils import validate_coco_annotations
is_valid, msg = validate_coco_annotations("annotations.json")
print(f"Validation: {is_valid}, Message: {msg}")

# 3. 获取统计信息
from utils_coco.coco_utils import get_coco_statistics, get_coco_class_mapping
stats = get_coco_statistics("annotations.json")
class_mapping = get_coco_class_mapping("annotations.json")
print(f"Images: {stats['num_images']}, Annotations: {stats['num_annotations']}")

# 4. 计算评估指标
from utils_coco.coco_evaluation import evaluate_coco
mAP_results = evaluate_coco("predictions.json", "annotations.json")
print(f"mAP@0.5: {mAP_results['mAP_50']:.4f}")
```

### 高级使用
```python
# 数据质量检查
from utils_coco.coco_utils import check_data_quality
issues = check_data_quality("annotations.json", "./images")
for issue in issues:
    print(f"Issue: {issue}")

# 可视化标注
from utils_coco.coco_utils import visualize_annotations
visualize_annotations("image1.jpg", annotations, "output.jpg")

# 自定义评估
from utils_coco.coco_evaluation import comprehensive_evaluation
results = comprehensive_evaluation("predictions.json", "annotations.json")
print(f"Overall mAP: {results['overall']['mAP']:.4f}")
```

## 错误处理

### 常见错误与解决方案
```python
def safe_load_annotations(annotation_file):
    """安全加载标注文件"""
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    except FileNotFoundError:
        print(f"Annotation file not found: {annotation_file}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def validate_bbox(bbox, image_size):
    """验证边界框有效性"""
    x, y, w, h = bbox
    img_w, img_h = image_size

    # 检查边界框是否在图像范围内
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return False, "Bounding box exceeds image boundaries"

    # 检查尺寸是否合理
    if w <= 0 or h <= 0:
        return False, "Invalid bounding box size"

    return True, "Valid bounding box"
```

## 常见问题 (FAQ)

### Q: COCO 数据集如何转换为 YOLO 格式？
A: 使用 `coco_annotation()` 函数，输入 COCO 数据路径，自动生成 train.txt、val.txt 和 classes.txt。

### Q: 如何验证 COCO 标注文件的正确性？
A: 使用 `validate_coco_annotations()` 函数检查必需字段、数据结构和格式正确性。

### Q: mAP 计算结果不准确怎么办？
A: 检查预测结果格式是否正确，确保 IoU 计算准确，验证类别 ID 映射是否一致。

### Q: 处理大型数据集时内存不足如何解决？
A: 使用分批处理、启用内存优化、减少并行进程数，或使用数据流式处理。

### Q: 自定义类别如何添加？
A: 在 categories 数组中添加新类别，确保 category_id 连续，重新运行转换脚本。

## 相关文件清单

### 核心功能文件
- `coco_annotation.py` - COCO 到 YOLO 格式转换，包含数据处理和文件生成
- `coco_evaluation.py` - COCO 评估指标计算，包含 mAP、精确率、召回率等
- `coco_utils.py` - COCO 工具函数，包含验证、统计、映射等功能

### 评估计算文件
- `get_map_coco.py` - mAP 计算的专用实现，支持多种 IoU 阈值

### 功能特点
- **数据验证**: 完整的 COCO 格式验证机制
- **格式转换**: 自动化 COCO 到 YOLO 格式转换
- **评估计算**: 标准 COCO 评估指标实现
- **统计分析**: 数据集统计信息和质量检查
- **可视化**: 标注结果可视化工具

---

*此文档由 AI 自动生成，最后更新时间: 2025-10-13 03:01:09*