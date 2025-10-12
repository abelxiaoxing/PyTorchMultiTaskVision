"""
COCO数据集工具函数
用于COCO格式和YOLO格式之间的转换
Author: 哈雷酱大小姐 (o(￣▽￣)ｄ)
"""

import numpy as np
from pycocotools.coco import COCO
from typing import List, Dict, Tuple, Any


def coco_to_yolo_bbox(coco_bbox: List[float]) -> List[int]:
    """
    将COCO格式的边界框 [x, y, width, height] 转换为YOLO格式 [x_min, y_min, x_max, y_max]

    Args:
        coco_bbox: COCO格式边界框 [x, y, width, height]

    Returns:
        YOLO格式边界框 [x_min, y_min, x_max, y_max]
    """
    x, y, w, h = coco_bbox
    x_min = int(x)
    y_min = int(y)
    x_max = int(x + w)
    y_max = int(y + h)
    return [x_min, y_min, x_max, y_max]


def yolo_to_coco_bbox(yolo_bbox: List[int]) -> List[float]:
    """
    将YOLO格式的边界框 [x_min, y_min, x_max, y_max] 转换为COCO格式 [x, y, width, height]

    Args:
        yolo_bbox: YOLO格式边界框 [x_min, y_min, x_max, y_max]

    Returns:
        COCO格式边界框 [x, y, width, height]
    """
    x_min, y_min, x_max, y_max = yolo_bbox
    x = float(x_min)
    y = float(y_min)
    w = float(x_max - x_min)
    h = float(y_max - y_min)
    return [x, y, w, h]


def get_coco_class_mapping(coco_annotation_path: str) -> Dict[int, str]:
    """
    获取COCO类别映射，从category_id(1-based)到class_index(0-based)

    Args:
        coco_annotation_path: COCO标注文件路径

    Returns:
        类别映射字典 {category_id: class_name}
    """
    coco = COCO(coco_annotation_path)
    cats = coco.loadCats(coco.getCatIds())

    # COCO的category_id是从1开始的，我们转换为从0开始的索引
    class_mapping = {}
    for i, cat in enumerate(cats):
        class_mapping[cat['id']] = cat['name']

    return class_mapping


def filter_coco_annotations(
    coco: COCO,
    image_ids: List[int],
    category_ids: List[int] = None
) -> Dict[int, List[Dict]]:
    """
    过滤COCO标注，只保留指定图片和类别的标注

    Args:
        coco: COCO对象
        image_ids: 图片ID列表
        category_ids: 类别ID列表，如果为None则保留所有类别

    Returns:
        过滤后的标注字典 {image_id: [annotations]}
    """
    filtered_annotations = {}

    for img_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=category_ids)
        anns = coco.loadAnns(ann_ids)

        if anns:  # 只保留有标注的图片
            filtered_annotations[img_id] = anns

    return filtered_annotations


def validate_coco_annotations(coco_annotation_path: str) -> Tuple[bool, str]:
    """
    验证COCO标注文件的完整性

    Args:
        coco_annotation_path: COCO标注文件路径

    Returns:
        (is_valid, error_message)
    """
    try:
        coco = COCO(coco_annotation_path)

        # 检查基本结构
        if not hasattr(coco, 'dataset'):
            return False, "Invalid COCO dataset structure"

        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in coco.dataset:
                return False, f"Missing required key: {key}"

        # 检查数据一致性
        img_ids = set(coco.getImgIds())
        ann_img_ids = set(ann['image_id'] for ann in coco.dataset['annotations'])

        if not ann_img_ids.issubset(img_ids):
            invalid_anns = ann_img_ids - img_ids
            return False, f"Annotations reference non-existent images: {invalid_anns}"

        return True, "COCO annotation is valid"

    except Exception as e:
        return False, f"Error validating COCO annotation: {str(e)}"


def get_coco_statistics(coco_annotation_path: str) -> Dict[str, Any]:
    """
    获取COCO数据集的统计信息

    Args:
        coco_annotation_path: COCO标注文件路径

    Returns:
        统计信息字典
    """
    coco = COCO(coco_annotation_path)

    # 基本统计
    num_images = len(coco.getImgIds())
    num_annotations = len(coco.getAnnIds())
    num_categories = len(coco.getCatIds())

    # 每个类别的标注数量
    category_stats = {}
    for cat_id in coco.getCatIds():
        ann_ids = coco.getAnnIds(catIds=cat_id)
        category_stats[cat_id] = len(ann_ids)

    # 每张图片的标注数量统计
    anns_per_img = []
    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns_per_img.append(len(ann_ids))

    stats = {
        'num_images': num_images,
        'num_annotations': num_annotations,
        'num_categories': num_categories,
        'avg_annotations_per_image': np.mean(anns_per_img) if anns_per_img else 0,
        'max_annotations_per_image': max(anns_per_img) if anns_per_img else 0,
        'min_annotations_per_image': min(anns_per_img) if anns_per_img else 0,
        'category_stats': category_stats
    }

    return stats


def create_yolo_target_from_coco(
    coco_annotation: Dict,
    image_size: Tuple[int, int],
    class_mapping: Dict[int, int]
) -> np.ndarray:
    """
    从COCO标注创建YOLO格式的目标张量

    Args:
        coco_annotation: COCO标注对象
        image_size: 图像尺寸 (width, height)
        class_mapping: 类别映射 {category_id: class_index}

    Returns:
        YOLO格式目标张量，形状为 (N, 5)，格式为 [x_min, y_min, x_max, y_max, class_index]
    """
    yolo_bbox = coco_to_yolo_bbox(coco_annotation['bbox'])
    category_id = coco_annotation['category_id']

    # 转换类别ID
    if category_id in class_mapping:
        class_index = class_mapping[category_id]
    else:
        # 如果找不到类别，跳过这个标注
        return None

    # 验证边界框是否有效
    x_min, y_min, x_max, y_max = yolo_bbox
    img_w, img_h = image_size

    # 确保边界框在图像范围内
    x_min = max(0, min(x_min, img_w - 1))
    y_min = max(0, min(y_min, img_h - 1))
    x_max = max(x_min + 1, min(x_max, img_w))
    y_max = max(y_min + 1, min(y_max, img_h))

    # 检查边界框大小
    if x_max - x_min < 1 or y_max - y_min < 1:
        return None

    return np.array([x_min, y_min, x_max, y_max, class_index], dtype=np.float32)


if __name__ == "__main__":
    # 测试函数
    print("COCO工具函数测试...")

    # 测试边界框转换
    coco_bbox = [100, 50, 200, 150]
    yolo_bbox = coco_to_yolo_bbox(coco_bbox)
    back_to_coco = yolo_to_coco_bbox(yolo_bbox)

    print(f"COCO格式: {coco_bbox}")
    print(f"YOLO格式: {yolo_bbox}")
    print(f"转回COCO: {back_to_coco}")

    print("测试完成！(￣▽￣)ノ")