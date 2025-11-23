"""COCO数据集加载器，支持YOLO训练流程。"""

import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data.dataset import Dataset

from utils.coco_utils import create_yolo_target_from_coco, get_coco_class_mapping
from utils.vision import cvtColor, preprocess_input
from utils.yolo_transforms import (
    adjust_boxes,
    apply_hsv_augmentation,
    letterbox_image,
    merge_bboxes,
    mixup_images,
    rand_uniform,
    place_on_canvas,
    resize_with_jitter,
)


class CocoYoloDataset(Dataset):
    """
    基于COCO API的YOLO数据集类
    完全兼容现有的YOLO训练流程，同时享受COCO API的强大功能！
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        input_size: int = 640,
        epoch_length: int = 100,
        mosaic: bool = True,
        mixup: bool = True,
        mosaic_prob: float = 0.5,
        mixup_prob: float = 0.5,
        train: bool = True,
        special_aug_ratio: float = 0.7,
        category_ids: Optional[List[int]] = None,
    ):
        """
        初始化COCO YOLO数据集

        Args:
            root: 图像根目录路径
            annFile: COCO标注文件路径
            input_size: 输入图像尺寸
            epoch_length: 训练轮数长度
            mosaic: 是否启用Mosaic数据增强
            mixup: 是否启用MixUp数据增强
            mosaic_prob: Mosaic增强概率
            mixup_prob: MixUp增强概率
            train: 是否为训练模式
            special_aug_ratio: 特殊增强比例
            category_ids: 指定使用的类别ID列表，None表示使用所有类别
        """
        self.root = root
        self.annFile = annFile
        self.input_size = input_size
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio
        self.category_ids = category_ids

        # 初始化COCO对象
        self.coco = COCO(annFile)

        # 获取类别映射
        self.class_mapping = get_coco_class_mapping(annFile)
        if not category_ids:
            # 默认使用所有类别，避免传入None导致COCO返回空标注
            self.category_ids = list(self.class_mapping.keys())
        else:
            # 只使用指定类别，保持映射顺序一致
            self.category_ids = category_ids
            self.class_mapping = {cat_id: self.class_mapping[cat_id] for cat_id in self.category_ids}
        self.category_to_class = {cat_id: i for i, cat_id in enumerate(self.category_ids)}

        # 获取图片列表
        self.image_ids = self.coco.getImgIds()
        if category_ids:
            # 过滤只包含指定类别的图片
            self.image_ids = self._filter_images_by_categories(self.category_ids)

        # 缓存图片信息以提高性能
        self.img_info_cache = {img_id: self.coco.loadImgs(img_id)[0] for img_id in self.image_ids}

        self.epoch_now = -1
        self.length = len(self.image_ids)

    def _filter_images_by_categories(self, category_ids: List[int]) -> List[int]:
        """过滤只包含指定类别的图片"""
        filtered_images = []
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=category_ids)
            if ann_ids:  # 如果该图片包含指定类别的标注
                filtered_images.append(img_id)
        return filtered_images

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        index = index % self.length
        img_id = self.image_ids[index]

        # 训练时进行数据的随机增强，验证时不进行数据的随机增强
        use_mosaic = (
            self.mosaic
            and rand_uniform() < self.mosaic_prob
            and self.epoch_now < self.epoch_length * self.special_aug_ratio
        )
        if use_mosaic:
            sample_indices = random.sample(range(self.length), 3)
            sample_indices.append(index)
            sample_img_ids = [self.image_ids[i] for i in sample_indices]

            image, boxes = self.get_random_data_with_mosaic(sample_img_ids, self.input_size)

            if self.mixup and rand_uniform() < self.mixup_prob:
                mixup_idx = random.randint(0, self.length - 1)
                mixup_img_id = self.image_ids[mixup_idx]
                image_2, boxes_2 = self.get_random_data(mixup_img_id, self.input_size, random=self.train)
                image, boxes = mixup_images(image, boxes, image_2, boxes_2)
        else:
            image, boxes = self.get_random_data(img_id, self.input_size, random=self.train)

        # 图像预处理
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        boxes = np.array(boxes, dtype=np.float32)

        # 归一化边界框坐标
        if len(boxes) != 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.input_size
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.input_size
            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]  # width, height
            boxes[:, 0:2] = boxes[:, 0:2] + boxes[:, 2:4] / 2  # center_x, center_y

        return image, boxes

    def get_image_and_annotations(self, img_id: int) -> Tuple[Image.Image, np.ndarray]:
        """获取图像和对应的标注"""
        img_info = self.img_info_cache[img_id]
        image_path = f"{self.root}/{img_info['file_name']}"

        # 读取图像
        image = Image.open(image_path)
        image = cvtColor(image)

        # 获取标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.category_ids)
        annotations = self.coco.loadAnns(ann_ids)

        # 转换为YOLO格式
        boxes = []
        for ann in annotations:
            if ann['area'] > 0:  # 忽略面积为0的标注
                yolo_target = create_yolo_target_from_coco(
                    ann, (img_info['width'], img_info['height']), self.category_to_class
                )
                if yolo_target is not None:
                    boxes.append(yolo_target)

        return image, np.array(boxes) if boxes else np.array([])

    def get_random_data(
        self,
        img_id: int,
        input_size: int,
        jitter: float = 0.3,
        hue: float = 0.1,
        sat: float = 0.7,
        val: float = 0.4,
        random: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """随机数据增强或验证阶段的letterbox预处理。"""
        image, box = self.get_image_and_annotations(img_id)

        if not random:
            return letterbox_image(image, box, input_size)

        if box.size > 0:
            box = box.copy()
            np.random.shuffle(box)

        resized, iw, ih, nw, nh = resize_with_jitter(
            image, input_size, jitter, (0.25, 2), rand_uniform
        )
        dx = int(rand_uniform(0, input_size - nw))
        dy = int(rand_uniform(0, input_size - nh))

        flip = rand_uniform() < 0.5
        if flip:
            resized = resized.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = place_on_canvas(resized, input_size, input_size, dx, dy)
        image_data = apply_hsv_augmentation(image_data, hue, sat, val)

        box = adjust_boxes(box, iw, ih, nw, nh, dx, dy, input_size, input_size, flip)
        return image_data, box

    def get_random_data_with_mosaic(
        self, img_ids: List[int], input_size: int, jitter: float = 0.3,
        hue: float = 0.1, sat: float = 0.7, val: float = 0.4
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Mosaic数据增强。"""
        h, w = input_size, input_size
        min_offset_x = rand_uniform(0.3, 0.7)
        min_offset_y = rand_uniform(0.3, 0.7)

        image_datas: List[np.ndarray] = []
        box_datas: List[np.ndarray] = []

        for i, img_id in enumerate(img_ids):
            image, box = self.get_image_and_annotations(img_id)
            if box.size > 0:
                box = box.copy()
                np.random.shuffle(box)
            resized, iw, ih, nw, nh = resize_with_jitter(
                image, input_size, jitter, (0.4, 1), rand_uniform
            )

            flip = rand_uniform() < 0.5
            if flip and box.size > 0:
                resized = resized.transpose(Image.FLIP_LEFT_RIGHT)
                box = box.copy()
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            if i == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif i == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif i == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            else:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            image_data = place_on_canvas(resized, w, h, dx, dy)
            box_adjusted = adjust_boxes(box, iw, ih, nw, nh, dx, dy, w, h, flip)
            box_data = np.zeros((len(box_adjusted), 5))
            if len(box_adjusted):
                box_data[: len(box_adjusted)] = box_adjusted

            image_datas.append(image_data)
            box_datas.append(box_data)

        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3], dtype=np.uint8)
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = apply_hsv_augmentation(new_image, hue, sat, val)
        new_boxes = merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        if self.category_ids:
            return [self.class_mapping[cat_id] for cat_id in self.category_ids]
        else:
            return list(self.class_mapping.values())


if __name__ == "__main__":
    # 测试代码
    print("COCO YOLO数据集测试...")

    # 这里需要根据实际路径调整
    root = "C:/datas/COCO2017/train2017"
    annFile = "C:/datas/COCO2017/annotations/instances_train2017.json"

    try:
        dataset = CocoYoloDataset(
            root=root,
            annFile=annFile,
            input_size=640,
            train=True
        )

        print(f"数据集大小: {len(dataset)}")
        print(f"类别数量: {len(dataset.get_class_names())}")

        # 测试数据加载
        if len(dataset) > 0:
            image, boxes = dataset[0]
            print(f"图像形状: {image.shape}")
            print(f"边界框数量: {len(boxes)}")

        print("测试完成！o(￣▽￣)ｄ")

    except Exception as e:
        print(f"测试失败: {e}")
        print("请检查COCO数据集路径是否正确... (｡•́︿•̀｡)")
