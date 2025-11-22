"""
COCO数据集加载器 - 完美替换TXT格式！
基于torchvision.datasets.CocoDetection实现，支持YOLO训练流程
Author: 哈雷酱大小姐 (￣▽￣)ゞ
"""

import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from typing import List, Dict, Tuple, Optional, Any

from utils.utils import cvtColor, preprocess_input
from utils.coco_utils import (
    coco_to_yolo_bbox,
    create_yolo_target_from_coco,
    get_coco_class_mapping,
    filter_coco_annotations
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
        num_classes: int = 80,
        epoch_length: int = 100,
        mosaic: bool = True,
        mixup: bool = True,
        mosaic_prob: float = 0.5,
        mixup_prob: float = 0.5,
        train: bool = True,
        special_aug_ratio: float = 0.7,
        category_ids: Optional[List[int]] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ):
        """
        初始化COCO YOLO数据集

        Args:
            root: 图像根目录路径
            annFile: COCO标注文件路径
            input_size: 输入图像尺寸
            num_classes: 类别数量
            epoch_length: 训练轮数长度
            mosaic: 是否启用Mosaic数据增强
            mixup: 是否启用MixUp数据增强
            mosaic_prob: Mosaic增强概率
            mixup_prob: MixUp增强概率
            train: 是否为训练模式
            special_aug_ratio: 特殊增强比例
            category_ids: 指定使用的类别ID列表，None表示使用所有类别
            transform: 图像变换
            target_transform: 目标变换
        """
        self.root = root
        self.annFile = annFile
        self.input_size = input_size
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio
        self.category_ids = category_ids
        self.transform = transform
        self.target_transform = target_transform

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray]:
        index = index % self.length
        img_id = self.image_ids[index]

        # 训练时进行数据的随机增强，验证时不进行数据的随机增强
        if (self.mosaic and self.rand() < self.mosaic_prob and
            self.epoch_now < self.epoch_length * self.special_aug_ratio):

            # Mosaic增强
            sample_indices = random.sample(range(self.length), 3)
            sample_indices.append(index)
            sample_img_ids = [self.image_ids[i] for i in sample_indices]

            image, boxes = self.get_random_data_with_mosaic(sample_img_ids, self.input_size)

            if self.mixup and self.rand() < self.mixup_prob:
                mixup_idx = random.randint(0, self.length - 1)
                mixup_img_id = self.image_ids[mixup_idx]
                image_2, boxes_2 = self.get_random_data(mixup_img_id, self.input_size, random=self.train)
                image, boxes = self.get_random_data_with_mixup(image, boxes, image_2, boxes_2)
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

        return torch.from_numpy(image), boxes

    def rand(self, a: float = 0, b: float = 1) -> float:
        """生成随机数"""
        return np.random.rand() * (b - a) + a

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
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        随机数据增强，完全复用原有的逻辑！
        """
        image, box = self.get_image_and_annotations(img_id)

        if not random:
            # 验证模式：只进行resize和padding
            iw, ih = image.size
            h, w = input_size, input_size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box

        # 训练模式：完整的数据增强
        iw, ih = image.size
        h, w = input_size, input_size

        # 长宽比扭曲和缩放
        new_ar = (iw / ih * self.rand(1 - jitter, 1 + jitter) /
                  self.rand(1 - jitter, 1 + jitter))
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 随机放置
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 随机翻转
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)

        # 色域变换
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
        )
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # 调整边界框
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def get_random_data_with_mosaic(
        self, img_ids: List[int], input_size: int, jitter: float = 0.3,
        hue: float = 0.1, sat: float = 0.7, val: float = 0.4
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Mosaic数据增强，完全复用原有逻辑！"""
        h, w = input_size, input_size
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []

        for i, img_id in enumerate(img_ids):
            image, box = self.get_random_data(img_id, input_size, random=False)

            # 获取原始图像信息用于Mosaic
            img_info = self.img_info_cache[img_id]
            iw, ih = img_info['width'], img_info['height']

            # 重新读取原始图像进行Mosaic处理
            image_path = f"{self.root}/{img_info['file_name']}"
            image = Image.open(image_path)
            image = cvtColor(image)

            # 重新获取标注
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.category_ids)
            annotations = self.coco.loadAnns(ann_ids)
            boxes = []
            for ann in annotations:
                if ann['area'] > 0:
                    yolo_target = create_yolo_target_from_coco(
                        ann, (iw, ih), self.category_to_class
                    )
                    if yolo_target is not None:
                        boxes.append(yolo_target)
            box = np.array(boxes) if boxes else np.array([])

            # 翻转处理
            flip = self.rand() < 0.5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # 缩放处理
            new_ar = (iw / ih * self.rand(1 - jitter, 1 + jitter) /
                      self.rand(1 - jitter, 1 + jitter))
            scale = self.rand(0.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 四个位置放置
            if i == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif i == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif i == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif i == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            # 处理边界框
            box_data = []
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[: len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # 合并四张图片
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)

        # 色域变换
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
        )
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        # 合并边界框
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def merge_bboxes(self, bboxes: List[np.ndarray], cutx: int, cuty: int) -> List[np.ndarray]:
        """合并Mosaic的边界框，复用原有逻辑！"""
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                if len(box) == 0:
                    continue

                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                tmp_box.extend([x1, y1, x2, y2, box[4]])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_mixup(
        self, image_1: np.ndarray, box_1: List, image_2: np.ndarray, box_2: List
    ) -> Tuple[np.ndarray, List]:
        """MixUp数据增强，复用原有逻辑！"""
        new_image = (np.array(image_1, np.float32) * 0.5 +
                    np.array(image_2, np.float32) * 0.5)

        def to_box_array(box):
            arr = np.asarray(box, dtype=np.float32)
            if arr.size == 0:
                return np.zeros((0, 5), dtype=np.float32)
            return arr.reshape(-1, 5)

        box_1_arr = to_box_array(box_1)
        box_2_arr = to_box_array(box_2)
        new_boxes = np.concatenate([box_1_arr, box_2_arr], axis=0)

        return new_image, new_boxes

    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        if self.category_ids:
            return [self.class_mapping[cat_id] for cat_id in self.category_ids]
        else:
            return list(self.class_mapping.values())


def coco_dataset_collate(batch: List) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    COCO数据集的collate函数，与原有函数兼容
    """
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]

    return images, bboxes


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
