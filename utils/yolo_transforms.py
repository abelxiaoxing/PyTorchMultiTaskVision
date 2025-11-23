from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def rand_uniform(a: float = 0.0, b: float = 1.0) -> float:
    """Return a uniform random float in [a, b)."""
    return float(np.random.rand() * (b - a) + a)


def apply_hsv_augmentation(image: np.ndarray, hue: float, sat: float, val: float) -> np.ndarray:
    """Apply HSV jitter on an RGB image array."""
    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    hue_channel, sat_channel, val_channel = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    dtype = image.dtype

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    jittered = cv2.merge(
        (cv2.LUT(hue_channel, lut_hue), cv2.LUT(sat_channel, lut_sat), cv2.LUT(val_channel, lut_val))
    )
    return cv2.cvtColor(jittered, cv2.COLOR_HSV2RGB)


def resize_with_jitter(
    image: Image.Image,
    input_size: int,
    jitter: float,
    scale_range: Tuple[float, float],
    rand_fn: Callable[[float, float], float] = rand_uniform,
) -> Tuple[Image.Image, int, int, int, int]:
    """
    Resize an image with random aspect-ratio jitter and scaling.

    Returns:
        resized image, original width, original height, new width, new height
    """
    iw, ih = image.size
    w = h = input_size
    new_ar = (iw / ih * rand_fn(1 - jitter, 1 + jitter) / rand_fn(1 - jitter, 1 + jitter))
    scale = rand_fn(*scale_range)

    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)

    resized = image.resize((nw, nh), Image.BICUBIC)
    return resized, iw, ih, nw, nh


def place_on_canvas(image: Image.Image, w: int, h: int, dx: int, dy: int) -> np.ndarray:
    """Place an image onto a padded canvas."""
    canvas = Image.new("RGB", (w, h), (128, 128, 128))
    canvas.paste(image, (dx, dy))
    return np.array(canvas, dtype=np.uint8)


def adjust_boxes(
    boxes: np.ndarray, iw: int, ih: int, nw: int, nh: int, dx: int, dy: int, w: int, h: int, flip: bool
) -> np.ndarray:
    """Scale, translate, flip, and clip boxes; remove degenerate boxes."""
    if boxes.size == 0:
        return boxes

    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy
    if flip:
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

    boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
    boxes[:, 2][boxes[:, 2] > w] = w
    boxes[:, 3][boxes[:, 3] > h] = h

    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    return boxes[np.logical_and(box_w > 1, box_h > 1)]


def letterbox_image(image: Image.Image, boxes: np.ndarray, input_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resize with unchanged aspect ratio and gray padding."""
    iw, ih = image.size
    w = h = input_size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    resized = image.resize((nw, nh), Image.BICUBIC)
    image_data = place_on_canvas(resized, w, h, dx, dy).astype(np.float32)
    adjusted_boxes = adjust_boxes(boxes, iw, ih, nw, nh, dx, dy, w, h, flip=False)
    return image_data, adjusted_boxes


def merge_bboxes(bboxes: List[np.ndarray], cutx: int, cuty: int) -> List[List[float]]:
    """Merge bounding boxes from four mosaic tiles."""
    merge_bbox: List[List[float]] = []
    for i, bbox in enumerate(bboxes):
        for box in bbox:
            if len(box) == 0:
                continue

            x1, y1, x2, y2, cls_id = box[0], box[1], box[2], box[3], box[4]

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

            merge_bbox.append([x1, y1, x2, y2, cls_id])
    return merge_bbox


def mixup_images(
    image_1: np.ndarray, box_1: np.ndarray, image_2: np.ndarray, box_2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend two images/boxes for MixUp."""
    new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
    box_1_arr = np.asarray(box_1, dtype=np.float32).reshape(-1, 5) if np.size(box_1) else np.zeros((0, 5))
    box_2_arr = np.asarray(box_2, dtype=np.float32).reshape(-1, 5) if np.size(box_2) else np.zeros((0, 5))
    new_boxes = np.concatenate([box_1_arr, box_2_arr], axis=0)
    return new_image, new_boxes


def collate_yolo_batch(batch: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """YOLO数据集通用 collate 函数，避免按维度自动拼接目标。"""
    images, bboxes = zip(*batch)
    images_tensor = torch.from_numpy(np.stack(images)).float()
    bbox_tensors = [torch.as_tensor(ann, dtype=torch.float32) for ann in bboxes]
    return images_tensor, bbox_tensors
