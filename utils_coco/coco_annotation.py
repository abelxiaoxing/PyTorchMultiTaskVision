import json
import os
import random
from collections import defaultdict
from pathlib import Path

def process_annotations(image_annotations, data_path, output_path, images):
    name_box_id = defaultdict(list)
    for image_id, annotations in image_annotations.items():
        image_entry = next(image for image in images if image['id'] == image_id)
        file_name = image_entry['file_name']
        name = os.path.join(data_path, file_name)
        for ant in annotations:
            cat = ant["category_id"] - 1  # 大坑！coco的id是从1开始的，这里修改成从0开始
            name_box_id[name].append([ant["bbox"], cat])

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, key in enumerate(name_box_id.keys(), start=1):
            # print(f"Writing annotation {idx}/{len(name_box_id)}")  # 打印当前写入的是第几条记录
            f.write(key)
            box_infos = name_box_id[key]
            for info in box_infos:
                x_min = int(info[0][0])
                y_min = int(info[0][1])
                x_max = x_min + int(info[0][2])
                y_max = y_min + int(info[0][3])
                box_info = f" {x_min},{y_min},{x_max},{y_max},{int(info[1])}"
                f.write(box_info)
            f.write("\n")


def generate_classes(categories, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for category in categories:
            f.write(f"{category['name']}\n")

def coco_annotation(data_path = "coco_dataset"):       
    train_output_path = Path("train_det/train.txt")
    val_output_path = Path("train_det/val.txt")
    classes_output_path = Path("train_det/classes.txt")
    train_output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在的话）
    annotation_path = data_path + "/annotation.json"
    with open(annotation_path, encoding="utf-8") as f:
        data = json.load(f)

    annotations = data["annotations"]
    categories = data["categories"]
    images = data["images"]

    # 按图片分组注释
    image_annotations = defaultdict(list)
    for ant in annotations:
        image_annotations[ant["image_id"]].append(ant)

    # 获取所有图片ID，并进行随机划分
    all_image_ids = list(image_annotations.keys())
    random.shuffle(all_image_ids)
    split_index = int(0.9 * len(all_image_ids))
    print(f"Split index: {split_index}")

    train_image_ids = all_image_ids[:split_index]
    val_image_ids = all_image_ids[split_index:]

    # 创建训练和验证集的注释字典
    train_annotations = {image_id: image_annotations[image_id] for image_id in train_image_ids}
    val_annotations = {image_id: image_annotations[image_id] for image_id in val_image_ids}
    process_annotations(train_annotations, data_path, train_output_path, images)
    process_annotations(val_annotations, data_path, val_output_path, images)
    generate_classes(categories, classes_output_path)
if __name__ == "__main__":
    coco_annotation()