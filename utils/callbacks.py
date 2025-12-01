import os
import tempfile

import torch
import matplotlib

matplotlib.use("Agg")
import scipy.signal
from matplotlib import pyplot as plt
import numpy as np

from PIL import Image
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from .vision import cvtColor, preprocess_input, resize_image
from .utils_bbox import DecodeBox
from .utils_map import get_coco_map, get_map

# 定义一个类LossHistory，用于记录损失值
class LossHistory:
    # 初始化函数，传入参数figure_dir，model，input_size
    def __init__(self, figure_dir, model, input_size):
        # 定义一个变量figure_dir，传入参数figure_dir
        self.figure_dir = figure_dir
        # 定义一个列表losses，用于存储损失值
        self.losses = []
        # 定义一个列表val_loss，用于存储验证损失值
        self.val_loss = []
        # 如果路径figure_dir不存在，则创建该路径
        os.makedirs(self.figure_dir, exist_ok=True)


    # 定义一个函数append_loss，用于添加损失值和验证损失值
    def append_loss(self, epoch, loss, val_loss):
        # 如果路径figure_dir不存在，则创建该路径
        if not os.path.exists(self.figure_dir):
            os.makedirs(self.figure_dir, exist_ok=True)

        # 将损失值添加到losses列表中
        self.losses.append(loss)
        # 将验证损失值添加到val_loss列表中
        self.val_loss.append(val_loss)
        # 定义一个变量log_file_path，用于存储日志文件的路径
        log_file_path = os.path.join(os.path.dirname(self.figure_dir), "log.txt")
        # 打开日志文件，追加内容
        with open(log_file_path, "a") as f:
            # 写入内容
            f.write(f"Epoch {epoch} - Train Loss: {loss}, Validation Loss: {val_loss}\n")
        # 调用函数loss_plot，绘制损失图
        self.loss_plot()

    # 定义一个函数loss_plot，用于绘制损失图
    def loss_plot(self):
        # 定义一个变量iters，用于存储迭代次数
        iters = range(len(self.losses))

        # 创建一个图形
        plt.figure()
        # 绘制损失值
        plt.plot(iters, self.losses, "red", linewidth=2, label="train loss")
        # 绘制验证损失值
        plt.plot(iters, self.val_loss, "coral", linewidth=2, label="val loss")
        try:
            # 如果损失值列表长度小于25，则设置num为5
            if len(self.losses) < 25:
                num = 5
            # 否则，设置num为15
            else:
                num = 15

            # 使用 Savitzky-Golay 算法平滑损失值
            plt.plot(
                iters,
                scipy.signal.savgol_filter(self.losses, num, 3),
                "green",
                linestyle="--",
                linewidth=2,
                label="smooth train loss",
            )
            plt.plot(
                iters,
                scipy.signal.savgol_filter(self.val_loss, num, 3),
                "#8B4513",
                linestyle="--",
                linewidth=2,
                label="smooth val loss",
            )
        except:
            pass

        # 添加网格
        plt.grid(True)
        # 添加x轴标签
        plt.xlabel("Epoch")
        # 添加y轴标签
        plt.ylabel("Loss")
        # 添加图例
        plt.legend(loc="upper right")

        # 保存损失图
        plt.savefig(os.path.join(self.figure_dir, "epoch_loss.png"))

        # 清除图形
        plt.cla()
        # 关闭所有图形
        plt.close("all")


class EvalCallback:
    def __init__(
        self,
        net,
        input_size,
        anchors,
        anchors_mask,
        class_names,
        num_classes,
        val_lines,
        figure_dir,
        device,
        val_dataset=None,
        max_boxes=100,
        confidence=0.05,
        nms_iou=0.5,
        letterbox_image=True,
        MINOVERLAP=0.5,
        eval_flag=True,
        period=1,
        num_epochs=10,
    ):
        super(EvalCallback, self).__init__()

        # 定义网络
        self.net = net
        # 定义输入尺寸
        self.input_size = input_size
        # 定义锚框
        self.anchors = anchors
        # 定义锚框的掩码
        self.anchors_mask = anchors_mask
        # 定义类别名称
        self.class_names = class_names
        # 定义类别数量
        self.num_classes = num_classes
        # 定义验证集的行数
        self.val_lines = val_lines
        # 定义图片保存的文件夹
        self.figure_dir = figure_dir
        # 定义设备
        self.device = device
        self.val_dataset = val_dataset
        # 定义最大检测到的框数
        self.max_boxes = max_boxes
        # 定义置信度
        self.confidence = confidence
        # 定义NMS的IOU
        self.nms_iou = nms_iou
        # 定义是否使用letterbox_image
        self.letterbox_image = letterbox_image
        # 定义MINOVERLAP
        self.MINOVERLAP = MINOVERLAP
        # 定义是否评估
        self.eval_flag = eval_flag
        # 定义评估周期
        self.period = period
        # 定义epoch数量
        self.num_epochs = num_epochs

        # 定义解码框
        self.bbox_util = DecodeBox(
            self.anchors,
            self.num_classes,
            self.input_size,
            self.anchors_mask,
            device=self.device,
        )

        # 定义mAP数组
        self.maps = [0]
        # 定义周期数组
        self.epoches = [0]
        # 如果需要评估
        if self.eval_flag:
            # 打开日志文件
            with open(os.path.join(os.path.dirname(self.figure_dir), "log.txt"), "a") as f:
                # 写入mAP为0
                f.write(f"mAP={str(0)}\n")

    def _predict_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(
            image, (self.input_size, self.input_size), self.letterbox_image
        )
        image_data = np.expand_dims(
            np.transpose(
                preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)
            ),
            0,
        )
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.to(self.device)
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(
                torch.cat(outputs, 1),
                self.num_classes,
                self.input_size,
                image_shape,
                self.letterbox_image,
                conf_thres=self.confidence,
                nms_thres=self.nms_iou,
            )
            if results[0] is None:
                return []

            top_label = np.array(results[0][:, 6], dtype="int32")
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        top_100 = np.argsort(top_conf)[::-1][: self.max_boxes]
        top_boxes = top_boxes[top_100]
        top_conf = top_conf[top_100]
        top_label = top_label[top_100]

        detections = []
        for label, score, box in zip(top_label, top_conf, top_boxes):
            detections.append(
                {"label": int(label), "score": float(score), "bbox": box.astype(float)}
            )
        return detections

    def _build_coco_detections(self, detections, image_id):
        if self.val_dataset is None or not hasattr(self.val_dataset, "category_to_class"):
            return []
        class_to_category = {
            cls_idx: cat_id for cat_id, cls_idx in self.val_dataset.category_to_class.items()
        }
        coco_detections = []
        for det in detections:
            category_id = class_to_category.get(det["label"])
            if category_id is None:
                continue
            left, top, right, bottom = det["bbox"]
            coco_detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [
                        float(left),
                        float(top),
                        float(right - left),
                        float(bottom - top),
                    ],
                    "score": float(det["score"]),
                }
            )
        return coco_detections

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        detections = self._predict_image(image)
        result_path = os.path.join(map_out_path, "detection-results/" + image_id + ".txt")
        if len(detections) == 0:
            open(result_path, "w", encoding="utf-8").close()
            return
        with open(result_path, "w", encoding="utf-8") as f:
            for det in detections:
                predicted_class = self.class_names[int(det["label"])]
                if predicted_class not in class_names:
                    continue
                top, left, bottom, right = det["bbox"]
                score = str(det["score"])
                f.write(
                    "%s %s %s %s %s %s\n"
                    % (
                        predicted_class,
                        score[:6],
                        str(int(left)),
                        str(int(top)),
                        str(int(right)),
                        str(int(bottom)),
                    )
                )
        return

    def _evaluate_in_memory_coco(self):
        if self.val_dataset is None or not hasattr(self.val_dataset, "coco"):
            return None
        print("使用COCO API在内存中计算mAP50。")
        image_ids = self.val_lines if self.val_lines else getattr(self.val_dataset, "image_ids", [])
        coco_detections = []
        for annotation_line in tqdm(image_ids):
            if isinstance(annotation_line, (np.integer, int)):
                image_id = int(annotation_line)
            else:
                image_id = int(annotation_line)
            image, _ = self.val_dataset.get_image_and_annotations(image_id)
            detections = self._predict_image(image)
            coco_detections.extend(self._build_coco_detections(detections, image_id))

        if len(coco_detections) == 0:
            print("未检测到任何目标。")
            return 0

        coco_gt = self.val_dataset.coco
        coco_dt = coco_gt.loadRes(coco_detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return float(coco_eval.stats[1])

    def _evaluate_with_files(self):
        with tempfile.TemporaryDirectory() as map_out_path:
            os.makedirs(os.path.join(map_out_path, "ground-truth"), exist_ok=True)
            os.makedirs(os.path.join(map_out_path, "detection-results"), exist_ok=True)
            print("Get map.")
            for annotation_line in tqdm(self.val_lines):
                if isinstance(annotation_line, str):
                    line = annotation_line.split()
                    image_id = os.path.basename(line[0]).split(".")[0]
                    image = Image.open(line[0])
                    gt_boxes = np.array(
                        [np.array(list(map(int, box.split(",")))) for box in line[1:]]
                    )
                elif self.val_dataset is not None:
                    image_id = annotation_line
                    image, gt_boxes = self.val_dataset.get_image_and_annotations(
                        image_id
                    )
                    image_id = os.path.splitext(
                        self.val_dataset.img_info_cache[image_id]["file_name"]
                    )[0]
                    if gt_boxes.size != 0:
                        gt_boxes = gt_boxes.astype(np.int32)
                else:
                    raise TypeError(
                        "EvalCallback expects string annotation lines or a val_dataset reference."
                    )
                self.get_map_txt(image_id, image, self.class_names, map_out_path)
                with open(
                    os.path.join(
                        map_out_path, "ground-truth/" + image_id + ".txt"
                    ),
                    "w",
                ) as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write(
                            "%s %s %s %s %s\n" % (obj_name, left, top, right, bottom)
                        )

            print("Calculate Map.")
            try:
                temp_map = get_coco_map(
                    class_names=self.class_names, path=map_out_path
                )[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, path=map_out_path)
            return temp_map

    def on_epoch_end(self, epoch, model_eval):
        current_epoch = epoch + 1
        if ((current_epoch % self.period == 0) or (current_epoch == self.num_epochs)) and self.eval_flag:
            self.net = model_eval
            if self.val_dataset is not None and hasattr(self.val_dataset, "coco"):
                temp_map = self._evaluate_in_memory_coco()
            else:
                temp_map = self._evaluate_with_files()
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(os.path.dirname(self.figure_dir), "epoch_map.txt"), "a") as f:
                f.write(str(temp_map))
                f.write("\n")

            plt.figure()
            plt.plot(self.epoches, self.maps, "red", linewidth=2, label="train map")

            plt.grid(True)
            plt.xlabel("Epoch")
            plt.ylabel("Map %s" % str(self.MINOVERLAP))
            plt.title("A Map Curve")
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.figure_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            return temp_map
