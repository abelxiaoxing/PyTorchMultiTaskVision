
import os
import shutil
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV3
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# 初始化模型的辅助函数
def initialize_model(model_weight_path, model_ema, device):
    checkpoint = torch.load(model_weight_path, map_location=device,weights_only=False)
    num_classes = checkpoint["num_classes"]
    if model_ema:
        model = checkpoint["model"]
        model_ema = ModelEmaV3(model,decay=0.999,device=device)
        if 'model_ema' in checkpoint.keys():
            model_ema.module.load_state_dict(checkpoint['model_ema'])
            print(f"initialize model_ema success")
        else:
            model_ema.module.load_state_dict(checkpoint['model'].state_dict())
        return model_ema.module,num_classes
    else:
        model= checkpoint["model"]
        return model,num_classes

# 创建数据变换的辅助函数
def create_data_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

# 根据模型预测移动图像
def val_move(img_path, model_weight_path, img_size, model_ema, device):

    empty_path = os.path.join(os.path.dirname(img_path), "Empty")
    non_empty_path = os.path.join(os.path.dirname(img_path), "NonEmpty")
    os.makedirs(empty_path, exist_ok=True)
    os.makedirs(non_empty_path, exist_ok=True)

    data_transform = create_data_transform(img_size)
    model,_ = initialize_model(model_weight_path, model_ema, device)
    model.eval()
    for file_name in os.listdir(img_path):
        file_path = os.path.join(img_path, file_name)
        img = Image.open(file_path).convert("RGB")
        img = data_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = torch.squeeze(model(img)).cpu()
            predict = torch.softmax(output, dim=0)

        predicted_class_index = torch.argmax(predict).item()
        target_path = empty_path if predicted_class_index == 0 else non_empty_path
        shutil.move(file_path, os.path.join(target_path, file_name))

# 计算并打印精确率和召回率
def val_precision(img_path, model_weight_path, img_size, model_ema, device):
    data_transform = create_data_transform(img_size)
    dataset = ImageFolder(root=img_path, transform=data_transform)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss()

    model,num_classes = initialize_model(model_weight_path, model_ema, device)
    model.eval()
    true_positives = [0] * num_classes
    false_positives = [0] * num_classes
    false_negatives = [0] * num_classes

    for images, target in data_loader:
        images, target = images.to(device), target.to(device)
        output = model(images)
        loss = criterion(output, target)
        _, preds = torch.max(output, 1)

        for i in range(num_classes):
            true_positives[i] += torch.sum((preds == i) & (target == i)).item()
            false_positives[i] += torch.sum((preds == i) & (target != i)).item()
            false_negatives[i] += torch.sum((preds != i) & (target == i)).item()

    for i in range(num_classes):
        precision = true_positives[i] / (true_positives[i] + false_positives[i]) if true_positives[i] + false_positives[i] > 0 else 0
        recall = true_positives[i] / (true_positives[i] + false_negatives[i]) if true_positives[i] + false_negatives[i] > 0 else 0
        print(f'Precision{i}: {precision:.5f}, Recall{i}: {recall:.5f}')

if __name__ == "__main__":

    img_path = ""
    model_weight_path = "train_cls/output/checkpoint-0.pth"
    img_size = 224
    model_ema = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Start calculation!")
    # val_move(img_path, model_weight_path, img_size, model_ema, device)
    val_precision(img_path, model_weight_path, img_size, model_ema, device)
