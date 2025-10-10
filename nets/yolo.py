from collections import OrderedDict
import timm
import torch
import torch.nn as nn
from .convnext_v2 import convnextv2_atto,convnextv2_pico,convnextv2_femto,convnextv2_nano,convnextv2_tiny,convnextv2_base

    
class ConvnextV2(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(ConvnextV2, self).__init__()
        convnextv2 = {
            "convnextv2_atto": convnextv2_atto,
            "convnextv2_femto": convnextv2_femto,
            "convnextv2_pico": convnextv2_pico,
            "convnextv2_nano": convnextv2_nano,
            "convnextv2_tiny": convnextv2_tiny,
            "convnextv2_base": convnextv2_base,
            }[backbone]
        model = convnextv2(pretrained)
        del model.head
        self.model = model

    def forward(self, x):
        x = self.model.downsample_layers[0](x)
        feat1 = self.model.stages[0](x)
        x = self.model.downsample_layers[1](feat1)
        feat2 = self.model.stages[1](x)
        x = self.model.downsample_layers[2](feat2)
        feat3 = self.model.stages[2](x)
        x = self.model.downsample_layers[3](feat3)
        feat4 = self.model.stages[3](x)
        x = self.model.norm(feat4.mean([-2, -1]))
        return [feat2, feat3, feat4]


class Efficientvit(nn.Module):
    def __init__(self, backbone='efficientvit_b0', pretrained=False):
        super(Efficientvit, self).__init__()
        efficientvit_models = {
            "efficientvit_b0": timm.models.efficientvit_b0,
            "efficientvit_b1": timm.models.efficientvit_b1,
            "efficientvit_b2": timm.models.efficientvit_b2,
            "efficientvit_b3": timm.models.efficientvit_b3,
            "efficientvit_l1": timm.models.efficientvit_l1,
            "efficientvit_l2": timm.models.efficientvit_l2,
            "efficientvit_l3": timm.models.efficientvit_l3,
        }
        efficientvit_model = efficientvit_models[backbone]
        self.model = efficientvit_model(pretrained=pretrained)

    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.stages[0](x)
        feat1 = x = self.model.stages[1](x)
        feat2 = x = self.model.stages[2](x)
        feat3 = x = self.model.stages[3](x)
        return [feat1, feat2, feat3]

def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "conv",
                    nn.Conv2d(
                        filter_in,
                        filter_out,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=pad,
                        groups=groups,
                        bias=False,
                    ),
                ),
                ("bn", nn.BatchNorm2d(filter_out)),
                ("relu", nn.ReLU6(inplace=True)),
            ]
        )
    )


def conv_dw(filter_in, filter_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),
        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )


# ---------------------------------------------------#
#   SPPF结构，利用不同大小的池化核进行池化
#   池化后堆叠
# ---------------------------------------------------#
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# ---------------------------------------------------#
#   三次卷积块
# ---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


# ---------------------------------------------------#
#   五次卷积块
# ---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


# ---------------------------------------------------#
#   最后获得yolov5的输出
# ---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.m = nn.Sequential(*[conv_dw(c_, c_) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat(
            (
                self.m(self.cv1(x)), 
                self.cv2(x)
            )
            , dim=1))
# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(
        self, anchors_mask, num_classes, backbone="convnextv2_atto", pretrained=False
    ):
        super(YoloBody, self).__init__()

        if backbone in ["efficientvit_b0","efficientvit_b1","efficientvit_b2","efficientvit_b3","efficientvit_l1","efficientvit_l2","efficientvit_l3",]:
            # ---------------------------------------------------#
            #   28,28,32; 14,14,64; 7,7,128
            # ---------------------------------------------------#
            self.backbone = Efficientvit(backbone,pretrained=pretrained)
            in_filters = {
                "efficientvit_b0": [32, 64, 128],
                "efficientvit_b1": [64, 128, 256],
                "efficientvit_b2": [96, 192, 384],
                "efficientvit_b3": [128, 256, 512],
                "efficientvit_l1": [128, 256, 512],
                "efficientvit_l2": [128, 256, 512],
                "efficientvit_l3": [256, 512, 1024],
            }[backbone]
        # elif backbone in ["convnextv2_atto", "convnextv2_femto", "convnextv2_pico", "convnextv2_nano", "convnextv2_tiny", "convnextv2_base"]:
        else:
            # ---------------------------------------------------#
            #   28,28,80；14,14,160；7,7,320
            # ---------------------------------------------------#
            self.backbone = ConvnextV2(backbone,pretrained=pretrained)
            in_filters = {
                "convnextv2_atto": [ 80, 160, 320],
                "convnextv2_femto": [ 96, 192, 384],
                "convnextv2_pico": [ 128, 256, 512],
                "convnextv2_nano": [ 160, 320, 640],
                "convnextv2_tiny": [ 192, 384, 768],
                "convnextv2_base": [ 256, 512, 1024],
            }[backbone]
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")
        self.SPPF = SPPF(c1=512,c2=512)

        self.conv_for_P5 = conv2d(in_filters[2], 512, 1)
        self.conv_for_feat3 = Conv(512, 256, 1, 1)
        self.conv3_for_upsample1    = C3(512, 256, 3, shortcut=False)

        self.conv_for_P4 = conv2d(in_filters[1], 256, 1)
        self.conv_for_feat2 = Conv(256, 128, 1, 1)
        self.conv3_for_upsample2 = C3(256, 128, 3, shortcut=False)

        self.conv_for_P3 = conv2d(in_filters[0], 128, 1)
        self.down_sample1 = conv_dw(128, 128, stride=2)
        self.conv3_for_downsample1  = C3(256, 256, 3, shortcut=False)
        self.down_sample2 = conv_dw(256, 256, stride=2)
        self.conv3_for_downsample2  = C3(512, 512, 3, shortcut=False)
        self.yolo_head_P3 = nn.Conv2d(128, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(256, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(512, len(anchors_mask[0]) * (5 + num_classes), 1)
    def forward(self, x):
        #  backbone for example using convnextv2_atto
        feat1, feat2, feat3 = self.backbone(x)
        # 28,28,80;14,14,160;7,7,320
        P5 = self.conv_for_P5(feat3)
        # P5:7,7,512
        P5 = self.SPPF(P5)
        # P5:7,7,512
        P5 = self.conv_for_feat3(P5)
        # P5:7,7,256
        P5_upsample = self.upsample(P5)
        # P5_upsample:14,14,256
        P4 = self.conv_for_P4(feat2)
        # P4:14,14,256
        P4 = torch.cat([P4, P5_upsample], axis=1)
        # P4:14,14,512
        P4 = self.conv3_for_upsample1(P4)
        # P4:14,14,256
        P4 = self.conv_for_feat2(P4)
        # P4:14,14,128
        P4_upsample = self.upsample(P4)
        # P4_upsample:28,28,128
        P3 = self.conv_for_P3(feat1)
        # P3:28,28,128
        P3 = torch.cat([P3, P4_upsample], axis=1)
        # P3:28,28,256
        P3 = self.conv3_for_upsample2(P3)
        # P3:28,28,128
        P3_downsample = self.down_sample1(P3)
        # P3_downsample:14,14,128
        P4 = torch.cat([P3_downsample, P4], axis=1)
        # P4:14,14,256
        P4 = self.conv3_for_downsample1(P4)
        # P4:14,14,256
        P4_downsample = self.down_sample2(P4)
        # P4_downsample:7,7,256
        P5 = torch.cat([P4_downsample, P5], axis=1)
        # P5:7,7,512
        P5 = self.conv3_for_downsample2(P5)
        # P5:7,7,512
        # ---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,3*(5+num_classes),28,28)
        # ---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        # ---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,3*(5+num_classes),14,14)
        # ---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        # ---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,3*(5+num_classes),7,7)
        # ---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)

        return out0, out1, out2
