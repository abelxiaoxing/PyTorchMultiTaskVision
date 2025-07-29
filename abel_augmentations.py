import random
import PIL
import numpy as np
import torch
from PIL import Image
_IMAGENET_PCA = {
    "eigval": [0.2175, 0.0188, 0.0045],
    "eigvec": [
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ],
}
_IMAGENET_PCA_GRAY = {
    "eigval": [0.2175],
    "eigvec": [
        [-0.5675],
        [-0.5808],
        [-0.5836],
    ],
}


def ShearX(img, min_val, max_val):  # [-0.3, 0.3]
    v = random.uniform(min_val, max_val)
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, min_val, max_val):  # [-0.3, 0.3]
    v = random.uniform(min_val, max_val)
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, min_val, max_val):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = random.uniform(min_val, max_val)
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, min_val, max_val):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = random.uniform(min_val, max_val)
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, min_val, max_val):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = random.uniform(min_val, max_val)
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, min_val, max_val):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = random.uniform(min_val, max_val)
    # if random.random() > 0.5:
    #     v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, min_val, max_val):  # [-30, 30]
    v = random.uniform(min_val, max_val)
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _, __):
    # img = set_dark_pixels_to_zero(img, 8, 12)
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _, __):
    return PIL.ImageOps.invert(img)


def Equalize(img, _, __):
    # img=set_dark_pixels_to_zero(img, 8, 12)
    return PIL.ImageOps.equalize(img)


def Flip(img, _, __):  # not from the paper
    return PIL.ImageOps.mirror(img)


def set_dark_pixels_to_zero(img, min, max):
    threshold=random.randint(min,max)
    lut = []
    for i in range(256):
        if i < threshold:
            lut.append(0)
        else:
            lut.append(i)

    return PIL.ImageOps._lut(img, lut)


def Solarize(img, min_val, max_val):  # [0, 256]
    v = random.uniform(min_val, max_val)
    return PIL.ImageOps.solarize(img, v)


def Anti_Solarize(img, min_val, max_val):  # [0, 256]
    v = random.uniform(min_val, max_val)
    threshold = int(v)
    lut = []
    for i in range(256):
        if i > threshold:
            lut.append(i)
        else:
            lut.append(255 - i)

    return PIL.ImageOps._lut(img, lut)


def SolarizeAdd(img, min_val, max_val):
    v = random.uniform(min_val, max_val)
    addition = random.uniform(0, 120)
    img_np = np.array(img).astype(int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, min_val, max_val):  # [4, 8]
    v = random.uniform(min_val, max_val)
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, min_val, max_val):  # [0.1,1.9]
    v = random.uniform(min_val, max_val)
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, min_val, max_val):  # [0.1,1.9]
    v = random.uniform(min_val, max_val)
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, min_val, max_val):  # [0.1,1.9]
    v = random.uniform(min_val, max_val)
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, min_val, max_val):  # [0.1,1.9]
    v = random.uniform(min_val, max_val)
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, min_val, max_val):
    v = random.uniform(min_val, max_val)
    if v <= 0.0:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, min_val, max_val):
    v = random.uniform(min_val, max_val)
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, min_val, max_val):
        v = random.uniform(min_val, max_val)
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, _, __):
    return img


def augment_list():
    l = [
        # (set_dark_pixels_to_zero, 0, 10),
        # (Identity, 0.0, 1.0),
        # (AutoContrast, 0, 1),
        # (Equalize, 0, 1),
        # (Invert, 0, 1),
        # (Rotate, 0, 90),
        # (Posterize, 0, 8),
        # (Anti_Solarize, 0, 100),
        # (Solarize, 0, 50),
        # (SolarizeAdd, 0, 50),
        # (Contrast, 0.6, 1.9),
        # (Brightness, 0.5, 1.9),
        (Sharpness, 0.4, 1.9),
        (ShearX, 0.0, 0.1),
        (ShearY, 0.0, 0.1),
        # (TranslateXabs, 0.0, 5),
        # (TranslateYabs, 0.0, 10),
    ]

    return l


class Lighting(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class LightingGray(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(1).normal_(0, self.alphastd)
        lighting = (
            self.eigvec[:, 0].type_as(img)
            .clone()
            .mul(alpha)
            .mul(self.eigval[0].view(1))
            .sum()
        )

        return img.add(lighting.view(1, 1, 1).expand_as(img))  # 返回扰动后的单通道灰度图


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


# class AbelAugment:
#     def __init__(self, n):
#         self.n = n
#         self.augment_list = augment_list()

#     def __call__(self, img):
#         ops = random.choices(self.augment_list, k=self.n)
#         img = set_dark_pixels_to_zero(img, 1, 30)
#         for op, min_val, max_val in ops:
#             img = op(img, min_val, max_val)

#         return img


class AbelAugment:
    def __init__(self, n):
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img):
        random.seed() 
        ops = random.choices(self.augment_list, k=self.n)
        if random.random() < 0.3:
            img = set_dark_pixels_to_zero(img, 1, random.randint(1,10))
        if random.random() < 0.1:
            return img
        for op, min_val, max_val in ops:
            img = op(img, min_val, max_val)

        return img


class Dark2zero:
    def __call__(self, img):
        img = set_dark_pixels_to_zero(img, 1, 10)
        return img
