import numpy as np
import random
import torch
from torchvision.transforms import functional as F


class ToFloat(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img.astype(np.float32)


class DVSTransform:
    def __init__(self, c, d, e, n):
        self.c = c
        self.d = d
        self.e = e
        self.n = n

    def __call__(self, img):
        # T C H W
        w = img.shape[-1]
        img = torch.Tensor(img.astype(np.float32))

        if random.random() > 0.5:
            img = F.hflip(img)

        # 1
        a = int(random.uniform(-self.c, self.c))
        b = int(random.uniform(-self.c, self.c))
        img = torch.roll(img, shifts=(a, b), dims=(1, 2))

        # 2
        mask = 0
        length = random.uniform(1, self.e)
        height = random.uniform(1, self.e)
        center_x = random.uniform(0, w)
        center_y = random.uniform(0, w)

        small_y = int(center_y - height / 2)
        big_y = int(center_y + height / 2)
        small_x = int(center_x - length / 2)
        big_x = int(center_x + length / 2)

        if small_y < 0:
            small_y = 0
        if small_x < 0:
            small_x = 0
        if big_y > w:
            big_y = w
        if big_x > w:
            big_x = w

        img[:, :, small_y:big_y, small_x:big_x] = mask

        return img
