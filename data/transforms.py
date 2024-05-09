"""
Transform functions
"""

import torch
import numpy as np
from torchvision import transforms
from models.feature_extractor.Swin3D import Swin3D_S_Weights
from models.feature_extractor.Swin2D import Swin_V2_S_Weights


def geometric_transforms(mode='3D'):
    if mode == '3D':
        # data_transforms = transforms.Compose([
        #     RandomFlip3d(p=0.5),
        #     RandomRotation3d(p=0.5)
        # ])
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])
    elif mode == '2D':
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])
    else:
        raise NotImplementedError("Not implemented for geometric transform of {}".format(mode))

    return data_transforms


def intensity_transforms(mode='CT'):
    data_transforms = transforms.Compose([
        RandomGammaVolume(p=0.5),
        RandomGaussianNoise(p=0.5)
    ])

    return data_transforms


def get_basic_data_transforms(augment=True, patch_mode='3D', data_mode='CT', invert=True):
    if augment:
        data_transforms = transforms.Compose([
            Invert(invert),
            ToTensorMultiDim(),
            geometric_transforms(patch_mode),
            intensity_transforms(data_mode),
            Normalize(data_mode)
        ])
    else:
        data_transforms = transforms.Compose([
            Invert(invert),
            ToTensorMultiDim(),
            Normalize(data_mode)
        ])

    return data_transforms


# def get_video_transforms():
#     """
#     Get transforms for video
#     """
#     print("VIDEO")
#     data_transforms = transforms.Compose([ToTensorVideo(),
#                                           Swin3D_S_Weights.DEFAULT.transforms()])
#
#     return data_transforms
#
# def get_img_transforms():
#     """
#     Get transforms for 2D for ViT (torchvision models)
#     """
#     print("IMG")
#     data_transforms = transforms.Compose([ToTensorMultiDim(),
#                                           Swin_V2_S_Weights.DEFAULT.transforms()])
#     return data_transforms

###################
# Transformations #
###################
class Invert(object):
    """
    Inverts the channel intensity
    """

    def __init__(self, invert):
        super().__init__()
        self.invert = invert

    def __call__(self, img):
        if self.invert:
            img = 1 - img

        return img

class ToTensorMultiDim(object):
    """
    Converts numpy image array to tensor and permute dimensions
    """
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        img_len = len(img.shape)
        img = torch.from_numpy(img)

        if img_len == 3:
            # Convert from WHC to CWH
            img = img.permute((2, 0, 1)).contiguous()
        elif img_len == 4:
            # Convert from ZWHC to CZWH
            img = img.permute((3, 0, 1, 2)).contiguous()
        else:
            raise NotImplementedError("Image shape {} not accepted".format(img_len))

        return img

class ToTensorVideo(object):
    """
    Converts numpy image array to tensor and permute dimensions
    """
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        img_len = len(img.shape)
        img = torch.from_numpy(img)

        # Convert from ZWHC to ZCWH
        img = img.permute((0, 3, 1, 2)).contiguous()

        return img

class Normalize(object):
    def __init__(self, mode='CT'):
        super().__init__()
        if mode == 'CT':
            self.mean = (0.45, 0.45, 0.45)
            self.std = (0.225, 0.225, 0.225)
        elif mode == 'HE':
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        else:
            raise NotImplementedError("Not implemented for normalization {}".format(mode))

    def __call__(self, img):
        for channel in range(img.shape[0]):
            if torch.std(img[channel]) == 0:
                img[channel] = (img[channel] - torch.mean(img[channel]))
            else:
                img[channel] = (img[channel] - torch.mean(img[channel])) / torch.std(img[channel])

        return img

class Identity(object):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return img

# Affine transforms
class RandomRotation3d(object):
    """
    Random rotation of input patch along each of the three axes.
    Possible rotations: 0, 90, 180, 270 degrees
    """

    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, img):
        rotated = img
        x = len(img.size())-3

        for i in range(x, x+3):
            prob = np.random.rand()
            j = (i + 1)
            if j >= (x+3):
                j -= 3

            if prob < self.p:
                rot_prob = np.random.rand()
                # 90 degree rotation
                if rot_prob < 0.25:
                    rotated = rotated.transpose(i, j).flip(i)
                # 180 degree rotation
                elif rot_prob < 0.5:
                    rotated = rotated.flip(i).flip(j)
                # 270 degree rotation
                elif rot_prob < 0.75:
                    rotated = rotated.transpose(i, j).flip(j)

        return rotated


class RandomFlip3d(object):
    """
    Random flip of input patch along each of the three axes.
    """

    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, img):
        flipped = img

        x = len(img.size())-3
        for i in range(x, x+3):
            prob = np.random.rand()
            if prob < self.p:
                flipped = flipped.flip(i)

        return flipped


# Intensity transforms
class RandomGammaVolume(object):
    """
    Random intensity (gamma) agumentation

    img_new = a + b * img^g
    a ~ Uniform[-alpha, alpha]
    b ~ Uniform[1-beta, 1+beta]
    g ~ Uniform[1-gamma, 1+gamma]
    """
    def __init__(self, p=0.5, alpha=0.05, beta=0.1, gamma=0.3):
        self.p = p
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, img):
        prob = np.random.rand()
        img_new = img

        if prob < self.p:
            alpha = torch.rand(1) * self.alpha * 2 - self.alpha
            beta = 1 + torch.rand(1) * self.beta * 2 - self.beta
            gamma = 1 + torch.rand(1) * self.gamma * 2 - self.gamma

            img_new = alpha + beta * torch.pow(img, gamma)
            img_new = torch.clamp(img_new, min=0, max=1)

        return img_new


class RandomGaussianNoise(object):
    """
    Random Gaussian noise augmentation

    img_new = img + sigma
    """
    def __init__(self, p=0.5, sigmas=[0.01, 0.05]):
        self.p = p
        self.sigma_lb, self.sigma_ub = sigmas

    def __call__(self, img):
        prob = np.random.rand()
        img_new = img

        sigma = self.sigma_lb + torch.rand(1) * (self.sigma_ub - self.sigma_lb)

        if prob < self.p:
            img_new = img + sigma * torch.randn(img.shape)
            img_new = torch.clamp(img_new, min=0, max=1)

        return img_new


class Cutout(object):
    def __init__(self, p=0.25, prop=0.2):
        self.p = p
        self.prop = prop

    def __call__(self, img):
        prob = np.random.rand()
        cutout_len = int(self.prop * img.shape[0])
        img_new = img

        if prob < self.p:
            x = np.random.randint(0, img.shape[0] - cutout_len)
            y = np.random.randint(0, img.shape[1] - cutout_len)
            z = np.random.randint(0, img.shape[2] - cutout_len)

            img_new[:, x: x+cutout_len, y: y+cutout_len, z: z+cutout_len] = 0

        return img_new


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform[i](x) for i in range(self.n_views)]
