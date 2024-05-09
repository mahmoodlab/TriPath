import torch
import numpy as np


def collate_features(batch):
    """
    Custom collate function for item = (img, coords)
    """
    img = torch.stack([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])

    return [img, coords]

def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]


def collate_features_multiple_views(batch):
    """
    Custom collate function for item = (list of imgs, coords)

    batch consists of list of (img_view_1, ..., img_view_n)
    """

    img_list = []
    numOfviews = len(batch[0][0])
    for idx in numOfviews:
        img_list.append(torch.vstack([item[0][idx] for item in batch], dim=0))

    return img_list
