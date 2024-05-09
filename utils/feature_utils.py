import numpy as np
import torch
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.collate import collate_features
from data.transforms import get_basic_data_transforms
from data.ThreeDimDataset import RawImgBag
from utils.file_utils import save_hdf5

def safe_list_to(data, device):
    """
    Moves data to device when data is either a torch.Tensor or a list/tuple/dict of tensors.
    e.g. [d.to(device) for d in data]

    Args:
    - data (torch.tensor, tuple, list, dict): The input data to put on a device.
    - device (torch.device): The device to move data do.

    Returns:
    - data (torch.tensor, tuple, list, dict): Data or each element of data on the device preserving the input structure
    """

    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple):
        return (d.to(device) for d in data)
    elif isinstance(data, list):
        return [d.to(device) for d in data]
    elif isinstance(data, dict):
        return {k: v.to(device) for (k, v) in data.keys()}
    else:
        raise RuntimeError("data should be a Tensor, tuple, list or dict, but"
                           " not {}".format(type(data)))

def adjust_batch_channels(model_channel_dim=3, batch=None):
    """
    Adjusts the data channel dimension to match with the network channel dimension

    Args:
    - model_channel_dim (int): channel dimension of the model (usually 3)
    - batch (torch.tensor): (Batch, channel, ...)

    Returns:
    - batch_new (torch.tensor): New data batch with adjusted channel dimension
    """

    if isinstance(batch, list):
        batch_channel_dim = batch[0].shape[1]
    else:
        batch_channel_dim = batch.shape[1]

    if model_channel_dim == batch_channel_dim:  # If channel dimension of data/architecture same, do nothing
        batch_new = batch
    elif model_channel_dim == 1 and batch_channel_dim == 3:  # Take average of data channel dim (e.g., RBG input & grayscale network)
        if isinstance(batch, list):
            batch_new = [torch.mean(batch_indiv, dim=1, keepdim=True) for batch_indiv in batch]
        else:
            batch_new = torch.mean(batch, dim=1, keepdim=True)
    elif model_channel_dim == 3 and batch_channel_dim == 1: # Copy data channel dim
        if isinstance(batch, list):
            rest_dim = [1] * len(batch[0].shape[2:])
            batch_new = [batch_indiv.repeat(1, 3, *rest_dim) for batch_indiv in batch]
        else:
            rest_dim = [1] * len(batch.shape[2:])
            batch_new = batch.repeat(1, 3, *rest_dim)
    else:
        raise NotImplementedError("Not implemented for channel {}, batch_channel {}".format(model_channel_dim,
                                                                                            batch_channel_dim))

    return batch_new


def extract_patch_features(dataset,
                           output_path,
                           model,
                           model_name=None,
                           batch_size=8,
                           channel=1,
                           leave=False,
                           device=None):

    """
    Extract features from 2D/3D patches and save them as h5 files

    Args:

    Returns:

    """
    collate_fn = collate_features

    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == "cuda" else {}

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        collate_fn=collate_fn,
                        drop_last=False,
                        **kwargs)

    pbar_extracting = tqdm(desc='Extracting Features (from HDF5)',
                           total=len(loader) * batch_size,
                           initial=0,
                           position=1,
                           leave=leave)

    s = time.time()
    mode = 'w'   # First create the hdf5 file
    for _, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            batch = adjust_batch_channels(channel, batch)
            batch = safe_list_to(batch, device)

            features = model(batch)

            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(output_path, asset_dict, mode=mode)
            mode = 'a'  # Once created, just append

            # Remove unnecessary memory
            del batch, features
            torch.cuda.empty_cache()

        pbar_extracting.update(batch_size)

    print("Total of {} features".format(len(dataset)))
    print("Took ", time.time() - s)


def extract_patch_raw(img_obj=None,
                      patch_level=0,
                      patch_size=(96, 96, 96),
                      step_size=(96, 96, 96),
                      step_size_z=None,
                      data_mode='CT',
                      slice_mode='all',
                      clip_min=0,
                      clip_max=1,
                      contour_fn='four_pt_easy',
                      augment=False,
                      **wsi_kwargs):

    trns = get_basic_data_transforms(augment=augment,
                                     data_mode=data_mode)

    roi_dataset = RawImgBag(img_object=img_obj,
                            patch_level=patch_level,
                            patch_size=patch_size,
                            slice_mode=slice_mode,
                            step_size=step_size,
                            contour_fn=contour_fn,
                            clip_min=clip_min,
                            clip_max=clip_max,
                            transforms=trns,
                            **wsi_kwargs)

    return roi_dataset


def extract_features_raw(roi_dataset,
                         feature_extractor=None,
                         batch_size=512,
                         channel=1,
                         device=None):
    kwargs = {'num_workers': 20, 'pin_memory': False} if device.type == "cuda" else {}
    dataloader = DataLoader(roi_dataset,
                            batch_size=batch_size,
                            sampler=torch.utils.data.SequentialSampler(roi_dataset),
                            collate_fn=collate_features,
                            **kwargs)

    print('total number of patches to process: ', len(roi_dataset))

    features_all = []
    coords_all = []

    for idx, (batch, coords) in enumerate(tqdm(dataloader)):
        batch = adjust_batch_channels(channel, batch)
        batch = batch.to(device)

        with torch.no_grad():
            features = feature_extractor(batch)  # (n_instances, instance_dim)

        features_all.append(features.cpu().numpy())
        coords_all.append(coords)

    features_all = np.concatenate(features_all)
    coords_all = np.concatenate(coords_all)

    asset_dict = {'features': features_all, 'coords': coords_all}
    return asset_dict

def extract_patch_features_raw(img_obj=None,
                               feature_extractor=None,
                               patch_level=0,
                               patch_size=96,
                               step_size=96,
                               batch_size=512,
                               data_mode='CT',
                               slice_mode='all',
                               channel=1,
                               clip_min=0,
                               clip_max=1,
                               area_thresh=0.5,
                               contour_fn='four_pt_easy',
                               augment=False,
                               device=None,
                               **wsi_kwargs):
    """
    Extract features from the raw image data.
    Useful for saving storage (patches are not stored)
    Primarily used for heatmap generation
    """

    trns = get_basic_data_transforms(augment=augment,
                                     data_mode=data_mode)

    roi_dataset = RawImgBag(img_object=img_obj,
                            patch_level=patch_level,
                            patch_size=patch_size,
                            slice_mode=slice_mode,
                            step_size=step_size,
                            area_thresh=area_thresh,
                            contour_fn=contour_fn,
                            clip_min=clip_min,
                            clip_max=clip_max,
                            transforms=trns,
                            **wsi_kwargs)

    kwargs = {'num_workers': 4, 'pin_memory': False} if device.type == "cuda" else {}
    dataloader = DataLoader(roi_dataset,
                            batch_size=batch_size,
                            sampler=torch.utils.data.SequentialSampler(roi_dataset),
                            collate_fn=collate_features,
                            **kwargs)

    print('total number of patches to process: ', len(roi_dataset))

    features_all = []
    coords_all = []

    for idx, (batch, coords) in enumerate(tqdm(dataloader)):
        batch = adjust_batch_channels(channel, batch)
        batch = batch.to(device)

        with torch.no_grad():
            features = feature_extractor(batch) # (n_instances, instance_dim)

        features_all.append(features.cpu().numpy())
        coords_all.append(coords)

    features_all = np.concatenate(features_all)
    coords_all = np.concatenate(coords_all)

    asset_dict = {'features': features_all, 'coords': coords_all}
    return asset_dict, img_obj
