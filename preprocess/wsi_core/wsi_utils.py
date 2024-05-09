"""
Contains the helper functions for creating h5 files
"""

import h5py
import numpy as np
import os
import pandas as pd

##########################
# 3D preprocessing utils #
##########################
def savePatchIter_bag_hdf5_3D(patch):
    """
    Save patch iteratively to hdf5.
    Assumes initialize_hdf5_bag_3D has been run before
    """
    x = patch['x']
    y = patch['y']
    z = patch['z']

    name = patch['name']
    img_patch = patch['patch']
    save_path = patch['save_path']

    # img_patch: list of (w, h)
    img_patch = img_patch[np.newaxis, ...]  # (z, w, h) -> (-1, z, w, h)
    img_shape = img_patch.shape  # (-1, z, w, h)

    file_path = os.path.join(save_path, name) + '_patches.h5'
    file = h5py.File(file_path, "a")

    dset = file['imgs']
    dset.resize(len(dset) + img_shape[0], axis=0)
    dset[-img_shape[0]:] = img_patch

    coord_dset = file['coords']
    coord_dset.resize(len(coord_dset) + img_shape[0], axis=0)
    coord_dset[-img_shape[0]:] = (z, x, y)

    file.close()


def initialize_hdf5_bag_3D(patch, z_level_start):
    """
    Initialize hdf5. Contains three datasets - imgs, coords, z_level

    imgs: holder for patches (numOfpatches, z , w, h)
    coords: Holder for top-left corner of each patch (Tuple of three element)
    z_level: indicates the z-level of the corresponding patch
    """
    x = patch['x']
    y = patch['y']
    z = patch['z']

    name = patch['name']
    patch_level = patch['patch_level']
    downsample = patch['downsample']
    downsampled_level_dim = patch['downsampled_level_dim']
    level_dim = patch['level_dim']
    img_patch = patch['patch']
    save_path = patch['save_path']
    resolution = patch['resolution']

    file_path = os.path.join(save_path, name) + '_patches.h5'
    file = h5py.File(file_path, "w")

    # img_patch: list of (w, h)
    img_patch = img_patch[np.newaxis, ...]  # (z, w, h) -> (-1, z, w, h)

    dtype = img_patch.dtype

    # Initialize a resizable dataset to hold the output
    img_shape = img_patch.shape # (-1, z, w, h)

    # maximum dimensions up to which dataset maybe resized (None means unlimited)
    # First dim: number of patches in each slice
    maxshape = (None,) + img_shape[1:]
    dset = file.create_dataset('imgs',
                               shape=img_shape,
                               maxshape=maxshape,
                               dtype=dtype)

    dset[:] = img_patch

    # Attributes
    dset.attrs['patch_level'] = patch_level
    dset.attrs['wsi_name'] = name
    dset.attrs['downsample'] = downsample
    dset.attrs['level_dim'] = level_dim
    dset.attrs['downsampled_level_dim'] = downsampled_level_dim
    dset.attrs['resolution'] = resolution

    coord_dset = file.create_dataset('coords',
                                     shape=(1, 3),
                                     maxshape=(None, 3),
                                     dtype=np.int32)

    coord_dset[:] = (z, x, y)
    coord_dset.attrs['z_level_start'] = z_level_start

    file.close()


###########################
# H&E preprocessing utils #
###########################
def savePatchIter_bag_hdf5(patch):
    """
    Save patch iteratively to hdf5
    """
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path = tuple(
        patch.values())
    img_patch = np.array(img_patch)[np.newaxis, ...]
    img_shape = img_patch.shape

    file_path = os.path.join(save_path, name) + '.h5'
    file = h5py.File(file_path, "a")

    dset = file['imgs']
    dset.resize(len(dset) + img_shape[0], axis=0)
    dset[-img_shape[0]:] = img_patch

    if 'coords' in file:
        coord_dset = file['coords']
        coord_dset.resize(len(coord_dset) + img_shape[0], axis=0)
        coord_dset[-img_shape[0]:] = (x, y)

    file.close()


def initialize_hdf5_bag(first_patch, save_coord=False):
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path = tuple(
        first_patch.values())
    file_path = os.path.join(save_path, name) + '.h5'
    file = h5py.File(file_path, "w")
    img_patch = np.array(img_patch)[np.newaxis, ...]
    dtype = img_patch.dtype

    # Initialize a resizable dataset to hold the output
    img_shape = img_patch.shape
    maxshape = (None,) + img_shape[1:]  # maximum dimensions up to which dataset maybe resized (None means unlimited)
    dset = file.create_dataset('imgs',
                               shape=img_shape, maxshape=maxshape, chunks=img_shape, dtype=dtype)

    dset[:] = img_patch
    dset.attrs['patch_level'] = patch_level
    dset.attrs['wsi_name'] = name
    dset.attrs['downsample'] = downsample
    dset.attrs['level_dim'] = level_dim
    dset.attrs['downsampled_level_dim'] = downsampled_level_dim

    if save_coord:
        coord_dset = file.create_dataset('coords', shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32)
        coord_dset[:] = (x, y)

    file.close()
    return file_path


def screen_coords(scores, coords, top_left, bot_right):
    """
    Filter coordinates/scores within the bounding box
    """
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords


def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100
    return scores


def initialize_df(slides, seg_params, filter_params, vis_params, patch_params, img_params):
    """
    Initialize dataframe with relevant parameters
    """
    total = len(slides)
    df = pd.DataFrame({'slide_id': slides, 'process': np.full((total), 1, dtype=np.uint8),
        'status': np.full((total), 'tbp'),

        ## seg params
        # Level at which to segment (Not yet used for 3D)
        'seg_level': np.full((total), int(seg_params['seg_level']), dtype=np.int8),
        # Threshold for binarization
        'sthresh': np.full((total), int(seg_params['sthresh']), dtype=np.uint8),
        # Median filtering raidus (Too low of a value will result in squiggly contours)
        'mthresh': np.full((total), int(seg_params['mthresh']), dtype=np.uint8),
        'close': np.full((total), int(seg_params['close']), dtype=np.uint32),
        'use_otsu': np.full((total), bool(seg_params['use_otsu']), dtype=bool),

        ## filter params
        # threshold for area of tissue (multiplier - will be multiplied by reference patch size)
        'a_t': np.full((total), int(filter_params['a_t']), dtype=np.uint32),
        # threshold for area of hole (multiplier - will be multiplied by reference patch size)
        'a_h': np.full((total), int(filter_params['a_h']), dtype=np.uint32),
        # maximum number of holes
        'max_n_holes': np.full((total), int(filter_params['max_n_holes']), dtype=np.uint32),

        # vis params
        'vis_level': np.full((total), int(vis_params['vis_level']), dtype=np.int8),    # Not used
        'line_thickness': np.full((total), int(vis_params['line_thickness']), dtype=np.uint32),

        # patching params
        'use_padding': np.full((total), bool(patch_params['use_padding']), dtype=bool),
        'contour_fn': np.full((total), patch_params['contour_fn']),

        # img_params
        'black_thresh': np.full((total), img_params['black_thresh'], dtype=np.float32),
        'clip_min': np.full((total), img_params['clip_min'], dtype=np.uint32),
        'clip_max': np.full((total), img_params['clip_max'], dtype=np.uint32)
    })

    return df

# def get_best_HE_patch_level(wsi, ref_res):
#     """
#     Given the resolution for the reference modality (i.e., CT), find the best level to process WSI
#
#     Inputs
#     ======
#     wsi: OpenSlide object
#     ref_res: float
#         Resolution for the reference modality
#     """
#     factor = ref_res / float(wsi.properties['openslide.mpp-x'])
#     patch_level = wsi.get_best_level_for_downsample(factor)
#
#     return patch_level
