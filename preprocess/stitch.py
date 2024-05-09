"""
Functions for stitching patches together
"""
import os
import h5py
import numpy as np
from tqdm import tqdm
import argparse

import math
import cv2
from PIL import Image
import PIL
from glob import glob

from scipy.ndimage import gaussian_filter1d
from wsi_core.img_utils import clip_and_normalize_img, convert_RGB

PIL.Image.MAX_IMAGE_PIXELS = 9000000000


def format_str(lev):
    """
    Ensure the filname for slice names end with appropriate number of zeros prepended ('0012' instead of '12')
    This allows more accurate sorting
    """
    lev_str = str(lev)

    assert len(lev_str) <= 4, "levels must be < 1e4"
    prefix = 4 - len(lev_str)

    lev_str = '0' * prefix + lev_str
    return lev_str


def StitchPatches3D(hdf5_file_path,
                    downscale=16,
                    draw_grid=True,
                    bg_color='black',
                    vmin=0,
                    vmax=1):
    """
    Stitch the patches for all z levels

    Returns
    =======
    heatmap_list: list of heatmaps
    z_list_abs: list of z levels, adjusted for the absolute z levels
    """
    file = h5py.File(hdf5_file_path, 'r')
    print("Loading patches...")
    dset = file['imgs']

    patch_dim = 2 if len(dset.shape) == 4 else 3
    if patch_dim == 2:    # 2D
        sub_levels = 1
        patch_size = (dset.shape[1], dset.shape[2])
    else:  # 3D. dset is (numOfpatches, Z, H, W, C)
        sub_levels = dset.shape[1]
        patch_size = (dset.shape[2], dset.shape[3])

    coords = file['coords'][:]
    z_list = coords[:, 0]
    z_unique_list = np.unique(z_list)   # List of unique z levels

    heatmap_list = []
    z_list_abs = []

    if 'downsampled_level_dim' in dset.attrs.keys():
        w, h = dset.attrs['downsampled_level_dim']
    else:
        w, h = dset.attrs['level_dim']
    canvas_size = (w, h)

    print("Stitching the patches...")

    # Empty canvas (filler) - Required not to throw off subsequent visualizations
    for z in range(z_unique_list[0]):
        heatmap = StitchPatches(dset,
                                coords,
                                downscale,
                                canvas_size=canvas_size,
                                patch_size=patch_size,
                                indices=None,
                                draw_grid=draw_grid,
                                bg_color=bg_color,
                                vmin=vmin,
                                vmax=vmax)

        heatmap_list.append(heatmap)
        z_list_abs.append(z)

    # Canvas with heatmaps
    for z in tqdm(z_unique_list):
        # Identify indices corresponding to desired z-level
        indices = np.flatnonzero(z_list == z)

        for sub_level in range(sub_levels):

            if patch_dim == 2:
                indices_refined = indices
            elif patch_dim == 3:
                indices_refined = [(idx, sub_level) for idx in indices]
            else:
                raise NotImplementedError

            heatmap = StitchPatches(dset,
                                    coords,
                                    downscale,
                                    canvas_size=canvas_size,
                                    patch_size=patch_size,
                                    indices=indices_refined,
                                    draw_grid=draw_grid,
                                    bg_color=bg_color,
                                    vmin=vmin,
                                    vmax=vmax)

            heatmap_list.append(heatmap)

            z_abs = z + sub_level
            z_list_abs.append(z_abs)

    file.close()

    return heatmap_list, z_list_abs


def StitchPatches(patch_dset,
                  coords,
                  downscale=16,
                  canvas_size=(100, 100),
                  patch_size=(96, 96),
                  draw_grid=True,
                  bg_color='black',
                  alpha=-1,
                  indices=None,
                  vmin=0,
                  vmax=1):
    """
    Wrapper for stitching patches. Downscale to desired level and then stitch the patches

    Inputs
    ======
    patch_dset: array of patches
        if '2D', (numOfpatches, w, h, c)
        if '3D', (numOfpatches, z, w, h, c)
    """

    w, h = canvas_size

    w = w // downscale
    h = h // downscale
    coords = (coords / downscale).astype(np.int32)

    downscaled_shape = (patch_size[1] // downscale, patch_size[0] // downscale)

    if w*h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)

    # Canvas for drawing heatmap
    # if alpha < 0 or alpha == -1:
    #     heatmap = Image.new(size=(w, h), mode="RGB", color=bg_color)
    # else:
    #     heatmap = Image.new(size=(w, h), mode="RGBA", color=bg_color + (int(255 * alpha),))

    heatmap = np.zeros((h, w, 3))
    heatmap_mask = np.zeros((h, w), dtype=np.int8)

    if indices is not None:
        patch_list = []
        coords_list = []

        # If 3D patching, indices are list of tuples (patch_idx, sub_level_idx)
        # If 2D patching, indices is a list of patch indices
        for item in indices:
            patch_list.append(patch_dset[item])

            if np.isscalar(item):
                coords_list.append(coords[item])
            else:
                coords_list.append(coords[item[0]])

        heatmap, heatmap_mask = DrawMap(heatmap,
                                        heatmap_mask,
                                        patch_list,
                                        coords_list,
                                        downscaled_shape,
                                        draw_grid=draw_grid,
                                        clip_min=vmin,
                                        clip_max=vmax)

    heatmap_mask = 1 - heatmap_mask

    if bg_color == 'white':
        bg_rgb = 255
    elif bg_color == 'gray':
        bg_rgb = 230
    elif bg_color == 'black':
        bg_rgb = 0

    heatmap[np.where(heatmap_mask == 1)] = bg_rgb

    return heatmap.astype(np.uint8)


def create_2D_kernel(patch_size, sigma=128):
    """
    Creates 2D Gaussian kernel for overlap & adding the patches
    """
    x, y = np.meshgrid(np.linspace(-patch_size[0]//2, patch_size[0]//2 - 1, patch_size[0]),
                       np.linspace(-patch_size[1]//2, patch_size[1]//2 - 1, patch_size[1]))

    kernel = np.exp(- (x * x + y * y) / (2 * sigma ** 2))

    return np.expand_dims(kernel, axis=-1)


def DrawGrid(img, coord, shape, thickness=1, color=(0, 0, 255)):
    """
    Draw rectangular patch grids on the stitched canvas
    """
    cv2.rectangle(img,
                  tuple(np.maximum([0, 0], coord-thickness//2)),
                  tuple(coord - thickness//2 + np.array(shape)),
                  color,
                  thickness=thickness)
    return img


def DrawMap(canvas,
            canvas_mask,
            patch_list,
            coords,
            patch_size,
            verbose=0,
            draw_grid=True,
            clip_min=0,
            clip_max=1,
            eps=1e-8):
    """
    Sitch 2D patches for 2D grayscale/RGB.
    This can also take care of overlapping patches.
    For each pixel, overlapping patches will be added with Gussian kernel weighting.

    Inputs
    ======
    patch_list: list of patches (w, h)
    coords: list of tuples (z, x, y)
    """
    numOfpatches = len(patch_list)

    # Gaussian weighting scheme
    gaussian_kernel = create_2D_kernel(patch_size)
    weights = np.ones((canvas.shape[0], canvas.shape[1], 1)) * eps

    if verbose > 0:
        twenty_percent_chunk = math.ceil(numOfpatches * 0.2)

    for patch_id in range(numOfpatches):
        if verbose > 0:
            if patch_id % twenty_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(patch_id, numOfpatches))

        patch = patch_list[patch_id]
        coord = coords[patch_id][1:]  # coord[0] is z

        patch = clip_and_normalize_img(patch, clip_min=clip_min, clip_max=clip_max) * 255
        patch = convert_RGB(patch)

        patch = cv2.resize(patch, patch_size)

        x, y = coord
        canvas_crop_shape = canvas[y: y + patch_size[1], x: x + patch_size[0], :3].shape[:2]
        patch_weighted = gaussian_kernel * patch

        canvas[y: y + patch_size[1], x: x + patch_size[0], :3] += patch_weighted[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        weights[y: y + patch_size[1], x: x + patch_size[0]] += gaussian_kernel[:canvas_crop_shape[0], :canvas_crop_shape[1]]
        canvas_mask[y: y + patch_size[1], x: x + patch_size[0]] = 1

        if draw_grid:
            # Draw grid to show patch boundaries
            DrawGrid(canvas, coord, patch_size)

    canvas = canvas / weights   # Normalize with weights
    canvas = canvas.astype(np.uint8)

    return canvas, canvas_mask


if __name__ == "__main__":
    """
    For H&E stitching, set vmin=0, vmax=255. Otherwise, for CT & OTLS, use appropriate thresholding
    """
    # Arguments
    parser = argparse.ArgumentParser(description='Stitch')
    parser.add_argument('--h5_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--downscale', type=int, default=1)
    parser.add_argument('--draw_grid', action='store_true', default=False)
    parser.add_argument('--bg_color', type=str, default='gray', choices=['black', 'white', 'gray'])
    parser.add_argument('--vmin', type=int, default=0)
    parser.add_argument('--vmax', type=int, default=1)
    parser.add_argument('--smooth', action='store_true', default=False)

    args = parser.parse_args()

    save_path = args.save_path if args.save_path is not None else args.h5_path

    flist = glob(args.h5_path + '/*.h5')
    flist.sort()

    print("\nStitching together patches...")
    print("Saving to {} with clip_min: {} clip_max: {}".format(save_path, args.vmin, args.vmax))

    for f in flist:
        subj = os.path.basename(f).split('_')[0]
        fpath = os.path.join(save_path, subj)
        print("\nInitiated for {}...".format(subj))

        if os.path.exists(fpath):
            print("Directory for {} already exists! Skipping...".format(subj))
            continue
        else:
            os.makedirs(os.path.join(save_path, subj), exist_ok=True)

            heatmap_list, z_list = StitchPatches3D(f,
                                                   downscale=args.downscale,
                                                   draw_grid=args.draw_grid,
                                                   bg_color=args.bg_color,
                                                   vmin=args.vmin,
                                                   vmax=args.vmax)

            heatmap_list = np.stack(heatmap_list)

            if args.smooth:
                print("Smoothing...")
                heatmap_list = gaussian_filter1d(heatmap_list, sigma=0.5, axis=0)

            print("Saving...")
            for heatmap, z_level in zip(heatmap_list, z_list):
                z_level_str = format_str(z_level)
                stitch_path = os.path.join(os.path.join(save_path, subj), '{}_stitched_zlevel_{}.png'.format(subj, z_level_str))
                Image.fromarray(heatmap).save(stitch_path)