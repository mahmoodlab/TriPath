"""
Contains the helper functions
"""

import numpy as np
from pydicom import dcmread
from tqdm import tqdm
import cv2
import glob
from PIL import Image
import xml.etree.ElementTree as ET


def read_img(dirpath, black_thresh=-1, resolution=1.0):
    """
    Load image files.
    Also removes slices with non-important information (simple thresholding) if a
    positive black threshold value is supplied.
    Returns
    =======
    img_arr: 3-D numpy array (uint16)
    ds: Dataset
    """

    if dirpath[-1] == '/':
        dirpath = dirpath[:-1]

    flist = sorted(glob.glob(dirpath + '/*[!t]'))
    ext = flist[0].split('.')[-1]

    # Read metadata file
    f_meta_path = glob.glob(dirpath + '/*.dat')
    if len(f_meta_path) > 0:
        f_meta_path = f_meta_path[0]
        info = read_metadata(f_meta_path)
        img_info = {
            'resolution': info['resolution'],
            'start_x': info['start_x'],
            'start_y': info['start_y'],
            'start_z': info['start_z']
        }
    else:
        img_info = {
            'resolution': 1,
            'start_x': 0,
            'start_y': 0,
            'start_z': 0
        }

    print("======================")
    print("Total of {} files in {}".format(len(flist), dirpath))

    img_arr = []
    for f in tqdm(flist):
        if ext == 'dcm':
            img = dcmread(f).pixel_array
            if resolution != 1.0:
                cur_size = np.shape(img)
                new_size = (round(cur_size[0] * resolution), round(cur_size[1] * resolution))
                img = Image.fromarray(img)
                img = np.array(img.resize(new_size))
        else:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if resolution != 1.0:
                new_size = (round(img.width * resolution), round(img.height * resolution))
                img = np.array(img.resize(new_size))
            else:
                img = np.array(img)

        img = adjust_channel(img)
        img_arr.append(img)

    if black_thresh >= 0:
        print("Filtering black slices started...")
        img_arr, z_levels = filter_black(img_arr, black_thresh)
        img_info['z_levels'] = z_levels
        print("Finished: z_start: {}, z_end: {}, total: {}".format(z_levels[0],
                                                                   z_levels[1],
                                                                   img_arr.shape[0]))
        print("==============================")
    else:
        img_info['z_levels'] = (0, len(img_arr))

    img_arr = np.stack(img_arr)

    # Now we must reduce the resolution (resize the image) in the z-axis via resampling
    if resolution != 1.0:
        new_arr = []
        prev_idx = 0
        for i in range(1, round(resolution * len(img_arr)) + 1):
            new_arr.append(img_arr[prev_idx:round(i / resolution)].mean(axis=0))
            prev_idx = round(i / resolution)
        img_arr = np.array(new_arr)

    return img_arr, img_info


def adjust_channel(img, verbose=False):
    """
    Adjust channel dimension 2-D images
    """
    img_shape_len = len(img.shape)

    if img_shape_len == 2:
        img_new = img[..., np.newaxis]
        msg = '\n Grayscale image => Adjusting channel dimension to 1'
    elif img_shape_len == 3:
        channel_dim = img.shape[-1]
        if channel_dim == 4:
            img_new = img[..., :3]
            msg = "\n Image with 4 channels (Assuming RGBA) => Adjusting channel dimension to 3"
        else:
            img_new = img
            msg = "\n Image with {} channels => Channel dimension not adjusted".format(channel_dim)
    else:
        raise NotImplementedError("Image shape of {} wrong".format(img_shape_len))

    if verbose:
        print(msg)

    return img_new

def read_metadata(file):
    """
    Read meta data from XML file
    """

    tree = ET.parse(file)
    root = tree.getroot()

    info = {'resolution': float(root.find('Spacing').get('X')),
            'start_x': float(root.find('Position').get('P1')),
            'start_y': float(root.find('Position').get('P2')),
            'start_z': float(root.find('Position').get('P3'))}

    return info


def clip_img(img, clip_min=20000, clip_max=36000):
    """
    Clip image.

    Inputs
    ======
    img: multi-dimensional numpy array
    clip: If True, clip image
    vmin: float. Anything below this value will be set to vmin
    vmax: float. Anything above this value will be set to vmax
    """
    assert clip_min < clip_max, "vmax {} should be larger than vmin {}".format(clip_max, clip_min)

    img_trns = np.copy(img)
    img_trns[img_trns < clip_min] = clip_min
    img_trns[img_trns > clip_max] = clip_max

    return img_trns


def clip_and_normalize_img(img, clip_min=25000, clip_max=55000):
    img_new = clip_img(img, clip_min=clip_min, clip_max=clip_max)
    img_new = (img_new - clip_min) / (clip_max - clip_min)

    return img_new

def clip_and_invert_img(img, clip_min=25000, clip_max=55000):
    """
    Clip and invert images (Useful for CycleGAN)
    """
    img_new = clip_img(img, clip_min=clip_min, clip_max=clip_max)
    img_new = clip_max - img_new

    return img_new


def identify_image_thresholds(img, clip_min=0, clip_max=1, thresh_mode='fixed'):
    """
    Identify upper threshold (1 %) and normalize image accordingly

    Args:
    - img (np.ndarray): 2D or 3D image input
    - clip_min (int): Lower threshold to clip image
    - clip_max (int): Upper threshold to clip image (only if thresh_mode=='fixed')
    - thresh_mode (str): 'fixed' or 'global'
        'fixed': Use given clip_max
        'global': Identify the upper threshold by sorting the voxel intensities and using top 1% value

    Returns:
    - upper_thresh (int): Upper threshold
    - lower_thresh (int): Lower threshold
    """

    print("\nThresholding with {} mode ".format(thresh_mode))

    if thresh_mode == 'fixed':
        upper_thresh = clip_max
        lower_thresh = clip_min

    elif thresh_mode == 'global': # Automatically determine maximum threshold from entire image (Top 1%)
        img_temp = img.flatten()
        img_temp.sort()
        img_filtered = img_temp[img_temp > clip_min]

        clip_max_adaptive = img_filtered[-len(img_temp) // 100]
        clip_min_adaptive = clip_min
        print("Adaptive min, max threshold: {}, {}".format(clip_min_adaptive, clip_max_adaptive))

        upper_thresh = clip_max_adaptive
        lower_thresh = clip_min_adaptive

    else:    # Normalize no matter what
        raise NotImplementedError("Not yet implemented")

    return upper_thresh, lower_thresh


def convert_RGB(img):
    """
    Transform image to RGB, in order to maintain grayscale
    Basically copies the same information to all channels
    """
    if img.shape[-1] == 1:
        img_new = np.tile(img, (1, 1, 3))
    else:   # If already 3-channel data, don't do anything
        img_new = img

    return img_new


def process_raw_img(img, vmin=20000, vmax=36000):
    """
    Performs 3 steps of processing the image: 1) clip 2) normalize 3) RGB conversion
    """
    img_new = clip_and_normalize_img(img, clip_min=vmin, clip_max=vmax) * 255
    img_new = img_new.astype(np.uint8)
    img_new = convert_RGB(img_new)

    return img_new


def filter_black(img_arr, black_thresh):
    """
    Remove slices that are empty (black), useful for 3D data
    This is required since significant portion of the scans are just empty

    Inputs
    ======
    black_thresh: Threshold for the mean of the slice to be below, to be considered a black slice

    Returns
    =======
    z_levels: Tuple of (z_start, z_end)
    """

    img_arr_new = []
    z_end = None

    for stack_idx in range(len(img_arr)):
        img = img_arr[stack_idx]

        if np.mean(img) > black_thresh:
            # If stack is not black for the first time (Actual Paraffin/Tissue)
            if len(img_arr_new) == 0:
                z_start = stack_idx

            img_arr_new.append(img)
        else:
            # If stack is black again
            if len(img_arr_new) != 0:
                z_end = stack_idx
                break

    if z_end is None:
        z_end = len(img_arr)

    assert len(img_arr) > 0, "Img arr has no elements - Black thresh {}".format(black_thresh)

    img_arr_new = np.stack(img_arr_new)
    z_levels = (z_start, z_end)

    return img_arr_new, z_levels


def get_slice_spacing(resolution, gap):
    assert gap > resolution, "Gap needs to be larger than the resolution"
    return gap // resolution + 1
