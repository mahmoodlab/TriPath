"""
DONE

Main script for 3D image processing
"""

import sys
sys.path.append('..')
from stitch import StitchPatches3D

# internal imports
from wsi_core.SerialTwoDimImage import SerialTwoDimImage
from wsi_core.ThreeDimImage import ThreeDimImage
from wsi_core.wsi_utils import initialize_df
from wsi_core.img_utils import identify_image_thresholds
from utils.contour_utils import get_contour_check_fn
from PIL import Image

# other imports
import os
import time
import argparse
import pandas as pd

def setup(args):
    patch_size_z = args.patch_size if args.patch_size_z is None else args.patch_size_z
    step_z = args.step_size if args.step_z is None else args.step_z
    exp_dir = os.path.join(args.save_dir, 'patch_{}-{}_step_{}-{}_{}_{}_{}'.format(args.patch_size,
                                                                                   patch_size_z,
                                                                                   args.step_size,
                                                                                   step_z,
                                                                                   args.patch_mode,
                                                                                   args.slice_mode,
                                                                                   args.thresh_mode))

    patch_save_dir = os.path.join(exp_dir, 'patches')
    mask_save_dir = os.path.join(exp_dir, 'masks')
    stitch_save_dir = os.path.join(exp_dir, 'stitches')

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)
    else:
        process_list = None

    print('source: ', args.source)
    print('patch_save_dir: ', patch_save_dir)
    print('mask_save_dir: ', mask_save_dir)
    print('stitch_save_dir: ', stitch_save_dir)

    directories = {'source': args.source,
                'save_dir': args.save_dir,
                'exp_dir': exp_dir,
                'patch_save_dir': patch_save_dir,
                'mask_save_dir' : mask_save_dir,
                'stitch_save_dir': stitch_save_dir}

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    # Default parameters
    img_params = {'clip_min': args.clip_min, 'clip_max': args.clip_max, 'black_thresh': args.black_thresh}
    seg_params = {'seg_level': 0, 'sthresh': args.sthresh, 'mthresh': args.mthresh, 'close': 4, 'use_otsu': False} # No seg level required
    filter_params = {'a_t': args.a_t, 'a_h': args.a_h, 'max_n_holes': 8}
    vis_params = {'vis_level': 0, 'line_thickness': 8}
    patch_params = {'use_padding': True,
                    'contour_fn': args.contour_fn,
                    'area_thresh': args.area_thresh,
                    'area_check': True,
                    'contour_check': True}

    parameters = {'seg_params': seg_params,
                'filter_params': filter_params,
                'patch_params': patch_params,
                'vis_params': vis_params,
                'img_params': img_params}
    print("=====================Parameters")
    print(parameters)

    return directories, parameters, process_list

###################
# Wrapper functions
###################
def stitching(file_path, downscale=4, vmin=0, vmax=1):
    """
    Stitch the segmented & patched patches together
    """
    start = time.time()

    # Patch
    heatmap, z_list_abs = StitchPatches3D(file_path,
                                          downscale=downscale,
                                          draw_grid=True,
                                          vmin=vmin,
                                          vmax=vmax)

    total_time = time.time() - start

    return heatmap, z_list_abs, total_time


def segment(ThreeDim_object, seg_params, filter_params):
    """
    Segment tissue & holes

    Returns:
    - seg_success (boolean): True if segmentation was successful
    """
    ### Start Seg Timer
    start_time = time.time()

    # Segment
    seg_success = ThreeDim_object.segmentTissueSerial(**seg_params,
                                                        filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time

    return ThreeDim_object, seg_time_elapsed, seg_success


def patching(ThreeDim_object,
             patch_params,
             seg_params,
             patch_level,
             patch_size=96,
             patch_size_z=None,
             step_size=96,
             step_size_z=None,
             save_path='.',
             mode='single',
             verbose=False):
    """
    Create patches from the segmented volume

    Inputs:
    - seg_params (dict): Dictionary of segmentation parameters. Required for contour-based patch filtering
    """

    # Start Patch Timer
    start_time = time.time()

    # Patch
    cont_check_fn = get_contour_check_fn(contour_fn=patch_params['contour_fn'],
                                         patch_size=patch_size,
                                         step_size=step_size)

    file_path = ThreeDim_object.process_contours(**patch_params,
                                                 save_path=save_path,
                                                 patch_level=patch_level,
                                                 patch_size=patch_size,
                                                 patch_size_z=patch_size_z,
                                                 step_size=step_size,
                                                 step_size_z=step_size_z,
                                                 seg_params=seg_params,
                                                 cont_check_fn=cont_check_fn,
                                                 save_patches=True,
                                                 mode=mode,
                                                 verbose=verbose)


    # Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch_3D(source, save_dir, exp_dir, patch_save_dir, mask_save_dir, stitch_save_dir,
                     patch_size=96,
                     patch_size_z=None,
                     step_size=96,
                     step_size_z=None,
                     depth=None,
                     seg_params={'seg_level': 0, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False},
                     filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':10},
                     vis_params = {'vis_level': 0, 'line_thickness': 10},
                     patch_params = {'use_padding': True, 'contour_fn': 'four_pt_easy'},
                     img_params={'clip_min': 0, 'clip_max': 1, 'black_thresh': 0},
                     patch_level=0,
                     down_ratio=4,
                     seg=False,
                     save_mask=True,
                     stitch=False,
                     patch=False,
                     patch_mode='2D',
                     slice_mode='single',
                     verbose=False,
                     save_step=10,
                     thresh_mode='fixed',
                     process_list = None):
    """
    The main function for data preprocessing, comprised of the following steps.
    1. Initialize Three-dim object and load raw volumetric image
    2. Segment the raw volumetric image
    3. Patch the segmented volume
    4. (Optional) Stitch back the patches for visual verification
    """

    scans = sorted([folder for folder in os.listdir(source) if os.path.isdir(os.path.join(source, folder))])

    if process_list is None:
        df = initialize_df(scans,
                           seg_params,
                           filter_params,
                           vis_params,
                           patch_params,
                           img_params)
    else:
        df = pd.read_csv(process_list)

    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)
    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        # Update csv
        df.to_csv(os.path.join(exp_dir, 'process_list_seg.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i+1, total))
        print('processing {}'.format(slide))

        if patch:
            patch_path = os.path.join(patch_save_dir, '{}_patches.h5'.format(slide))
            if os.path.isfile(patch_path):
                print("{} already exists! Skipping ..".format(patch_path))
                continue

        df.loc[idx, 'process'] = 0

        # Load params
        current_vis_params = {}
        current_filter_params = {}
        current_seg_params = {}
        current_patch_params = {}
        current_img_params = {}

        for key in vis_params.keys():
            current_vis_params.update({key: df.loc[idx, key] if key in df.columns else vis_params[key]})

        for key in filter_params.keys():
            current_filter_params.update({key: df.loc[idx, key] if key in df.columns else filter_params[key]})

        for key in seg_params.keys():
            current_seg_params.update({key: df.loc[idx, key] if key in df.columns else seg_params[key]})

        for key in patch_params.keys():
            current_patch_params.update({key: df.loc[idx, key] if key in df.columns else patch_params[key]})

        for key in img_params.keys():
            current_img_params.update({key: df.loc[idx, key] if key in df.columns else img_params[key]})

        ###################################
        # Initialize Three-Dim Image object
        ###################################
        full_path = os.path.join(source, slide)

        if not os.path.exists(full_path):
            print(full_path)
            print("Path for " + slide + " doesn't exist! Skipping to next slide ...")
            continue

        print(current_img_params)
        print(current_seg_params)
        if patch_mode == '2D':
            ThreeDim_object = SerialTwoDimImage(path=full_path,
                                                depth=depth,
                                                **current_img_params,
                                                **current_seg_params)
        elif patch_mode == '3D':
            ThreeDim_object = ThreeDimImage(path=full_path,
                                            depth=depth,
                                            **current_img_params,
                                            **current_seg_params)
        else:
            raise NotImplementedError("Not implemented")

        w, h = ThreeDim_object.level_dim[current_seg_params['seg_level']]
        if w * h > 1e20:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        ################
        # Segmentation #
        ################
        seg_time_elapsed = -1
        if seg:
            print("\nSegmentation")
            print("====================")
            ThreeDim_object, seg_time_elapsed, seg_success = segment(ThreeDim_object,
                                                                     current_seg_params,
                                                                     current_filter_params)

            # Create new directory
            mask_save_subdir = os.path.join(mask_save_dir, slide)
            os.makedirs(mask_save_subdir, exist_ok=True)

            # Save segmentation contours
            ThreeDim_object.saveSegmentation(os.path.join(mask_save_subdir, 'segmentation.pkl'))

            if save_mask:
                print("\nSaving segmentation masks")
                mask_list, z_levels_list = ThreeDim_object.visWSI3D(**current_vis_params,
                                                                    **current_img_params)

                # Save every 20th contours (Just to save time)
                for j, (mask, z_level) in enumerate(zip(mask_list, z_levels_list)):
                    if j % save_step == 0:
                        mask_path = os.path.join(mask_save_subdir, '{}_zlevel_{}.png'.format(slide, z_level))
                        mask.save(mask_path)

        ########################
        # Patching & Stitching #
        ########################
        # Patching & stitching requires segmentation to be successful!
        if seg_success:
            upper_thresh, lower_thresh = identify_image_thresholds(ThreeDim_object.wsi.img,
                                                                   clip_min=df.loc[idx, 'clip_min'],
                                                                   clip_max=df.loc[idx, 'clip_max'],
                                                                   thresh_mode=thresh_mode)

            df.loc[idx, 'clip_max'] = upper_thresh
            df.loc[idx, 'clip_min'] = lower_thresh

            patch_time_elapsed = -1 # Default time
            if patch:
                print("\nPatching")
                print("====================")

                file_path, patch_time_elapsed = patching(ThreeDim_object,
                                                         current_patch_params,
                                                         current_seg_params,
                                                         patch_level=patch_level,
                                                         patch_size=patch_size,
                                                         patch_size_z=patch_size_z,
                                                         step_size=step_size,
                                                         step_size_z=step_size_z,
                                                         save_path=patch_save_dir,
                                                         mode=slice_mode,
                                                         verbose=verbose)

            stitch_time_elapsed = -1
            if stitch:
                print("\nStitching")
                print("====================")
                file_path = os.path.join(patch_save_dir, slide + '_patches.h5')
                heatmap_list, z_levels_list, stitch_time_elapsed = stitching(file_path,
                                                                             downscale=down_ratio,
                                                                             vmin=current_img_params['clip_min'],
                                                                             vmax=current_img_params['clip_max'])

                # Create new directory
                stitch_save_subdir = os.path.join(stitch_save_dir, slide)
                os.makedirs(stitch_save_subdir, exist_ok=True)

                for j, (heatmap, z_level) in enumerate(zip(heatmap_list, z_levels_list)):
                    if j % save_step == 0:
                        stitch_path = os.path.join(stitch_save_subdir, '{}_zlevel_{}.png'.format(slide, z_level))
                        Image.fromarray(heatmap).save(stitch_path)

            df.loc[idx, 'status'] = 'processed'

            print("\nsegmentation took {} seconds".format(seg_time_elapsed))
            print("patching took {} seconds".format(patch_time_elapsed))
            print("stitching took {} seconds".format(stitch_time_elapsed))

        else:
            seg_time_elapsed = 0
            patch_time_elapsed = 0
            stitch_time_elapsed = 0
            df.loc[idx, 'status'] = 'seg_failed'

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    #####################
    # Create & Modify CSV
    df.to_csv(os.path.join(exp_dir, 'process_list_seg.csv'), index=False)

    # Drop unnecessary columns from before
    df = df.drop(columns=['process', 'status', 'seg_level', 'sthresh', 'mthresh', 'close',
                          'use_otsu', 'a_h', 'a_t', 'max_n_holes', 'line_thickness', 'vis_level',
                          'use_padding', 'contour_fn'])
    df.to_csv(os.path.join(exp_dir, 'process_list_extract.csv'), index=False)

    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))

    return seg_times, patch_times


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='seg and patch')
    parser.add_argument('--patch_mode', default='serial', choices=['2D', '3D'], type=str,
                        help='Serial patching or 3D patching')
    parser.add_argument('--depth', default=None, type=int,
                        help='Number of slices')
    parser.add_argument('--source', type=str,
                        help='path to folder containing raw wsi image files')
    parser.add_argument('--patch_size', type=int, default=96,
                        help='patch_size')
    parser.add_argument('--patch_size_z', type=int,
                        help='patch_size along z-dimension')
    parser.add_argument('--slice_mode', type=str, default='single', choices=['single', 'all', 'step'],
                        help='Which slice to take?')
    parser.add_argument('--step_size', type=int, default=96,
                        help='step_size along x-y plane (equal step size for now)')
    parser.add_argument('--step_z', type=int,
                        help='Step size along z-direction. This allows for non-isotropic step size along z direction. If None, same as step_size')
    parser.add_argument('--patch', default=False, action='store_true')
    parser.add_argument('--seg', default=False, action='store_true')
    parser.add_argument('--stitch', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--save_step', type=int, default=10,
                        help='For segmenting and stitching, save every n-th layer. For speed-up')
    parser.add_argument('--save_mask', default=False, action='store_true',
                        help='Save segmentation mask')
    parser.add_argument('--save_dir', default='./patches_save_dir', type=str,
                        help='directory to save processed data')
    parser.add_argument('--preset', default=None, type=str,
                        help='predefined profile of default segmentation and filter parameters (.csv)')
    parser.add_argument('--patch_level', type=int, default=0,
                        help='downsample level at which to patch')
    parser.add_argument('--downscale', default=4, type= int)
    parser.add_argument('--process_list',  type=str, default=None,
                        help='name of list of images to process with parameters (.csv)')
    parser.add_argument('--clip_min', type=float)
    parser.add_argument('--clip_max', type=float)
    parser.add_argument('--black_thresh', type=float)
    parser.add_argument('--mthresh', type=int, default=15)
    parser.add_argument('--sthresh', type=int, default=100)
    parser.add_argument('--a_h', type=int, default=1)
    parser.add_argument('--a_t', type=int, default=5)
    parser.add_argument('--contour_fn', type=str, default='four_pt_easy')
    parser.add_argument('--area_thresh', type=float, default=0.7)
    parser.add_argument('--thresh_mode', type=str, default='fixed', choices=['fixed', 'global', 'local'],
                        help='Method for identifying upper threshold')

    parsed_args = parser.parse_args()

    directories, parameters, process_list = setup(parsed_args)
    # Segmentation and Patching
    seg_times, patch_times = seg_and_patch_3D(**directories,
                                              **parameters,
                                              patch_size=parsed_args.patch_size,
                                              patch_size_z=parsed_args.patch_size_z,
                                              step_size=parsed_args.step_size,
                                              step_size_z=parsed_args.step_z,
                                              depth = parsed_args.depth,
                                              down_ratio=parsed_args.downscale,
                                              seg = parsed_args.seg,
                                              save_mask=parsed_args.save_mask,
                                              stitch=parsed_args.stitch,
                                              patch_level=parsed_args.patch_level,
                                              patch=parsed_args.patch,
                                              patch_mode=parsed_args.patch_mode,
                                              slice_mode=parsed_args.slice_mode,
                                              process_list = process_list,
                                              save_step=parsed_args.save_step,
                                              thresh_mode=parsed_args.thresh_mode,
                                              verbose=parsed_args.verbose)