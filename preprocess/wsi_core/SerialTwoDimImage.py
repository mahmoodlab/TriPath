"""
SerialTwoDimImage treats the volumetric data as a stack of 2D planes, and creates 2D patches.
For the 3D treatment of the volume, refer to ThreeDimImage class
"""

import math
import os

import cv2
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from .BaseImage import BaseImage
from .wsi_utils import savePatchIter_bag_hdf5_3D, initialize_hdf5_bag_3D, screen_coords, to_percentiles
from .img_utils import clip_and_normalize_img, convert_RGB


def normalize_ig_scores(ig_attr, ig_min_val=None, ig_max_val=None):
    """
    Given raw integrated gradient scores, normalize them from -1 to 1

    Args:
    - ig_attr (list): list of ig scores

    Returns:
    - ig_normalized (list): list of normalized ig scores
    """
    if ig_min_val is None:
        ig_min = torch.min(ig_attr)
    else:
        ig_min = torch.tensor(ig_min_val)

    if ig_max_val is None:
        ig_max = torch.max(ig_attr)
    else:
        ig_max = torch.tensor(ig_max_val)

    ig_normalized = torch.zeros_like(ig_attr)
    neg_indices = torch.where(ig_attr < 0)
    pos_indices = torch.where(ig_attr > 0)

    ig_normalized[neg_indices] = ig_attr[neg_indices] / torch.abs(ig_min)
    ig_normalized[pos_indices] = ig_attr[pos_indices] / ig_max

    return ig_normalized


class SerialTwoDimImage(BaseImage):
    """
    Class for treating the volume as a stack of 2D planes. Inherits BaseImage class

    Inputs:
    - clip_min (int): Lower threshold to clip image
    - clip_max (int): Upper threshold to clip image
    - black_thresh (int): Slides whose mean values are below black_thresh will be removed (computational efficiency)
    - sthresh (int): segmentation threshold (range 0 to 255)
    - mthresh (int): radius for median filtering
    - binarize (boolean): If true, binarize the image based on sthresh
    """

    def __init__(self,
                 path,
                 clip_min=0,
                 clip_max=36000,
                 black_thresh=18000,
                 sthresh=30,
                 mthresh=11,
                 z_start=None,
                 depth=None,
                 binarize=True,
                 **kwargs):

        super().__init__(path=path,
                         black_thresh=black_thresh,
                         z_start=z_start,
                         depth=depth)

        self.level_downsamples = [(1, 1)]  # No downsampling
        self.level_dim = self.wsi.level_dimensions

        ################
        # Binarization #
        ################
        # Binarization is required for segmenting the tissue & threshold checks.
        if binarize:
            img_bin = []
            print("Binarizing volume...")  # Required for segmenting & patching (at Level 0)
            for z_level in tqdm(np.arange(self.z_end - self.z_start, dtype=np.int16)):
                img_bin.append(
                    self._getBinarizedImage(z_level=z_level,
                                            sthresh=sthresh,
                                            mthresh=mthresh,
                                            clip_min=clip_min,
                                            clip_max=clip_max)
                )
            self.img_bin = np.stack(img_bin)
        else:
            self.img_bin = []

    def get_levels(self):
        return self.wsi.z_start, self.wsi.z_end

    def process_contours(self,
                         save_path,
                         patch_level=0,
                         step_size_z=1,
                         patch_size=256,
                         step_size=256,
                         area_thresh=0.5,
                         save_patches=True,
                         cont_check_fn=None,
                         use_padding=True,
                         verbose=False,
                         mode='all',
                         **kwargs):
        """
        Create a set of 2D patches from the segmented volume and save the patches to hdf5 files

        Inputs:
        - save_path (str): Path for saving patches
        - cont_check_fn (str): Contour function
        - area_thresh (float): [0, 1]
            Threshold ratio for foreground content to exceed in patching
        - save_patches (boolean): If True, save the patches (Used for patching operation)

        Returns:
        - coordinates (list of tuples): List of coordinates (tuple of 3 elements)
        """

        filtered_coords = []

        if save_patches:
            read_img = True
        else:
            read_img = False

        # Loop through z-levels
        init_flag = False

        # Get the best slice to start from
        if mode == 'single':
            best_slice_idx = self.getBestSlice(self.contours_tissue, mode='max')
            z_levels_list = [best_slice_idx]
            print("\nBest slice: {}".format(best_slice_idx))
        elif mode == 'all':
            z_levels_list = np.arange(0, self.z_end - self.z_start, 1)
        elif mode == 'step':
            z_levels_list = np.arange(0, self.z_end - self.z_start, step_size_z)
        else:
            raise NotImplementedError("Not implemented for mode {}".format(mode))

        for z_level in tqdm(z_levels_list):
            if verbose:
                print("---------------------")
                print("Creating patches for: ", self.name + " slice {}".format(z_level + self.z_start), "...")

            contours = self.contours_tissue[z_level]
            contour_holes = self.holes_tissue[z_level]

            # Within each z-level, loop through contours
            for idx, cont in enumerate(contours):
                holes = contour_holes[idx]
                patch_gen = self._getPatchGenerator(cont=cont,
                                                    save_path=save_path,
                                                    holes=holes,
                                                    cont_idx=idx,
                                                    z_level=z_level,
                                                    patch_level=patch_level,
                                                    patch_size=patch_size,
                                                    step_size=step_size,
                                                    area_thresh=area_thresh,
                                                    cont_check_fn=cont_check_fn,
                                                    read_img=read_img,
                                                    use_padding=use_padding,
                                                    verbose=verbose,
                                                    **kwargs)

                # Initialize first time
                # Level 0, first patch - Otherwise, just iteratively add to original hdf5
                if not init_flag:
                    try:
                        patch = next(patch_gen)
                        init_flag = True

                    # empty contour, continue
                    except StopIteration:
                        continue

                    coords = [patch['z'], patch['x'], patch['y']]
                    filtered_coords.append(coords)

                    if save_patches:
                        initialize_hdf5_bag_3D(patch, self.z_start)

                for patch in patch_gen:

                    coords = [patch['z'], patch['x'], patch['y']]
                    filtered_coords.append(coords)

                    if save_patches:
                        savePatchIter_bag_hdf5_3D(patch)

        return filtered_coords

    def _getPatchGenerator(self,
                           cont,
                           holes,
                           cont_idx,
                           patch_level,
                           save_path,
                           z_level=0,
                           patch_size=256,
                           step_size=256,
                           area_thresh=0.5,
                           cont_check_fn=None,
                           read_img=True,
                           area_check=True,
                           contour_check=True,
                           use_padding=True,
                           verbose=False,
                           **kwargs):

        """
        Get generator for patches.
        Only the patches that passes a series of checks (Contour check + Area check) will be loaded

        Parameters
        ==========
        img: 2D numpy array of binarized image
        area_check: Boolean
            Check whether the foreground/patch area ratio exceeds area_thresh ratio
        contour_check: Boolean
            Check whether patch coordinates are within contours
        """
        # If contour is empty, just ignore it
        # This happens when the tissue area is smaller than a threshold
        if len(cont) == 0:
            return

        coords = self.getContourGrid(cont, patch_level, patch_size, step_size, use_padding)

        start_x = coords['start_x']
        start_y = coords['start_y']
        stop_x = coords['stop_x']
        stop_y = coords['stop_y']
        step_size_x = coords['step_size_x']
        step_size_y = coords['step_size_y']
        patch_downsample = coords['patch_downsample']
        ref_patch_size = coords['ref_patch_size']

        count_pass = 0
        count_contour_fail = 0
        count_area_fail = 0

        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x):

                patch = self.wsi.read_region((z_level, x, y),
                                             patch_level,
                                             (patch_size, patch_size))  # (w, h, 1)

                #############################
                # Initiate a series of checks
                #############################
                # White patch check
                if self.isWhitePatch(patch):
                    print("Patch at z-level ({}, {}, {}) is white patch!!".format(z_level, x, y))
                    continue

                # Contour check
                if contour_check:
                    if not self.isInContours(cont_check_fn,
                                             cont,
                                             (x, y),
                                             holes,
                                             ref_patch_size[0]):
                        count_contour_fail += 1
                        continue

                # Area check (Always perform at highest level)
                if area_check:
                    img_bin = self.img_bin[z_level, y:y + patch_size, x:x + patch_size]
                    if not self.checkArea(img_bin,
                                          area_thresh,
                                          ref_patch_size,
                                          verbose=False):
                        count_area_fail += 1
                        continue

                if not read_img:
                    patch = None

                count_pass += 1

                # x, y coordinates become the coordinates in the downsample, and no long correspond to level 0 of WSI
                patch_info = {'x': x // patch_downsample[0],
                              'y': y // patch_downsample[1],
                              'z': z_level,
                              'cont_idx': cont_idx,
                              'patch_level': patch_level,
                              'downsample': self.level_downsamples[patch_level],
                              'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])),
                              'level_dim': self.level_dim[patch_level],
                              'patch': patch,
                              'name': self.name,
                              'resolution': self.wsi.resolution,
                              'save_path': save_path}

                yield patch_info

        if verbose:
            print("Extracted patches: {}/{}, contour fail {} area fail {}".format(count_pass,
                                                                                  count_pass + count_contour_fail + count_area_fail,
                                                                                  count_contour_fail,
                                                                                  count_area_fail))

    def set_contours(self, contours_tissue, holes_tissue):
        self.contours_tissue = contours_tissue
        self.holes_tissue = holes_tissue

    def filter_contours(self,
                        contours,
                        hierarchy,
                        area_tissue,
                        area_hole,
                        max_n_holes):
        """
        Filter contours (both the tissue and the holes) by area.
        Contours first need to be generated with segmentation
        For holes, also filter by maximum number of holes

        Outputs
        =======
        foreground_contours: tissue contours
        hole_contours: hole contours
        """
        filtered = []

        #########################################
        # find foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)

        for cont_idx in hierarchy_1:
            cont = contours[cont_idx]
            a = cv2.contourArea(cont)

            if a == 0: continue
            if tuple((area_tissue,)) < tuple((a,)): # Only if the contour area exceeds the threshold
                filtered.append(cont_idx)

        all_holes = []
        for parent in filtered:
            all_holes.append(np.flatnonzero(hierarchy[:, 1] == parent))

        foreground_contours = [contours[cont_idx] for cont_idx in filtered]

        hole_contours = []

        for hole_ids in all_holes:
            unfiltered_holes = [contours[idx] for idx in hole_ids]
            unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
            unfilered_holes = unfilered_holes[:max_n_holes]
            filtered_holes = []

            for hole in unfilered_holes:
                if cv2.contourArea(hole) > area_hole:
                    filtered_holes.append(hole)

            hole_contours.append(filtered_holes)

        return foreground_contours, hole_contours

    ################
    # Segmentation #
    ################
    def segmentTissueSerial(self,
                            seg_level=0,
                            filter_params={'a': 100},
                            ref_patch_size=512,
                            **kwargs):
        """
        Wrapper function for segmentTissue, for processing multiple slices.

        Inputs
        ======
        z_start: int, start index of the z_stack
        z_end: end index of the z-stack
        """

        for stack_idx in tqdm(range(self.z_end - self.z_start)):
            self.segmentTissue(seg_level=seg_level,
                               z_level=stack_idx,
                               filter_params=filter_params,
                               ref_patch_size=ref_patch_size)

        # Summary of contour processing
        tissue_count = np.sum([len(tissue) > 0 for tissue in self.contours_tissue])
        hole_count = np.sum([len(hole) > 0 for hole in self.holes_tissue])

        if tissue_count == 0:
            print("No tissue contours exist. Check the segemntation parameters")
            seg_success = False
        else:
            print("Tissue found in {} out of {} slices".format(tissue_count, len(self.contours_tissue)))
            seg_success = True

        return seg_success


    def segmentTissue(self,
                      seg_level=0,
                      z_level=0,
                      filter_params={'a': 100},
                      ref_patch_size=256):
        """
        Segment the tissue and produces contours for the tissue.
        Median thresholding -> Binary threshold

        For good reference for OpenCV contours, refer to
        https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18

        Inputs
        ======
        z_level: int
            z-level at which to segment the tissue
        sthresh: Threshold for binary thresholding
        sthresh_up: Any pixel above sthresh will be converted to this value. Otherwise, converted to zero.
        mthresh: int
            Kernel size for median filtering
        ref_patch_size: int
            If the contour area is smaller than the ref_patch_size x a_t, ignore it
        """

        # Get binarized images
        img_binarized = self.img_bin[z_level]

        scale = self.level_downsamples[seg_level]
        scaled_ref_patch_area = int(ref_patch_size ** 2 / (scale[0] * scale[1]))

        area_tissue = filter_params['a_t'] * scaled_ref_patch_area
        area_hole = filter_params['a_h'] * scaled_ref_patch_area

        # Find and filter contours
        contours, hierarchy = cv2.findContours(img_binarized,
                                               cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_NONE)  # Find contours

        if len(contours) == 0:
            # If no contours found (This is a possibility, since some slices might have insufficient info)
            self.contours_tissue.append([])
            self.holes_tissue.append([])

        else:
            ###########################################
            # hiearchy comes in (1 x numOfcontours x 4)
            # For each contour, the third element is child and the fourth element is parent
            # We only need these two information for filtering
            ###########################################
            hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

            if filter_params:
                foreground_contours, hole_contours = self.filter_contours(contours,
                                                                          hierarchy,
                                                                          area_tissue=area_tissue,
                                                                          area_hole=area_hole,
                                                                          max_n_holes=filter_params['max_n_holes'])  # Necessary for filtering out artifacts

            self.contours_tissue.append(self.scaleContourDim(foreground_contours, scale[1:]))   # Scale is 3-element tuple. scaleContourDim is for 2d
            self.holes_tissue.append(self.scaleHolesDim(hole_contours, scale[1:]))
        self.seg_level = seg_level


    @staticmethod
    def checkArea(img_binarized,
                  area_ratio,
                  patch_size,
                  verbose=False):
        """
        Check whether the area of the intersection of contour and patch box
        exceeds area threshold

        Parameters
        ==========
        img_binarized: binarized numpy array (255 for foreground, 0 for background)
        area_ratio: float between [0, 1]
        patch_size: tuple (patch_size_x, patch_size_y)
        """
        area_thresh = area_ratio * patch_size[0] * patch_size[1]
        area = np.sum(img_binarized) / 255

        if verbose:
            print("shape ", img_binarized.shape,
                  " Area: {} Total_area: {} ratio: {}".format(area,
                                                              patch_size[0] * patch_size[1],
                                                              area / (patch_size[0] * patch_size[1])))

        return 1 if area >= area_thresh else 0

    def visWSI3D(self,
                  vis_level=0,
                  color=(0, 255, 0),
                  hole_color=(0, 0, 255),
                  annot_color=(255, 0, 0),
                  line_thickness=12,
                  max_size=None,
                  crop_window=None,
                 clip_min=0,
                 clip_max=1,
                 **kwargs
                 ):
        """
        Create contour masks for each slice for the 3D volume
        This is a wrapper function for visWSI method
        """

        img_list = []
        z_levels_list = []

        for z_level in tqdm(range(self.z_end - self.z_start)):
            img = self.visWSI(vis_level=vis_level,
                              z_level=z_level,
                              color=color,
                              hole_color=hole_color,
                              annot_color=annot_color,
                              line_thickness=line_thickness,
                              max_size=max_size,
                              crop_window=crop_window,
                              clip_min=clip_min,
                              clip_max=clip_max)

            img_list.append(img)
            z_levels_list.append(z_level + self.z_start)

        return img_list, z_levels_list

    def visWSI(self,
               vis_level=0,
               z_level=0,
               color=(0, 255, 0),
               hole_color=(0, 0, 255),
               annot_color=(255, 0, 0),
               clip_min=0,
               clip_max=1,
               line_thickness=12,
               max_size=None,
               crop_window=None):
        """
        Draw contours on the slice
        """

        img = np.array(self.wsi.read_region((z_level, 0, 0),
                                            vis_level,
                                            self.level_dim[vis_level]))

        img = clip_and_normalize_img(img, clip_min=clip_min, clip_max=clip_max)
        img = convert_RGB(img)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        downsample = self.level_downsamples[vis_level]
        scale = [1 / downsample[0], 1 / downsample[1]]  # Scaling from 0 to desired level
        line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))

        contours = self.contours_tissue[z_level]
        contour_holes = self.holes_tissue[z_level]

        if contours is not None:
            cv2.drawContours(img, self.scaleContourDim(contours, scale),
                             -1, color, line_thickness, lineType=cv2.LINE_8)

            for holes in contour_holes:
                cv2.drawContours(img, self.scaleContourDim(holes, scale),
                                 -1, hole_color, line_thickness, lineType=cv2.LINE_8)

        img = Image.fromarray(img)

        if crop_window is not None:
            top, left, bot, right = crop_window
            left = int(left * scale[0])
            right = int(right * scale[0])
            top = int(top * scale[1])
            bot = int(bot * scale[1])
            crop_window = (top, left, bot, right)
            img = img.crop(crop_window)

        w, h = img.size
        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        return img

    ################################
    # Attention heatmap generation #
    ################################
    def visHeatmap(self,
                   scores,
                   coords,
                   vis_level=-1,
                   top_left=None, bot_right=None,
                   patch_size=96,
                   blank_canvas=False,
                   alpha=0.4,
                   blur=False,
                   overlap=0.0,
                   segment=True,
                   use_holes=True,
                   convert_to_percentiles=False,
                   binarize=False,
                   thresh=0.5,
                   max_size=None,
                   custom_downsample=1,
                   clip_min=0,
                   clip_max=1,
                   cmap_normalize='all',
                   cmap_min=-2,
                   cmap_max=2,
                   save_path_temp='/home/andrew/workspace/ThreeDimPlayground/temp',
                   cmap='coolwarm'):
        """
        Visualize heatmap

        Args:
            scores (numpy array of float): Attention scores
            coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
            vis_level (int): WSI pyramid level to visualize
            patch_size (tuple of int): Patch dimensions (relative to lvl 0)
            blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
            canvas_color (tuple of uint8): Canvas color
            alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
            blur (bool): apply gaussian blurring
            overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
            segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that
                            self.contours_tissue and self.holes_tissue are not None
            use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
            convert_to_percentiles (bool): whether to convert attention scores to percentiles
            binarize (bool): only display patches > threshold
            threshold (float): binarization threshold
            max_size (int): Maximum canvas size (clip if goes over)
            custom_downsample (int): additionally downscale the heatmap by specified factor
            cmap (str): name of matplotlib colormap to use
            cmpa_normlize (str): 'all' or 'slice'. How to normalize scores within each heatmap

        Returns:
        - img_list (list of Pillow images): Heatmaps img lists
        - z_levels_list (list of int): List of heatmap levels
        """
        if vis_level < 0:
            vis_level = self.wsi.get_best_level_for_downsample(32)

        downsample = self.level_downsamples[vis_level]
        if len(downsample) == 2:
            scale = [1 / downsample[idx] for idx in range(len(downsample))]  # Scaling from 0 to desired level
            scale = [1] + scale  # z-dimension
        else:
            scale = [1/downsample[idx] for idx in range(len(downsample))] # Scaling from 0 to desired level

        ## img_obj = self if overlay_obj is None else overlay_obj

        if len(scores.shape) == 2:
            scores = scores.flatten()

        if binarize:
            if thresh < 0:
                threshold = 1.0 / len(scores)
            else:
                threshold = thresh
        else:
            threshold = -100

        ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
        if top_left is not None and bot_right is not None:
            scores, coords = screen_coords(scores, coords, top_left, bot_right)
            coords = coords - top_left
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
            z = self.z_end - self.z_start

        else:
            region_size = self.level_dim[vis_level]
            top_left = (0, 0)
            bot_right = self.level_dim[0]
            w, h = region_size
            z = self.z_end - self.z_start

        patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)

        print('\ncreating heatmap for: ')
        print('top_left: ', top_left, 'bot_right: ', bot_right)
        print('w: {}, h: {}'.format(w, h))
        print('scaled patch size: ', patch_size)

        scores_min = 1e5
        scores_max = -1e5
        scores_dict = {}

        ###### normalize filtered scores ######
        if convert_to_percentiles:
            scores = to_percentiles(scores)
            scores /= 100

        ############################
        # Compute attention scores #
        ############################
        # Calculate the heatmap of raw attention scores (before colormap) by accumulating scores over overlapped regions
        # To prevent memory overflow, overlay/counter information must be saved/loaded dynamically
        #
        # heatmap overlay: tracks attention score over each pixel of heatmap
        # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap

        for z_level in tqdm(range(z)):
            counter = np.full((h, w), 0).astype(np.uint16)
            overlay = np.full((h, w), 0).astype(np.float32)
            np.save(os.path.join(save_path_temp, 'counter_{}'.format(z_level)), counter)
            np.save(os.path.join(save_path_temp, 'overlay_{}'.format(z_level)), overlay)
            del counter, overlay

        count = 0
        # Identify unique z levels
        z_list = [coords[idx][0, ...] for idx in range(len(coords))]
        z_unique_list = np.unique(z_list)

        print("Accumulating heatmap attention scores...")
        for z_level in tqdm(z_unique_list):
            indices = np.flatnonzero(z_list == z_level)
            coords_lev = coords[indices]
            scores_lev = scores[indices]

            # Edge cases
            z_first = max(0, z_level)
            z_last = min(z_level + patch_size[0], z)

            for z_level_inner in range(z_first, z_last):
                overlay = np.load(os.path.join(save_path_temp, 'overlay_{}.npy'.format(z_level_inner)))
                counter = np.load(os.path.join(save_path_temp, 'counter_{}.npy'.format(z_level_inner)))

                # Within each z-level, accumulate scores and counter
                for coord, score in zip(coords_lev, scores_lev):
                    if score >= threshold:
                        if binarize:
                            score = 1.0
                            count += 1
                    else:
                        score = 0.0

                    overlay[coord[2]:coord[2] + patch_size[2], coord[1]:coord[1] + patch_size[1]] += score
                    counter[coord[2]:coord[2] + patch_size[2], coord[1]:coord[1] + patch_size[1]] += 1

                np.save(os.path.join(save_path_temp, 'counter_{}'.format(z_level_inner)), counter)
                np.save(os.path.join(save_path_temp, 'overlay_{}'.format(z_level_inner)), overlay)

        if binarize:
            print('\nbinarized tiles based on cutoff of {}'.format(threshold))
            print('identified {}/{} patches as positive'.format(count, len(coords)))

        # Divide the accumulated attention score by number of overlaps
        for z_level in range(z):
            overlay = np.load(os.path.join(save_path_temp, 'overlay_{}.npy'.format(z_level)))
            counter = np.load(os.path.join(save_path_temp, 'counter_{}.npy'.format(z_level)))

            # fetch attended region and average accumulated attention
            zero_mask = counter == 0

            if binarize:
                overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
            else:
                overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]

            if len(overlay[~zero_mask]) == 0:
                continue

            slice_min = np.min(overlay[~zero_mask])
            slice_max = np.max(overlay[~zero_mask])

            if slice_min < scores_min:
                scores_min = slice_min

            if slice_max > scores_max:
                scores_max = slice_max

            scores_dict[z_level] = [slice_min, slice_max]
            np.save(os.path.join(save_path_temp, 'overlay_{}'.format(z_level)), overlay)
            # print(z_level, slice_min, slice_max)
        scores_dict['all'] = [scores_min, scores_max]
        print("Min score: {}, Max score: {} ".format(scores_min, scores_max))
        ###################################
        # Blend attention map onto images #
        ###################################
        print("Blending attention map onto images")
        img_list = []
        z_levels_list = []

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        for i, z_level in enumerate(tqdm(z_unique_list)):
            indices = np.flatnonzero(z_list == z_level)
            coords_lev = coords[indices]
            scores_lev = scores[indices]

            z_inner_start = max(0, z_level)
            # Iterate only until the next unique z index
            if i == len(z_unique_list) - 1:
                z_inner_end = z
            else:
                z_inner_end = z_unique_list[i + 1]

            # Parse through all the sublevels within 3D patch
            for z_level_inner in range(z_inner_start, z_inner_end):
                overlay = np.load(os.path.join(save_path_temp, 'overlay_{}.npy'.format(z_level_inner)))

                overlay_cvt = self.compute_heatmap_scores(overlay,
                                                          scores_dict,
                                                          cmap_normalize,
                                                          z_level_inner)

                if blur:
                    overlay_cvt = cv2.GaussianBlur(overlay_cvt,
                                                   tuple((patch_size[1:] * (1 - overlap)).astype(int) * 2 + 1), 0)

                if segment:
                    tissue_mask = get_seg_mask(self.contours_tissue[z_level_inner],
                                               self.holes_tissue[z_level_inner],
                                               region_size,
                                               scale[:2],
                                               use_holes=use_holes,
                                               offset=tuple(top_left))

                img = np.array(Image.new(size=region_size, mode="RGB", color=(255, 255, 255)))

                for idx in range(len(coords_lev)):
                    score = scores_lev[idx]
                    coord = coords_lev[idx]

                    if score >= threshold:
                        # attention block
                        attn_block = overlay_cvt[coord[2]:coord[2] + patch_size[2], coord[1]:coord[1] + patch_size[1]]
                        color_block = (cmap(attn_block) * 255)[:, :, :3].astype(
                            np.uint8)  # color block (cmap applied to attention block)

                        if segment:
                            img_block = img[coord[2]:coord[2] + patch_size[2], coord[1]:coord[1] + patch_size[1]].copy()
                            mask_block = tissue_mask[coord[2]:coord[2] + patch_size[2], coord[1]:coord[1] + patch_size[1]]
                            img_block[mask_block] = color_block[
                                mask_block]  # copy over only tissue masked portion of color block
                            pass
                        else:
                            # copy over entire color block
                            img_block = color_block

                        # rewrite image block with scores
                        img[coord[2]:coord[2] + patch_size[2], coord[1]:coord[1] + patch_size[1]] = img_block.copy()

                if blur:
                    img = cv2.GaussianBlur(img, tuple((patch_size[1:] * (1 - overlap)).astype(int) * 2 + 1), 0)

                if alpha < 1.0:
                    img = self.block_blending(img,
                                              z_level_inner,
                                              vis_level,
                                              top_left,
                                              bot_right,
                                              clip_min=clip_min,
                                              clip_max=clip_max,
                                              alpha=alpha,
                                              blank_canvas=blank_canvas,
                                              block_size=1024)

                img = Image.fromarray(img)
                w, h = img.size

                if custom_downsample > 1:
                    img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

                if max_size is not None and (w > max_size or h > max_size):
                    resizeFactor = max_size / w if w > h else max_size / h
                    img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

                img_list.append(img)
                z_levels_list.append(z_level_inner)

        flist = glob(save_path_temp + '/*')
        for fname in flist:
            os.remove(fname)

        return img_list, z_levels_list

    ##################
    # Helper methods #
    ##################
    def compute_heatmap_scores(self, overlay, scores_dict={}, cmap_normalize='all', lev=0):
        """
        Compute heatmap scores based on different normalization mode
        """
        scores_min, scores_max = scores_dict['all']

        if cmap_normalize == 'all':
            overlay_cvt = (overlay - scores_min) / (scores_max - scores_min)

        elif cmap_normalize == 'ig':
            overlay_cvt = normalize_ig_scores(torch.tensor(overlay),
                                              ig_min_val=scores_min,
                                              ig_max_val=scores_max).numpy()

            overlay_cvt[overlay_cvt < 0] = 0


        elif cmap_normalize == 'slice':
            if lev in scores_dict.keys():
                s_min, s_max = scores_dict[lev]
            else:
                s_min = scores_min
                s_max = scores_max
            overlay_cvt = (overlay - s_min) / (s_max - s_min)

        elif cmap_normalize == 'ig_slice':
            if lev in scores_dict.keys():
                s_min, s_max = scores_dict[lev]
            else:
                s_min = scores_min
                s_max = scores_max

            overlay_cvt = normalize_ig_scores(torch.tensor(overlay),
                                              ig_min_val=s_min,
                                              ig_max_val=s_max).numpy()

            lower_val = -1
            upper_val = 1
            overlay_cvt[overlay_cvt < lower_val] = lower_val
            overlay_cvt[overlay_cvt > upper_val] = upper_val
            overlay_cvt = (overlay_cvt - lower_val) / (upper_val - lower_val)

        else:
            raise NotImplementedError("Not implemented for cmap normalization method of ", cmap_normalize)

        return overlay_cvt

    def _getBinarizedImage(self,
                           z_level=0,
                           level=0,
                           sthresh=20,
                           sthresh_up=255,
                           mthresh=7,
                           close=0,
                           clip_min=0,
                           clip_max=1,
                           use_otsu=False):
        """
        Returns binarized whole image for segmentation

        Inputs
        ======
        clip_min: int
            minimum intensity value below which every intensity will be cast to
        clip_max: int
            maximum intensity value above which every intensity will be cast to
        sthresh: int
            binarization threshold (between 0 and 255)
        """
        # Read the whole image corresponding to z_level
        img = self.wsi.read_region((int(z_level), 0, 0),
                                   level,
                                   self.level_dim[level])
        img = np.array(img)
        channel_dim = img.shape[-1]

        # Clip image
        img = clip_and_normalize_img(img, clip_min=clip_min, clip_max=clip_max) * 255
        img = img.astype(np.uint8)

        #####################
        # Normalize to uint8 (Downstream CV2 methods only take uint8 dtype)
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_med = cv2.medianBlur(img, mthresh)  # Median blurring helps identify smoother contours

        if channel_dim == 1:
            img_gray = img_med
        else:
            img_gray = cv2.cvtColor(img_med, cv2.COLOR_RGB2GRAY)

        # Thresholding
        if use_otsu:
            _, img_binarized = cv2.threshold(img_gray, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        else:
            _, img_binarized = cv2.threshold(img_gray, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_binarized = cv2.morphologyEx(img_binarized, cv2.MORPH_CLOSE, kernel)

        return img_binarized

    def getContourGrid(self,
                       contour=None,
                       patch_level=0,
                       patch_size=224,
                       step_size=224,
                       use_padding=True):
        """
        Get grid info based on bounding box for given contour
        """
        if contour is not None:
            start_x, start_y, w, h = cv2.boundingRect(contour)
        else:
            start_x, start_y, w, h = (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])

        # the downsample corresponding to the patch_level
        patch_downsample = self.level_downsamples[patch_level]
        # size of patch at level 0 (reference size)
        ref_patch_size = tuple((np.array((patch_size,) * 2) * np.array(patch_downsample)).astype(int))
        # step sizes to take at level 0 (No need for slice axis)
        ref_step_size = tuple((np.array((step_size,) * 2) * np.array(patch_downsample)).astype(int))

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1])
            stop_x = min(start_x + w, img_w - ref_patch_size[0])

        coords = {'start_x': start_x,
                  'start_y': start_y,
                  'stop_x': stop_x,
                  'stop_y': stop_y,
                  'step_size_x': ref_step_size[0],
                  'step_size_y': ref_step_size[1],
                  'patch_downsample': patch_downsample,
                  'ref_patch_size': ref_patch_size
                  }

        return coords


def get_seg_mask(contours_tissue,
                 contours_holes,
                 region_size,
                 scale, use_holes=False, offset=(0, 0, 0)):
    tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
    offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))

    if len(contours_tissue) > 0:
        contours_tissue, contours_holes = zip(*sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))
        for idx in range(len(contours_tissue)):
            cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1), offset=offset, thickness=-1)

            if use_holes:
                cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0), offset=offset, thickness=-1)
            # contours_holes = self._scaleContourDim(self.holes_tissue, scale, holes=True, area_thresh=area_thresh)

    tissue_mask = tissue_mask.astype(bool)

    return tissue_mask