"""
Classes for 3D files. Extension of the SerialTwoDimImage object

Except for visualization purpose, all the images/patches are in uint16 format
(only converted to uint8 for visualization), since 1) DICOM comes in uint16 format
and 2) uint16 contains more information

"""

import cv2
import numpy as np

import sys
sys.path.append('../')
from .SerialTwoDimImage import SerialTwoDimImage
from .wsi_utils import savePatchIter_bag_hdf5_3D, initialize_hdf5_bag_3D,\
                                screen_coords, to_percentiles

# Threshold for microCT calcification
WHITE_THRESHOLD = 6e4


class ThreeDimImage(SerialTwoDimImage):
    """
    Inherits SerialTwoDimImage object. The only difference is how patching operation is performed.
    """
    def __init__(self,
                 path,
                 clip_min=22000,
                 clip_max=36000,
                 black_thresh=11000,
                 sthresh=80,
                 mthresh=11,
                 z_start=None,
                 depth=None,
                 binarize=True,
                 **kwargs):

        """
        contours_tissue: List (different z-levels) of tissue contours
        """
        super().__init__(path=path,
                         black_thresh=black_thresh,
                         clip_min=clip_min,
                         clip_max=clip_max,
                         sthresh=sthresh,
                         mthresh=mthresh,
                         z_start=z_start,
                         depth=depth,
                         binarize=binarize)

        self.level_downsamples = [(1, 1, 1)]    # No downsampling
        self.level_dim = self.wsi.level_dimensions  # Two-dimensional

    def process_contours(self,
                         save_path,
                         patch_level=0,
                         patch_size=96,
                         patch_size_z=None,
                         step_size=96,
                         step_size_z=None,
                         area_thresh=0.5,
                         save_patches=True,
                         cont_check_fn=None,
                         mode='all',
                         use_padding=True,
                         **kwargs):
        """
        Create bag of 3D patches from the segmented results and optionally save to hdf5 file.

        The pipeline is as follows:
        1) Use segmentTissueSerial from the parent class to get contours across all slice levels
        2) Determine best slice (getBestSlice) and generate index array of top z coordinates
        3) 3D patch the data based on the z coordinates & contours at each level
        4) Filter the patches based on contour and area check

        Parameters
        ==========
        area_thresh: float in [0, 1]
            Threshold ratio for foreground content to exceed in patching
        save_patches: Boolean
            Whether to save 3d patch or not
        """

        # Get the best slice to start from
        best_slice_idx = self.getBestSlice(self.contours_tissue, mode='max')

        print("---------------------------")
        print(self.z_start, self.z_end, len(self.contours_tissue))
        print("Best slice: {}".format(best_slice_idx))

        coords_cntr = self.getContourGrid(best_slice_idx,
                                          patch_level,
                                          patch_size=patch_size,
                                          patch_size_z=patch_size_z if patch_size_z is not None else patch_size,
                                          step_size=step_size,
                                          step_size_z=step_size_z if step_size_z is not None else step_size,
                                          use_padding=use_padding,
                                          mode=mode)

        filtered_coords = []
        read_img = True if save_patches else False

        # Loop through z-levels
        step_size_z = coords_cntr['ref_step_size'][0] // coords_cntr['patch_downsample'][0]
        init_flag = False
        for z_level in np.arange(coords_cntr['start_z'], coords_cntr['stop_z'], step_size_z, dtype=np.int16):
            slice_start = z_level
            slice_end = z_level + coords_cntr['ref_patch_size'][0] // coords_cntr['patch_downsample'][0]
            print("\nCreating patches for: ", self.name + " spanning slice {} ~ {}".format(slice_start, slice_end))

            contours = self.contours_tissue[z_level]
            contour_holes = self.holes_tissue[z_level]

            # Within each z-level, loop through patches
            for idx, cont in enumerate(contours):
                holes = contour_holes[idx]
                patch_gen = self._getPatchGenerator(cont=cont,
                                                    coords=coords_cntr,
                                                    holes=holes,
                                                    cont_idx=idx,
                                                    z_level=z_level,
                                                    patch_level=patch_level,
                                                    save_path=save_path,
                                                    read_img=read_img,
                                                    area_thresh=area_thresh,
                                                    cont_check_fn=cont_check_fn,
                                                    **kwargs)

                # Initialize hdf5 file the very first time
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
                        initialize_hdf5_bag_3D(patch,
                                               self.z_start)

                for patch in patch_gen:
                    coords = [patch['z'], patch['x'], patch['y']]
                    filtered_coords.append(coords)

                    if save_patches:
                        savePatchIter_bag_hdf5_3D(patch)

        print("\nTotal {} patches created!".format(len(filtered_coords)))
        return filtered_coords

    def _getPatchGenerator(self,
                           cont,
                           coords,
                           holes,
                           cont_idx,
                           patch_level,
                           save_path,
                           z_level=0,
                           start_z=0,
                           area_thresh=0.75,
                           cont_check_fn=None,
                           read_img=True,
                           area_check=True,
                           contour_check=True,
                           **kwargs):

        """
        Get generator for patches.
        Only the patches that passes a series of checks (Contour check + Area check) will be loaded

        Parameters
        ==========
        img: 3D Numpy array of binarized image - Used for segmenting
        area_check: Boolean
            Check whether the foreground/patch area ratio exceeds area_thresh ratio
        contour_check: Boolean
            Check whether patch coordinates are within contours
        """
        # If contour is empty, just ignore it
        # This happens when the tissue area is smaller than a threshold
        if len(cont) == 0:
            return

        count_pass = 0
        count_contour_fail = 0
        count_area_fail = 0

        patch_downsample = coords['patch_downsample']

        ref_step_size = coords['ref_step_size']
        step_size_x = ref_step_size[1] // patch_downsample[1]
        step_size_y = ref_step_size[2] // patch_downsample[2]

        ref_patch_size = coords['ref_patch_size']
        patch_size = (ref_patch_size[0] // patch_downsample[0],
                      ref_patch_size[1] // patch_downsample[1],
                      ref_patch_size[2] // patch_downsample[2])

        for y in range(coords['start_y'], coords['stop_y'], step_size_y):
            for x in range(coords['start_x'], coords['stop_x'], step_size_x):
                # Patch size of (z, w, h, c)
                patch = self.wsi.read_region((z_level, x, y),
                                             patch_level,
                                             patch_size)

                #############################
                # Initiate a series of checks
                #############################
                # Contour check (TODO)
                # Right now this only performs contour check on top slice
                # Need to change to 3D contour incorporation

                if self.isWhitePatch(patch, area_ratio=0.05):
                    continue

                if contour_check:
                    if not self.isInContours(cont_check_fn,
                                             cont,
                                             (x, y),
                                             holes,
                                             ref_patch_size[1]):
                        count_contour_fail += 1
                        continue

                # Volume check (Always perform at highest level)
                if area_check:
                    if not self.checkVolume(self.img_bin,
                                            (x, y, z_level),
                                            start_z,
                                            area_thresh,
                                            ref_patch_size,
                                            verbose=False):
                        count_area_fail += 1
                        continue

                # If all checks are passed, generate the list of patches
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

        count_total = count_pass + count_contour_fail + count_area_fail
        print("Extracted patches: {}/{}, contour fail {} area fail {}".format(count_pass,
                                                                              count_total,
                                                                              count_contour_fail,
                                                                              count_area_fail))

    #########################################
    # Static method for checking contour volume
    @staticmethod
    def checkVolume(vol_bin,
                    coord,
                    start_z,
                    vol_ratio,
                    patch_size,
                    verbose=False):
        """
        Check whether the area of the intersection of contour and patch box
        exceeds area threshold

        Parameters
        ==========
        vol_bin: binarized 3D numpy array (255 for foreground, 0 for background)
        coord: Start coordinate (x, y, z)
        vol_ratio: float between [0, 1]
        patch_size: tuple (patch_size_x, patch_size_y, patch_size_z)
        """

        x, y, z = coord

        # If no binarized image, simply bypass the volume check
        if len(vol_bin) == 0:
            return 1
        else:
            # Padding
            if z < start_z:
                patch = vol_bin[: z + patch_size[0], y: y + patch_size[1], x: x + patch_size[2]]
            else:
                patch = vol_bin[z: z + patch_size[0], y: y + patch_size[1], x: x + patch_size[2]]

            vol_thresh = vol_ratio * patch_size[0] * patch_size[1] * patch_size[2]
            vol = np.sum(patch / 255)

            if verbose:
                print("shape ", vol_bin.shape, " Volume: {} Total_volume: {} ratio: {}".format(vol,
                                                                                               patch_size[0] * patch_size[1] * patch_size[2],
                                                                                               vol/(patch_size[0] * patch_size[1] * patch_size[2])))

            return 1 if vol >= vol_thresh else 0

    def getContourGrid(self,
                       best_slice_idx=0,
                       patch_level=0,
                       patch_size=96,
                       patch_size_z=None,
                       step_size=96,
                       step_size_z=None,
                       use_padding=True,
                       mode='all'):
        """
        Get grid info based on bounding box for given contour
        """

        # If there exists multiple contours, this concatenates them into single contour
        if len(self.contours_tissue[best_slice_idx]) > 0:
            contour = np.concatenate(self.contours_tissue[best_slice_idx])
        else:
            contour = None

        if contour is not None:
            start_x, start_y, w, h = cv2.boundingRect(contour)
        else:
            start_x, start_y, w, h = (0, 0,
                                      self.level_dim[patch_level][0],
                                      self.level_dim[patch_level][1])

        # the downsample corresponding to the patch_level
        patch_downsample = self.level_downsamples[patch_level]
        # size of patch at level 0 (reference size)
        ref_patch_size = tuple((np.array([patch_size_z, patch_size, patch_size]) * np.array(patch_downsample)).astype(int))
        # step sizes to take at level 0 (No need for slice axis)
        ref_step_size = tuple((np.array([step_size_z, step_size, step_size]) * np.array(patch_downsample)).astype(int))

        # Get z-direction information
        if mode == 'all':
            start_z = np.arange(best_slice_idx, -ref_step_size[0], -ref_step_size[0], dtype=np.int16)[-1]
            stop_z = len(self.contours_tissue)
        elif mode == 'single':
            start_z = best_slice_idx
            stop_z = best_slice_idx + ref_step_size[0]
        else:
            raise NotImplementedError("Not implemented for slice mode {}".format(mode))

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[2])
            stop_x = min(start_x + w, img_w - ref_patch_size[1])

        coords = {'start_x': start_x,
                  'start_y': start_y,
                  'start_z': start_z,
                  'stop_x': stop_x,
                  'stop_y': stop_y,
                  'stop_z': stop_z,
                  'patch_downsample': patch_downsample,
                  'ref_step_size': ref_step_size,
                  'ref_patch_size': ref_patch_size}

        return coords

    ################################
    # Attention heatmap generation #
    ################################
    # def visHeatmap(self,
    #                scores,
    #                coords,
    #                overlay_obj=None,
    #                vis_level=-1,
    #                top_left=None, bot_right=None,
    #                patch_size=(96, 96, 96),
    #                blank_canvas=False,
    #                canvas_color=(220, 20, 50),
    #                alpha=0.4,
    #                blur=False,
    #                overlap=0.0,
    #                segment=True,
    #                use_holes=True,
    #                convert_to_percentiles=False,
    #                binarize=False,
    #                thresh=0.5,
    #                max_size=None,
    #                custom_downsample=1,
    #                clip_min=0,
    #                clip_max=1,
    #                cmap_normalize='all',
    #                save_path_temp='/home/andrew/workspace/ThreeDimPlayground/temp',
    #                cmap='coolwarm'):
    #     """
    #     Visualize heatmap
    #
    #     Args:
    #         scores (numpy array of float): Attention scores
    #         coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
    #         vis_level (int): WSI pyramid level to visualize
    #         patch_size (tuple of int): Patch dimensions (relative to lvl 0)
    #         blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
    #         canvas_color (tuple of uint8): Canvas color
    #         alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
    #         blur (bool): apply gaussian blurring
    #         overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
    #         segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that
    #                         self.contours_tissue and self.holes_tissue are not None
    #         use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
    #         convert_to_percentiles (bool): whether to convert attention scores to percentiles
    #         binarize (bool): only display patches > threshold
    #         threshold (float): binarization threshold
    #         max_size (int): Maximum canvas size (clip if goes over)
    #         custom_downsample (int): additionally downscale the heatmap by specified factor
    #         cmap (str): name of matplotlib colormap to use
    #         save_path_temp (str): path for saving intermediate files
    #     """
    #
    #     if vis_level < 0:
    #         vis_level = self.wsi.get_best_level_for_downsample(32)
    #
    #     downsample = self.level_downsamples[vis_level]
    #     if len(downsample) == 2:
    #         scale = [1 / downsample[idx] for idx in range(len(downsample))]  # Scaling from 0 to desired level
    #         scale = [1] + scale  # z-dimension
    #     else:
    #         scale = [1/downsample[idx] for idx in range(len(downsample))] # Scaling from 0 to desired level
    #
    #     img_obj = self if overlay_obj is None else overlay_obj
    #
    #     if len(scores.shape) == 2:
    #         scores = scores.flatten()
    #
    #     if binarize:
    #         if thresh < 0:
    #             threshold = 1.0 / len(scores)
    #         else:
    #             threshold = thresh
    #     else:
    #         # TODO verify this
    #         threshold = -100
    #
    #     ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
    #     if top_left is not None and bot_right is not None:
    #         scores, coords = screen_coords(scores, coords, top_left, bot_right)
    #         coords = coords - top_left
    #         top_left = tuple(top_left)
    #         bot_right = tuple(bot_right)
    #         w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
    #         region_size = (w, h)
    #         z = self.z_end - self.z_start
    #
    #     else:
    #         region_size = self.level_dim[vis_level]
    #         top_left = (0, 0)
    #         bot_right = self.level_dim[0]
    #         w, h = region_size
    #         z = self.z_end - self.z_start
    #
    #     patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    #     coords = np.ceil(coords * np.array(scale)).astype(int)
    #
    #     print('top_left: ', top_left, 'bot_right: ', bot_right, ' w: {}, h: {}'.format(w, h))
    #     print('patch size: ', patch_size)
    #
    #     scores_min = 1e5
    #     scores_max = -1e5
    #     scores_dict = {}
    #
    #     ###### normalize filtered scores ######
    #     if convert_to_percentiles:
    #         scores = to_percentiles(scores)
    #         scores /= 100
    #
    #     ############################
    #     # Compute attention scores #
    #     ############################
    #     # Calculate the heatmap of raw attention scores (before colormap) by accumulating scores over overlapped regions
    #     # To prevent memory overflow, overlay/counter information must be saved/loaded dynamically
    #     #
    #     # heatmap overlay: tracks attention score over each pixel of heatmap
    #     # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
    #
    #     for z_level in tqdm(range(z)):
    #         counter = np.full((h, w), 0).astype(np.uint16)
    #         overlay = np.full((h, w), 0).astype(np.float32)
    #         np.save(os.path.join(save_path_temp, 'counter_{}'.format(z_level)), counter)
    #         np.save(os.path.join(save_path_temp, 'overlay_{}'.format(z_level)), overlay)
    #         del counter, overlay
    #
    #
    #     count = 0
    #     # Identify unique z levels
    #     z_list = [coords[idx][0, ...] for idx in range(len(coords))]
    #     z_unique_list = np.unique(z_list)
    #     print("Unique z levels ", z_unique_list)
    #     print("Accumulating heatmap attention scores...")
    #     for z_level in tqdm(z_unique_list):
    #         indices = np.flatnonzero(z_list == z_level)
    #         coords_lev = coords[indices]
    #         scores_lev = scores[indices]
    #
    #         # Edge cases
    #         z_first = max(0, z_level)
    #         z_last = min(z_level + patch_size[0], z)
    #
    #         for z_level_inner in range(z_first, z_last):
    #             overlay = np.load(os.path.join(save_path_temp, 'overlay_{}.npy'.format(z_level_inner)))
    #             counter = np.load(os.path.join(save_path_temp, 'counter_{}.npy'.format(z_level_inner)))
    #
    #             # Within each z-level, accumulate scores and counter
    #             for coord, score in zip(coords_lev, scores_lev):
    #                 if score >= threshold:
    #                     if binarize:
    #                         score = 1.0
    #                         count += 1
    #                 else:
    #                     score = 0.0
    #
    #                 overlay[coord[2]:coord[2] + patch_size[2], coord[1]:coord[1] + patch_size[1]] += score
    #                 counter[coord[2]:coord[2] + patch_size[2], coord[1]:coord[1] + patch_size[1]] += 1
    #
    #             np.save(os.path.join(save_path_temp, 'counter_{}'.format(z_level_inner)), counter)
    #             np.save(os.path.join(save_path_temp, 'overlay_{}'.format(z_level_inner)), overlay)
    #
    #     if binarize:
    #         print('\nbinarized tiles based on cutoff of {}'.format(threshold))
    #         print('identified {}/{} patches as positive'.format(count, len(coords)))
    #
    #     # Divide the accumulated attention score by number of overlaps
    #     for z_level in range(z):
    #         overlay = np.load(os.path.join(save_path_temp, 'overlay_{}.npy'.format(z_level)))
    #         counter = np.load(os.path.join(save_path_temp, 'counter_{}.npy'.format(z_level)))
    #
    #         # fetch attended region and average accumulated attention
    #         zero_mask = counter == 0
    #
    #         if binarize:
    #             overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
    #         else:
    #             overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
    #
    #         if len(overlay[~zero_mask]) == 0:
    #             continue
    #
    #         slice_min = np.min(overlay[~zero_mask])
    #         slice_max = np.max(overlay[~zero_mask])
    #
    #         if slice_min < scores_min:
    #             scores_min = slice_min
    #
    #         if slice_max > scores_max:
    #             scores_max = slice_max
    #
    #         scores_dict[z_level] = [slice_min, slice_max]
    #         # print("Z: ", z_level, slice_min, slice_max)
    #         np.save(os.path.join(save_path_temp, 'overlay_{}'.format(z_level)), overlay)
    #
    #     scores_dict['all'] = [scores_min, scores_max]
    #     print("ALL ", scores_min, scores_max)
    #     ###################################
    #     # Blend attention map onto images #
    #     ###################################
    #     print("Blending attention map onto images")
    #     img_list = []
    #     z_levels_list = []
    #
    #     if isinstance(cmap, str):
    #         cmap = plt.get_cmap(cmap)
    #
    #     for i, z_level in enumerate(tqdm(z_unique_list)):
    #         indices = np.flatnonzero(z_list == z_level)
    #         coords_lev = coords[indices]
    #         scores_lev = scores[indices]
    #
    #         z_inner_start = max(0, z_level)
    #         # Iterate only until the next unique z index
    #         if i == len(z_unique_list) - 1:
    #             z_inner_end = z
    #         else:
    #             z_inner_end = z_unique_list[i+1]
    #
    #         # Comb through all the sublevels within 3D patch
    #         for z_level_inner in range(z_inner_start, z_inner_end):
    #             overlay = np.load(os.path.join(save_path_temp, 'overlay_{}.npy'.format(z_level_inner)))
    #             if cmap_normalize == 'all':
    #                 overlay_cvt = (overlay - scores_min) / (scores_max - scores_min)    # Convert the scores to [0, 1] for visualization
    #             elif cmap_normalize == 'slice':
    #                 if z_level_inner in scores_dict.keys():
    #                     s_min, s_max = scores_dict[z_level_inner]
    #                 else:
    #                     s_min = scores_min
    #                     s_max = scores_max
    #                 overlay_cvt = (overlay - s_min) / (s_max - s_min)
    #             else:
    #                 raise NotImplementedError
    #
    #             if blur:
    #                 overlay_cvt = cv2.GaussianBlur(overlay_cvt, tuple((patch_size[1:] * (1 - overlap)).astype(int) * 2 + 1), 0)
    #
    #             if segment:
    #                 tissue_mask = self.get_seg_mask(self.contours_tissue[z_level_inner],
    #                                                 self.holes_tissue[z_level_inner],
    #                                                 region_size,
    #                                                 scale[:2],
    #                                                 use_holes=use_holes,
    #                                                 offset=tuple(top_left))
    #
    #             img = np.array(Image.new(size=region_size, mode="RGB", color=(255, 255, 255)))
    #
    #             for idx in range(len(coords_lev)):
    #                 score = scores_lev[idx]
    #                 coord = coords_lev[idx]
    #
    #                 if score >= threshold:
    #                     # attention block
    #                     attn_block = overlay_cvt[coord[2]:coord[2] + patch_size[2], coord[1]:coord[1] + patch_size[1]]
    #                     color_block = (cmap(attn_block) * 255)[:, :, :3].astype(np.uint8) # color block (cmap applied to attention block)
    #
    #                     if segment:
    #                         img_block = img[coord[2]:coord[2] + patch_size[2], coord[1]:coord[1] + patch_size[1]].copy()
    #                         mask_block = tissue_mask[coord[2]:coord[2] + patch_size[2], coord[1]:coord[1] + patch_size[1]]
    #                         img_block[mask_block] = color_block[mask_block] # copy over only tissue masked portion of color block
    #                         pass
    #                     else:
    #                         # copy over entire color block
    #                         img_block = color_block
    #
    #                     # rewrite image block with scores
    #                     img[coord[2]:coord[2] + patch_size[2], coord[1]:coord[1] + patch_size[1]] = img_block.copy()
    #
    #             if blur:
    #                 img = cv2.GaussianBlur(img, tuple((patch_size[1:] * (1 - overlap)).astype(int) * 2 + 1), 0)
    #
    #             if alpha < 1.0:
    #                 img = self.block_blending(img_obj,
    #                                           img,
    #                                           z_level_inner,
    #                                           vis_level,
    #                                           top_left,
    #                                           bot_right,
    #                                           clip_min=clip_min,
    #                                           clip_max=clip_max,
    #                                           alpha=alpha,
    #                                           blank_canvas=blank_canvas,
    #                                           block_size=1024)
    #
    #             img = Image.fromarray(img)
    #             w, h = img.size
    #
    #             if custom_downsample > 1:
    #                 img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))
    #
    #             if max_size is not None and (w > max_size or h > max_size):
    #                 resizeFactor = max_size / w if w > h else max_size / h
    #                 img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))
    #
    #             img_list.append(img)
    #             z_levels_list.append(z_level_inner)
    #
    #     flist = glob(save_path_temp + '/*')
    #     for fname in flist:
    #         os.remove(fname)
    #
    #     return img_list, z_levels_list


    # def get_seg_mask(self, contours_tissue, contours_holes, region_size, scale, use_holes=False, offset=(0, 0, 0)):
    #     tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
    #     # contours_tissue = self.scaleContourDim(self.contours_tissue, scale)
    #     offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))
    #
    #     if len(contours_tissue) > 0:
    #         contours_tissue, contours_holes = zip(*sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))
    #         for idx in range(len(contours_tissue)):
    #             cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1), offset=offset, thickness=-1)
    #
    #             if use_holes:
    #                 cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0), offset=offset, thickness=-1)
    #             # contours_holes = self._scaleContourDim(self.holes_tissue, scale, holes=True, area_thresh=area_thresh)
    #
    #     tissue_mask = tissue_mask.astype(bool)
    #
    #     return tissue_mask