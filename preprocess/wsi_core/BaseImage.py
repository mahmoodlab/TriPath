"""
DONE
"""

import cv2
import numpy as np
from PIL import Image
import PIL
import pickle
from abc import ABC
import os

from .img_utils import read_img, process_raw_img

PIL.Image.MAX_IMAGE_PIXELS = 9000000000


class ThreeDimObject:
    """
    Always have the first dimension of img (z) be the traversing direction.
    The grayscale image is saved in uint16 format, since CT scans come in uint16,
    and provides more diverse detail than uint8.

    Args:
    - path (str): path for raw images
    - view_axis (int): The depth axis. There are three axes to choose from
    - black_thresh (int): Slides whose mean values are below black_thresh will be removed (computational efficiency)
    """
    def __init__(self,
                 path,
                 view_axis=0,
                 black_thresh=0):

        self.path = path
        self.view_axis = view_axis

        img_arr, img_info = read_img(path, black_thresh)

        if self.view_axis == 1:
            img_arr = img_arr.transpose(1, 0, 2)
        elif self.view_axis == 2:
            img_arr = img_arr.transpose(2, 0, 1)

        self.img = img_arr
        z, h, w, c = self.img.shape

        self.level_dimensions = [(w, h)]
        self.level_depth = z
        self.channel = c
        self.dtype = self.img.dtype
        self.z_start, self.z_end = img_info['z_levels']
        self.resolution = img_info['resolution']

    def read_region(self, location, level, size):
        """
        Reads a stack of 2D images from the data.
        Emulates read_region method of the Openslide package

        Args:
        - location (tuple): Tuple of starting locations in (z, x, y) format
        - level (int): Zoom level to read data. Always set to 1 for now
        - size (tuple): Size for image - 2D: (width, height) or 3D: (depth, width, height)

        Returns:
        - img (np.ndarray): (w, h) if 2D or (z, w, h) if 3D
        """

        z, x, y = (int(loc) for loc in location)
        if len(size) == 2:  # Reading a 2D slice
            width, height = size
            img = self._read_region_single_slice(location, level, width, height)

        elif len(size) == 3:    # Reading a volume
            width, height = size[1:]

            patch_list = []
            for slice_idx in range(z, z + size[0]):
                temp_location = (slice_idx, x, y)
                if slice_idx < 0 or slice_idx >= self.img.shape[0]:
                    patch_list.append(np.zeros((width, height, self.channel), dtype=self.dtype))
                else:
                    patch_list.append(self._read_region_single_slice(temp_location,
                                                                     level,
                                                                     width,
                                                                     height))

            img = np.stack([patch for patch in patch_list])
        else:
            raise NotImplementedError("Wrong z_level input")

        return img

    def _read_region_single_slice(self, location, level, width, height):
        """
        Reads a 2D image from the data.

        Args:
        - location (tuple): Tuple of starting locations in (z, x, y) format
        - level (int): Zoom level to read data. Always set to 1 for now
        - width (int): Width of the image to be read
        - height (int): Height of the image to be read

        Returns:
        - img (np.ndarray): image with size of (w, h)
        """
        z, x, y = location

        if z >= self.img.shape[0]:  # If z-index is out of bounds
            img = np.zeros(self.img.shape[1:])
        else:
            img = self.img[z]   # (w, h, 1)

        # zero-pad if the patch goes out of bounds
        if x + width > img.shape[1]:
            pad_x = x + width - img.shape[1]
        else:
            pad_x = 0

        if y + height > img.shape[0]:
            pad_y = y + height - img.shape[0]
        else:
            pad_y = 0

        img = np.pad(img[y: y + height, x: x + width],
                     ((0, pad_y), (0, pad_x), (0, 0)),
                     'constant')
        return img


class BaseImage(ABC):
    """
    Abstract base class for storing raw image and their segmentation results.

    Args:
    - path (str): path to the data
    - black_thresh (int): Slides whose mean values are below black_thresh will be removed (computational efficiency)
    - z_start (int): Desired start index for depth dimension
    - depth (int): Desired depth
    """
    def __init__(self, path, black_thresh=18000, z_start=None, depth=None):

        self.name = os.path.basename(path)
        self.wsi = ThreeDimObject(path, black_thresh=black_thresh)

        if z_start is None:
            self.z_start = self.wsi.z_start
        else:
            self.z_start = z_start

        if depth is None:
            self.depth = self.wsi.level_depth
        else:
            self.depth = depth

        z_level_last = self.z_start + self.wsi.level_depth
        z_level_temp = self.z_start + self.depth

        z_end = z_level_temp if z_level_temp < z_level_last else z_level_last

        self.z_end = z_end

        self.seg_level = None
        # Array holding each slice's tissue contours
        # The length of the array will be number of slices
        self.contours_tissue = []
        # Array holding each slice's hole contours
        # The length of the array will be number of slices
        self.holes_tissue = []

    def saveSegmentation(self, mask_file):
        """
        Save segmentation results
        """

        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}

        with open(mask_file, 'wb') as f:
            pickle.dump(asset_dict, f)

    def loadSegmentation(self, mask_file):

        assert os.path.isfile(mask_file), "{} doesn't exist!".format(mask_file)

        with open(mask_file, 'rb') as f:
            info = pickle.load(f)

        self.contours_tissue = info['tissue']
        self.holes_tissue = info['holes']

    def block_blending(self,
                       img,
                       z_level=0,
                       vis_level=0,
                       top_left=0,
                       bot_right=0,
                       alpha=0.5,
                       blank_canvas=False,
                       block_size=1024,
                       clip_min=0,
                       clip_max=1):
        """
        Blends the raw 2D image (weighted by 1-alpha) with score-based heatmap (weighted by alpha)

        Args:
        - img (np.ndarray): 2D array of scores to be blended with the raw image
        """

        downsample = self.level_downsamples[vis_level]
        w = img.shape[1]
        h = img.shape[0]
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)

        shift = top_left  # amount shifted w.r.t. (0,0)
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):

                # 1. convert wsi coordinates to image coordinates via shift and scale
                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))

                # 2. compute end points of blend tile, careful not to go over the edge of the image
                y_end_img = min(h, y_start_img + block_size_y)
                x_end_img = min(w, x_start_img + block_size_x)

                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue

                # 3. fetch blend block and size
                blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
                blend_block_size = (x_end_img - x_start_img, y_end_img - y_start_img)
                if not blank_canvas:
                    # 4. read actual wsi block as canvas block
                    pt = (x_start, y_start)

                    # downsample original image and use as canvas
                    canvas = self.wsi.read_region((z_level,) + pt,
                                                  vis_level,
                                                  blend_block_size)
                    canvas = process_raw_img(canvas, clip_min, clip_max)
                else:
                    # 4. OR create blank canvas block
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255, 255, 255)))
                # 5. blend color block and canvas block
                img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block,
                                                                                    alpha,
                                                                                    canvas[..., ::-1],
                                                                                    1 - alpha,
                                                                                    0)

        return img


    ##############################################
    # Static helper methods for contour processing
    @staticmethod
    def getBestSlice(contours_list, mode='max'):
        """
        Determines the reference slice to start patching the volume along the depth dimension

        Inputs
        - contours_list (list): list of contours, where each element is a list of foreground contours
                                at the specific slice level
        - mode (str): 'max' or 'top'
            'max': Choose the slice where the contour area is the largest
            'top': choose the top slide
            'middle': choose the middle slide
        """
        numOflevels = len(contours_list)

        best_slice_idx = 0

        if mode == 'max':
            best_slice_area = 0

            for slice_idx in range(numOflevels):
                contours = contours_list[slice_idx]

                area = 0
                for contour in contours:
                    area += cv2.contourArea(contour)
                if area > best_slice_area:
                    best_slice_idx = slice_idx
                    best_slice_area = area

        elif mode == 'top':
            for slice_idx in range(numOflevels):
                if len(contours_list[slice_idx]) > 0:
                    best_slice_idx = slice_idx
                    break

        elif mode == 'middle':
            for slice_idx in range(int(numOflevels/2), numOflevels):
                if len(contours_list[slice_idx]) > 0:
                    best_slice_idx = slice_idx
                    break
        else:
            raise NotImplementedError("Not implemented!")

        return best_slice_idx

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                return 1

        return 0

    @staticmethod
    def isInContours(cont_check_fn, contour, pt, holes=None, patch_size=256):
        if cont_check_fn(contour, pt):
            if holes is not None:
                return not BaseImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]

    @staticmethod
    def isWhitePatch(patch, area_ratio=0.05, white_threshold=6e4):
        """
        Determine whether the patch is white (e.g. due to calcification), so that we can filter it out
        """
        area_thresh = patch.size * area_ratio

        return True if np.sum(patch > white_threshold) > area_thresh else False

    @staticmethod
    def isBlackPatch(patch, rgbThresh=40):
        return True if np.all(np.mean(patch, axis=(0,1)) < rgbThresh) else False
