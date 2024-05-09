"""
There are several types of datasets for 3D framework

ThreeDimPatchBag
- Self-supervised learning

ThreeDimImageBag
- Bag of patches for a single subject (Wrapper)
- Feature extraction

ThreeDimFeatsBag
- Bag of extracted features & patient-level label for the cohort
- MIL training

"""

import os
from torch.utils.data import Dataset

from preprocess.wsi_core.img_utils import clip_and_normalize_img
from utils.contour_utils import get_contour_check_fn

import h5py
import torch
import numpy as np

def transform_img(img,
                  clip_min=0,
                  clip_max=1,
                  transform=None):
    """
    Normalize and transform the input. Assume that the last dimension is the channel dimension
    """

    img = clip_and_normalize_img(img,
                                 clip_min=clip_min,
                                 clip_max=clip_max)

    if transform:
        img = transform(img)
    else:
        raise NotImplementedError("Not implemented yet!")

    return img


############
# Datasets #
############
class ResponseMixin:
    def get_y(self, name):
        if self.task == 'clf':
            #TODO: Work on general case
            y_out = int(self.data_df.loc[name, 'event'])

        elif self.task == 'surv':    # survival
            y = self.data_df.loc[name, 'event_days']
            c = self.data_df.loc[name, 'event']    # non-censorship
            time_bin = self.data_df.loc[name, 'bins']

            y_out = [y, c, time_bin]

        else:
            raise NotImplementedError("Not implemented for task ", self.task)

        y_out = np.array(y_out)
        return torch.from_numpy(y_out)


class ThreeDimFeatsBag(Dataset, ResponseMixin):
    """
    This dataset loads the extracted features of the 3D dataset
    """

    def __init__(self,
                 path,
                 data_df,
                 task='clf',
                 sample_prop=1,
                 sample_mode='volume',
                 numOfaug=0,
                 numOfclasses=2):

        super().__init__()
        self.task = task
        self.data_df = data_df
        self.path = path
        self.fnames = self._get_flist()
        self.sample_prop = sample_prop
        self.numOfclasses = numOfclasses
        self.sample_mode = sample_mode
        self.numOfaug = numOfaug

    def _get_flist(self):

        df_flist = self.data_df.index.values
        flist = df_flist

        return flist

    def __len__(self):
        return len(self.fnames) // (self.numOfaug + 1)

    def __getitem__(self, index):
        """
        Returns a tuple of (bag of patches, label)
        This retrieves all different slides belonging to the same patient.
        """
        # For each subject, choose random index to choose augmented subject (features extracted/augmented offline)
        offset_aug = np.random.choice(self.numOfaug + 1, 1)[0]
        index_aug = (self.numOfaug + 1) * index + offset_aug

        fname = self.fnames[index_aug]
        # Retrieve all the slides belonging to the patient
        slides = [s.strip() for s in self.data_df.at[fname, 'slide_id'].split(',')]    # Slide ids for each patient

        features_bag = []
        coords_bag = []
        for slide_idx, slide in enumerate(slides):

            if not slide or slide == 'nan' or slide == '':
                slide_name = fname
            else:
                fname_list = fname.split('_')
                if len(fname_list) > 1:
                    slide_ext = '-{}_'.format(slide) + fname_list[-1]    # '24367--A_aug5'
                else:
                    slide_ext = '-{}'.format(slide)
                
                slide_name = fname_list[0] + slide_ext

            fpath = os.path.join(self.path, slide_name) + '.h5'
            file = h5py.File(fpath, 'r')

            features = np.array(file['features'])
            coords = np.array(file['coords'], dtype=float)

            # Sample slices
            if self.sample_mode == 'seq_num':
                z_levels_unique = np.unique(coords[:, 0])
                numOfslices = len(z_levels_unique)
                numOfsamples = self.sample_prop

                indices = []
                levels_selected = np.random.choice(numOfslices, size=numOfsamples, replace=False)
                for lev in levels_selected:
                    indices.append(np.flatnonzero(coords[:, 0] == z_levels_unique[lev]))

                indices = np.concatenate(indices)
                features = features[indices]
                coords = coords[indices]
            else:
                # Sample random patches (Form of augmentation)
                if self.sample_prop < 1:
                    if self.sample_mode == 'volume':    # Sample randomly throughout volume
                        numOfinstances = len(features)
                        numOfsamples = int(numOfinstances * self.sample_prop)
                        indices = np.random.choice(numOfinstances, size=numOfsamples, replace=False)

                    elif self.sample_mode == "seq": # Sample along slice axis
                        z_levels_unique = np.unique(coords[:, 0])
                        numOfslices = len(z_levels_unique)
                        numOfsamples = int(numOfslices * self.sample_prop)
                        levels_selected = np.random.choice(numOfslices, size=numOfsamples, replace=False)

                        indices = []
                        for lev in levels_selected:
                            indices_temp = np.flatnonzero(coords[:, 0] == lev)
                            indices.append(indices_temp)
                        indices = np.concatenate(indices)

                    else:   # Sample proportionally from each slice
                        z_levels_unique = np.unique(coords[:, 0])
                        indices = []
                        for lev in z_levels_unique:
                            indices_temp = np.flatnonzero(coords[:, 0] == lev)
                            numOfsamples = int(len(indices_temp) * self.sample_prop)
                            indices.append(np.random.choice(indices_temp, size=numOfsamples, replace=False))

                        indices = np.concatenate(indices)
                    features = features[indices]
                    coords = coords[indices]

            features = torch.from_numpy(features)     # (numOfbags, z, w, h)
            coords = torch.from_numpy(coords)

            # Trick to treat z coordinates from different slides differently.
            # This is required for hierarchical attention
            # e.g.) Level 1 from patient A slide a: 1.1, Level 1 from Patent A slide b: 1.2
            # print("==============")
            # print(coords[:, 0], (slide_idx + 1) / 10)
            coords[:, 0] = coords[:, 0] + (slide_idx + 1) / 10

            # Gather in the patient bag
            features_bag.append(features)
            coords_bag.append(coords)

        features_bag = torch.cat(features_bag)
        coords_bag = torch.cat(coords_bag)
        # print(features_bag.shape)
        # print("==========")
        # Corresponding target
        y = self.get_y(fname)
        file.close()

        return index, features_bag, coords_bag, y


class ImgBag(Dataset):
    """
    Wrapper to load the image patches within each subject (For feature extraction)
    """
    def __init__(self,
                 file_path,
                 patch_mode='3D',
                 transform=None,
                 clip_min=0,
                 clip_max=65000):
        """
        Args:
        - file_path (string): Path to the .h5 file containing patched data.
        - transform (callable, optional): Optional transform to be applied on a sample
        - norm_min, norm_max (int): min and max intensity of the image

        Returns:
        - img (Tensor): 2D or 3D patch
        - coord: Coordinate of (z, x, y)
        """

        self.transform = transform
        self.file_path = file_path

        self.clip_min = clip_min
        self.clip_max = clip_max

        with h5py.File(self.file_path, "r") as f:
            img = f['imgs'][()]
            coords = f['coords'][()]

        self.img = img
        self.coords = coords
        self.length = len(self.img)

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = transform_img(self.img[idx],
                            clip_min=self.clip_min,
                            clip_max=self.clip_max,
                            transform=self.transform).float()

        return img, self.coords[idx]


class SimpleFeatsBag(Dataset):
    """
    Simple dataset to batchify input to prevent OOM
    """

    def __init__(self, features, coords):
        self.features = features
        self.coords = coords

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.coords[idx]

class RawImgBag(Dataset):
    """
    This is a wrapper dataset for bag of patches.
    Required for both heatmap visualization & end-to-end training

    Inputs
    ======
    img_object: instance of ThreeDim_object
    top_left: tuple of coordinates representing the top left corner of WSI region (Default: None)
    bot_right tuple of coordinates representing the bot right corner of WSI region (Default: None)
    level: downsample level at which to prcess the WSI region
    patch_size: tuple of width, height representing the patch size
    step_size: tuple of w_step, h_step representing the step size
    contour_fn (str):
        contour checking fn to use
        choice of ['four_pt_hard', 'four_pt_easy', 'center', 'basic'] (Default: 'four_pt_hard')
    t: custom torchvision transformation to apply
    custom_downsample (int): additional downscale factor to apply
    use_center_shift: for 'four_pt_hard' contour check, how far out to shift the 4 points
    """

    def __init__(self,
                 img_object,
                 patch_level,
                 slice_mode='all',
                 patch_size=(96, 96, 96),
                 step_size=(96, 96, 96),
                 patch_params={},
                 clip_min=0,
                 clip_max=1,
                 area_thresh=0.5,
                 transforms=None,
                 contour_fn='four_pt_easy',
                 top_left=None, bot_right=None,
                 use_center_shift=False,
                 **kwargs):

        assert len(patch_size) == len(step_size), "Patch size and step size have to be equal dimensions"

        self.img_object = img_object
        self.patch_level = patch_level
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.transforms = transforms
        self.patch_size = patch_size

        cont_check_fn = get_contour_check_fn(contour_fn=contour_fn,
                                             patch_size=patch_size[1],
                                             step_size=step_size[1],
                                             use_center_shift=use_center_shift)

        # Process the contours and obtain coordinates for patches
        coords = self.img_object.process_contours(**patch_params,
                                                  save_path=None,
                                                  patch_level=patch_level,
                                                  patch_size=patch_size[1],
                                                  patch_size_z=patch_size[0],
                                                  step_size=step_size[1],
                                                  step_size_z=step_size[0],
                                                  cont_check_fn=cont_check_fn,
                                                  area_thresh=area_thresh,
                                                  save_patches=False,
                                                  mode=slice_mode,
                                                  verbose=False)

        self.coords = coords
        print('\nFiltered a total of {} patches'.format(len(self.coords)))

    def set_transforms(self, transforms):
        self.transforms = transforms

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        img = self.img_object.wsi.read_region(tuple(coord),
                                              self.patch_level,
                                              self.patch_size)

        img = transform_img(img,
                            clip_min=self.clip_min,
                            clip_max=self.clip_max,
                            transform=self.transforms)

        return img.float(), coord