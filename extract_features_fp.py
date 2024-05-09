"""
Script for extracting features from the raw volumetric data (without saving the patches)
This is useful when low on storage space

If feature extraction from existing patches is desired, use extract_features.py
"""

import argparse
import os
from tqdm import tqdm

import h5py
import pandas as pd
import time

import torch
import torch.nn as nn
from models.feature_extractor import get_extractor_model
from utils.exp_utils import update_config
from utils.feature_utils import extract_patch_features_raw, extract_patch_raw, extract_features_raw
from utils.heatmap_utils import initialize_img
from utils.file_utils import save_hdf5
from data.transforms import get_basic_data_transforms

from preprocess.wsi_core.img_utils import identify_image_thresholds

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def extract_features(conf,
                     seg_params={'seg_level': 0, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False},
                     filter_params={'a_t': 100, 'a_h': 16, 'max_n_holes': 10},
                     patch_params={'use_padding': True, 'contour_fn': 'four_pt_easy'},
                     img_params={'clip_min': 0, 'clip_max': 1, 'black_thresh': 0},
                     ):

    if conf['patch_mode'] == '2D':
        patch_size = (conf['patch_size'], ) * 2
        step_size = (conf['step_size'], ) * 2
    elif conf['patch_mode'] == '3D':
        patch_size = (conf['patch_size_z'], conf['patch_size'], conf['patch_size'])
        step_size = (conf['step_z'], conf['step_size'], conf['step_size'])

    print("\n================")
    print('Loading model...')
    model = get_extractor_model(encoder=conf['encoder'],
                                mode=conf['patch_mode'],
                                input=patch_size)
    model.load_weights(**conf['pretrained'])

    model.eval()
    model = model.to(device)
    print(model)

    channel = model.get_channel_dim()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    aug_suffix = '_aug' if conf['augment_fold'] > 0 else ''

    ## Define subfolder name
    if conf['pretrained']['load_weights']:
        subfolder = conf['pretrained']['pretrained_name'] + aug_suffix
    else:
        subfolder = 'random' + aug_suffix

    feats_h5_subdir = os.path.join(conf['extracted_dir'],
                                   '{}_h5_patch_features'.format(conf['encoder']),
                                   subfolder)

    os.makedirs(feats_h5_subdir, exist_ok=True)

    ### Setting Up For-Loop for patch creation/feature extraction
    df = pd.read_csv(conf['process_list'])
    total = len(df)
    pbar_stack = tqdm(range(total))

    print(df)

    print("=============================")
    print("\nBeginning {} patch extraction with {} settings".format(conf['patch_mode'],
                                                                    conf['data_mode']))

    ######################
    # Feature extraction #
    ######################
    for i in pbar_stack:
        idx = df.index[i]

        slide_id = df.loc[idx, 'slide_id']
        slide_path = os.path.join(conf['data_dir'], slide_id)

        # Check whether the slide needs to be processed
        IS_PROCESSED = df.loc[idx, 'process'] == 0
        TO_SKIP = df.loc[idx, 'process'] == -1
        IS_FAILURE = df.loc[idx, 'process'] == -2

        if TO_SKIP or IS_FAILURE or IS_PROCESSED:
            if TO_SKIP:
                df.loc[idx, 'status'] = 'skip'
            elif IS_FAILURE:
                df.loc[idx, 'status'] = 'failure'
            else:
                df.loc[idx, 'status'] = 'proccessed'

            df.loc[idx, 'bag_size'] = 0
            continue

        # Load params
        current_filter_params = {}
        current_seg_params = {}
        current_patch_params = {}
        current_img_params = {}

        for key in filter_params.keys():
            current_filter_params.update({key: df.loc[idx, key] if key in df.columns else filter_params[key]})
        for key in seg_params.keys():
            current_seg_params.update({key: df.loc[idx, key] if key in df.columns else seg_params[key]})
        for key in patch_params.keys():
            current_patch_params.update({key: df.loc[idx, key] if key in df.columns else patch_params[key]})
        for key in img_params.keys():
            current_img_params.update({key: df.loc[idx, key] if key in df.columns else img_params[key]})

        print('\nSubject: {}'.format(slide_id))
        print('Initializing Three Dim object')
        if conf['mask_dir'] is None:
            mask_file = None
            binarize = True
            segment = True
        else:
            mask_file = os.path.join(conf['mask_dir'], slide_id, 'segmentation.pkl')
            binarize = False
            segment = False

        # Load and segment image
        ThreeDim_object = initialize_img(slide_path,
                                         patch_mode=conf['patch_mode'],
                                         binarize=binarize,
                                         segment=segment,
                                         seg_mask_path=mask_file,
                                         img_params=current_img_params,
                                         seg_params=current_seg_params,
                                         filter_params=current_filter_params)

        s = time.time()
        clip_max, clip_min = identify_image_thresholds(ThreeDim_object.wsi.img,
                                                       clip_min=current_img_params['clip_min'],
                                                       clip_max=current_img_params['clip_max'],
                                                       thresh_mode=conf['thresh_mode'])

        print("Took {} seconds to find min/max".format(time.time() - s))

        # Patch dataset
        roi_dataset = extract_patch_raw(img_obj=ThreeDim_object,
                                        patch_size=patch_size,
                                        step_size=step_size,
                                        batch_size=conf['batch_size'],
                                        data_mode=conf['data_mode'],
                                        slice_mode=conf['slice_mode'],
                                        clip_min=clip_min,
                                        clip_max=clip_max
                                        )

        ######################
        # Feature extraction #
        ######################
        print("\nFeature extraction started")
        for aug_idx in range(conf['augment_fold'] + 1):
            if aug_idx == 0:
                aug_suffix = ''
            else:
                aug_suffix = '_aug{}'.format(aug_idx)

            df.to_csv(os.path.join(conf['extracted_dir'], conf['process_list']), index=False)
            feats_h5_path = os.path.join(feats_h5_subdir, slide_id + aug_suffix + '.h5')

            if os.path.isfile(feats_h5_path):
                print(feats_h5_path + " exists!")
                try:
                    _ = h5py.File(feats_h5_path, "r")
                    df.loc[idx, 'status'] = 'processed'
                    continue
                except OSError:
                    print('Error Opening %s' % feats_h5_path)
                    os.system('rm %s' % feats_h5_path)
                    df.loc[idx, 'process'] = -2
                    df.loc[idx, 'status'] = 'failure'

            # Feature extraction
            augment = False if aug_idx == 0 else True
            trns = get_basic_data_transforms(augment=augment, data_mode=conf['data_mode'], invert=conf['invert'])
            roi_dataset.set_transforms(trns)
            print("Extracting from ", slide_id,
                  " for aug ", aug_idx, " with ", clip_min, clip_max,
                  " (aug: {})".format(augment))

            asset_dict = extract_features_raw(roi_dataset,
                                              feature_extractor=model,
                                              batch_size=conf['batch_size'],
                                              channel=channel,
                                              device=device)

            save_hdf5(feats_h5_path, asset_dict, mode='w')  # Save extracted features/coords


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--data_dir', type=str, help='The root project folder directory. \
        We assume that for most projects, you would want your extracted features to live in the same directory as your WSIs.')
    parser.add_argument('--save_dir', type=str, help='Folder to save extracted results for patching, tissue segmentation, and stitching. \
        By default, we assume args.extracted_dir is a directory within args.datroot. However, passing an absolute path into args.extracted_dir (by checking \
        if "/" is in args.extracted_dir) will override using args.dataroot as a root path.')
    parser.add_argument('--mask_dir', type=str, help="If segmentation file already exists, use it")
    parser.add_argument('--process_list', type=str,
                        help='name of list of images to process with parameters (.csv)')
    parser.add_argument('--config', type=str, default='.',
                        help='Config files that contain default parameters')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--patch_size_z', type=int)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--step_z', type=int,
                        help='Step size along z-direction. This allows for non-isotropic step size along z direction. If None, same as step_size')
    parser.add_argument('--patch_mode', choices=['2D', '3D'], type=str,
    					help='2D patching or 3D patching')
    parser.add_argument('--clip_min', type=int)
    parser.add_argument('--clip_max', type=int)
    parser.add_argument('--encoder', type=str,
                        help='cnn feature extractor to use')
    parser.add_argument('--augment_fold', default=5, type=int,
                        help='Number of augmentations to perform')
    parser.add_argument('--data_mode', type=str,
                        help='The input device mode, e.g., CT, OTLS')
    parser.add_argument('--thresh_mode', type=str, choices=['fixed', 'global'])
    parser.add_argument('--slice_mode', type=str)
    parser.add_argument('--invert', action='store_true', default=False,
                        help='Whether to invert intesinty or not')

    args = parser.parse_args()
    # Update args namespace with parameters in config file
    conf = update_config(args)
    print("\nPARAMETERS: ", conf)

    if conf['patch_size_z'] is None:
        conf['patch_size_z'] = conf['patch_size']

    if conf['step_z'] is None:
        conf['step_z'] = conf['step_size']

    conf['extracted_dir'] = os.path.join(conf['save_dir'], 'patch_{}-{}_step_{}-{}_{}_{}_{}'.format(conf['patch_size'],
                                                                                                 conf['patch_size_z'],
                                                                                                 conf['step_size'],
                                                                                                 conf['step_z'],
                                                                                                 conf['patch_mode'],
                                                                                                 conf['slice_mode'],
                                                                                                 conf['thresh_mode']))
    os.makedirs(conf['extracted_dir'], exist_ok=True)

    if conf['patch_mode'] == '3D' and conf['batch_size'] >= 100:
        print("*************************************")
        print("WARNING: Make sure you are using pytorch 2.0 for large batch size of 3D inputs!")

    extract_features(conf)
