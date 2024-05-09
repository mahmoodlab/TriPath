"""
Script for extracting features from the 3D patches, assuming that preprocess/create_patches_3D.py has already been run
For fast-processing version, refer to extract_patches_fp.py
"""

import argparse
import os
from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from models.feature_extractor import get_extractor_model
from utils.exp_utils import update_config
from utils.feature_utils import extract_patch_features
from data.ThreeDimDataset import ImgBag
from data.transforms import get_basic_data_transforms

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def extract_features(conf):
    """
    Extract features from the patches

    Args:
    - conf (dict)

    Returns:
    - None
    """
    print("\n================")
    print('Loading model...')

    if conf['target_patch_size_z'] is None:
        patch_size = (conf['target_patch_size'], ) * 3
    else:
        patch_size = (conf['target_patch_size_z'], conf['target_patch_size'], conf['target_patch_size'])
    print("Patch size ", patch_size)
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

    conf['extracted_dir'] = os.path.join(conf['dataroot'], conf['extracted_dir']) if (conf['extracted_dir'][0] != '/') else conf['extracted_dir']
    patches_subdir = os.path.join(conf['extracted_dir'], 'patches')

    ## Define subfolder name
    if conf['pretrained']['load_weights']:
        subfolder = conf['pretrained']['pretrained_name'] + aug_suffix
    else:
        subfolder = 'random' + aug_suffix

    feats_h5_subdir = os.path.join(conf['extracted_dir'],
                                   '{}_h5_patch_features'.format(conf['encoder']),
                                   subfolder)

    os.makedirs(feats_h5_subdir, exist_ok=True)

    ### Setting Up For-Loop for feature extraction
    df = pd.read_csv(os.path.join(conf['extracted_dir'], conf['process_list']))
    total = len(df)
    pbar_stack = tqdm(range(total))

    if 'process_features' not in df.columns:
        df['process_features'] = np.ones(total, dtype=np.int32)

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
        patches_h5_path = os.path.join(patches_subdir, slide_id + '_patches.h5')

        clip_min = df.loc[idx, 'clip_min'] if 'clip_min' not in conf else conf['clip_min']
        clip_max = df.loc[idx, 'clip_max'] if 'clip_max' not in conf else conf['clip_max']

        # Error-Handling in disrupted scripts
        IS_PROCESSED = df.loc[idx, 'process_features'] == 0
        TO_SKIP = df.loc[idx, 'process_features'] == -1
        IS_FAILURE = df.loc[idx, 'process_features'] == -2

        if TO_SKIP or IS_FAILURE or IS_PROCESSED:
            if TO_SKIP:
                df.loc[idx, 'status_features'] = 'skip'
            elif IS_FAILURE:
                df.loc[idx, 'status_features'] = 'failure'
            else:
                df.loc[idx, 'status_features'] = 'proccessed'

            df.loc[idx, 'bag_size'] = 0
            continue

        if not os.path.isfile(patches_h5_path):
            df.loc[idx, 'status_features'] = 'skip'
            df.loc[idx, 'bag_size'] = 0
            print('Could not find patch file for: %s' % patches_h5_path)
            continue

        # If no issue, proceed with patch loading
        img_dataset = ImgBag(file_path=patches_h5_path,
                                patch_mode=conf['patch_mode'],
                                clip_min=clip_min,
                                clip_max=clip_max)

        # Augmentation loop
        for aug_idx in range(conf['augment_fold'] + 1):
            if aug_idx == 0:
                aug_suffix = ''
            else:
                aug_suffix = '_aug{}'.format(aug_idx)

            feats_h5_path = os.path.join(feats_h5_subdir, slide_id + aug_suffix + '.h5')

            if os.path.isfile(feats_h5_path):
                print(feats_h5_path + " exists!")
                try:
                    _ = h5py.File(feats_h5_path, "r")
                    df.loc[idx, 'status_features'] = 'processed'
                    continue
                except OSError:
                    print('Error Opening %s' % feats_h5_path)
                    os.system('rm %s' % feats_h5_path)
                    df.loc[idx, 'process_features'] = -2
                    df.loc[idx, 'status_features'] = 'failure'

            # Feature extraction
            print("Extracting features from ", slide_id, " for aug ", aug_idx, " with ", clip_min, clip_max)

            data_transforms = get_basic_data_transforms(augment=False if aug_idx == 0 else True,
                                                        patch_mode=conf['patch_mode'],
                                                        data_mode=conf['data_mode'],
                                                        invert=conf['invert'])
            img_dataset.set_transform(data_transforms)

            extract_patch_features(dataset=img_dataset,
                                   output_path=feats_h5_path,
                                   model=model,
                                   model_name=conf['encoder'],
                                   batch_size=conf['batch_size'],
                                   leave=bool(idx == len(pbar_stack) - 1),
                                   channel=channel,
                                   device=device)

            if aug_idx == conf['augment_fold']:
                df.loc[idx, 'process_features'] = 0
                df.loc[idx, 'status_features'] = 'processed'

            df.to_csv(os.path.join(conf['extracted_dir'], conf['process_list']), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--dataroot', type=str, help='The root project folder directory. \
        We assume that for most projects, you would want your extracted features to live in the same directory as your WSIs.')
    parser.add_argument('--extracted_dir', type=str, default=None, help='Folder to save extracted results for patching, tissue segmentation, and stitching. \
        By default, we assume args.extracted_dir is a directory within args.datroot. However, passing an absolute path into args.extracted_dir (by checking \
        if "/" is in args.extracted_dir) will override using args.dataroot as a root path.')
    parser.add_argument('--config', type=str, default='.',
                        help='Config files that contain default parameters')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--patch_mode', default='2D', choices=['2D', '3D'], type=str,
    					help='2D patching or 3D patching')
    parser.add_argument('--clip_min', type=int)
    parser.add_argument('--clip_max', type=int)
    parser.add_argument('--process_list', type=str, default='process_list_extract.csv',
                        help='name of list of images to process with parameters (.csv)')
    parser.add_argument('--encoder', type=str,
                        help='cnn feature extractor to use')
    parser.add_argument('--target_patch_size_z', type=int, default=96,
                        help='the desired size of patches for optional scaling before feature embedding')
    parser.add_argument('--target_patch_size', type=int, default=96,
                        help='the desired size of patches for optional scaling before feature embedding')
    parser.add_argument('--augment_fold', default=5, type=int,
                        help='Number of augmentations to perform')
    parser.add_argument('--data_mode', type=str,
                        help='The input device mode, e.g., CT, OTLS')
    parser.add_argument('--invert', action='store_true', default=False,
                        help='Whether to invert intesinty or not')

    args = parser.parse_args()
    conf = update_config(args) # Update args namespace with parameters in config file
    print("\nPARAMETERS: ", conf)

    if conf['patch_mode'] == '3D' and conf['batch_size'] >= 100:
        print("*************************************")
        print("WARNING: Make sure you are using pytorch 2.0 for large batch size of 3D inputs!")

    extract_features(conf)
