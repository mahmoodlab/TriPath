"""
Helper functions for heatmap generation
"""

import cv2
import os
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from preprocess.wsi_core.SerialTwoDimImage import SerialTwoDimImage
from preprocess.wsi_core.ThreeDimImage import ThreeDimImage
from preprocess.wsi_core.img_utils import clip_and_normalize_img
from scipy.stats import percentileofscore
from models.head import get_decoder_model
from PIL import Image
from utils.file_utils import save_hdf5
from tqdm import tqdm
from captum.attr import IntegratedGradients

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## Heatmap generation
def drawHeatmap(scores,
                coords,
                img_obj=None,
                segment=True,
                patch_size=(96, 96, 96),
                vis_level=-1,
                save_path_temp=None,
                cmap_normalize='slice',
                cmap_min=-2,
                cmap_max=2,
                **kwargs):
    """
    Draw heatmap based on attribution scores and coordinates

    Inputs:
    - save_path_temp (str): temporary path to save heatmaps (removed after completion)

    Returns:
    - heatmap_list (list of Pillow images): List containing final heatmaps
    - z_levels_list (list of int): List of heatmap z levels
    """

    heatmap_list, z_levels_list = img_obj.visHeatmap(scores=scores,
                                                     coords=coords,
                                                     vis_level=vis_level,
                                                     patch_size=patch_size,
                                                     segment=segment,
                                                     save_path_temp=save_path_temp,
                                                     cmap_normalize=cmap_normalize,
                                                     cmap_min=cmap_min,
                                                     cmap_max=cmap_max,
                                                     **kwargs)
    return heatmap_list, z_levels_list


def saveHeatmap(slide_save_dir, slide_id, heatmap_list, z_levels_list, down_factor=1, video=False):
    """
    Save heatmap

    Args:
    - down_factor (int): Downsampling factor along xy plane (for saving memory)
    - video (bool): Flag for saving video
    """

    print("\nSaving heatmaps...")
    slide_save_dir_indiv = os.path.join(slide_save_dir, slide_id)
    os.makedirs(slide_save_dir_indiv, exist_ok=True)

    heatmap_list_resized = []
    for heatmap in heatmap_list:
        w, h, _ = np.array(heatmap).shape
        heatmap_resized = cv2.resize(np.array(heatmap), (h // down_factor, w // down_factor))
        heatmap_resized = Image.fromarray(heatmap_resized)
        heatmap_list_resized.append(heatmap_resized)

    for j, (heatmap, z_level) in enumerate(tqdm(zip(heatmap_list_resized, z_levels_list))):
        heatmap.save(
            os.path.join(slide_save_dir_indiv, '{}_lev_{}.png'.format(slide_id, z_level)))

    if video:
        write_video(heatmap_list_resized,
                    video_length=15,
                    save_fpath=os.path.join(slide_save_dir_indiv, 'heatmaps.avi'))


def initialize_img(path,
                   patch_mode='3D',
                   binarize=True,
                   segment=True,
                   seg_mask_path=None,
                   seg_params={'seg_level': 0},
                   img_params={},
                   filter_params=None):
    """
    Create the ThreeDim object and segment the tissues.
    Can be sped up with with pre-segmented tissue


    Returns:
    - ThreeDim_object: Object with segmented tissue information
    """

    if segment:
        assert binarize, "To segment, you need to binarize the image first"

    if patch_mode == '2D':
        ThreeDim_object = SerialTwoDimImage(path,
                                            binarize=binarize,
                                            **img_params,
                                            **seg_params)
    elif patch_mode == '3D':
        ThreeDim_object = ThreeDimImage(path,
                                        binarize=binarize,
                                        **img_params,
                                        **seg_params)
    else:
        raise NotImplementedError("Not implemented")

    if seg_params['seg_level'] < 0:
        best_level = ThreeDim_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    if segment: # Segment the tissue
        print('\nSegmenting the tissue ...')
        ThreeDim_object.segmentTissueSerial(**seg_params,
                                            filter_params=filter_params)
    else:   # Load pre-segmented information
        if seg_mask_path is None:
            print('\nSegmentation mask does not exist. Not loading ..')
        else:
            print('\nLoading segmentation results ...')
            ThreeDim_object.loadSegmentation(seg_mask_path)

    return ThreeDim_object


def encode_features(model,
                    features):
    """
    Encode the features with shallow MLP

    Args:
    - features (numOfbatches, numOfinstances, latent_dim)

    Returns:
    - features_enc (numOfbatches, numOfinstances, decoder_enc_dim)
    """
    features_enc = model.encode(features)
    return features_enc


def attend_features(model,
                    features_enc,
                    coords):

    _, features_attn = model.attend(features_enc, coords)

    return features_attn


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


def infer_block(model,
                features,
                coords):
    """
    Given the slide feature, estimate attention scores

    Args:
    - model: Attention model
    - features (numOfinstances, latent_dim)

    Returns:
    - attn_dict: Dictionary containing attn/ig scores
    """
    with torch.no_grad():
        out, attn_dict = model(features, coords)
        attn_intra = attn_dict['intra'].view(-1, 1).cpu().numpy()
        attn_inter = attn_dict['inter'].view(-1, 1).cpu().numpy()
        del attn_dict

        prob_pos = sigmoid(out.detach().cpu().numpy()).reshape(-1, 1)
        prob = np.concatenate([1 - prob_pos, prob_pos], axis=1)

        def interpret_patient(features):
            return model.captum(x=features)

        # Integrated gradient
        ig = IntegratedGradients(interpret_patient)
        features.requires_grad_()

        for target in range(1):
            ig_attr = ig.attribute((features), n_steps=50, target=target, internal_batch_size=len(features))
            # pdb.set_trace()
            ig_attr = ig_attr.squeeze().sum(dim=1).cpu().detach()

        ig_attr = ig_attr.view(-1, 1)
        # ig_normalized = normalize_ig_scores(ig_attr).cpu().numpy()

        probs = F.softmax(out, dim=1).cpu().flatten()

    score_dict = {
                    'intra': attn_intra,
                    'inter': attn_inter,
                    'ig': ig_attr.numpy()
                 }

    return probs, score_dict


def initiate_attn_model(model_dict, ckpt_path=None, verbose=True):
    """
    Initiate attention model to compute attention scores

    Args:
    - model_dict (Dict): Model dictionary containing attention model parameters
    - ckpt_path (str): Model checkpoint path

    Returns:
    - model (nn.Module): Attention module
    """
    if verbose:
        print('\nInitializing attention model...')
        print('From path ', ckpt_path)
        print(model_dict)
    model = get_decoder_model(**model_dict)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}

    for key in ckpt.keys():
        ckpt_clean.update({key.replace('decoder.', ''): ckpt[key]})

    model.load_state_dict(ckpt_clean, strict=True)
    model.eval()
    return model


def attend_whole(feats_path_list=[],
                 attn_model=None,
                 label=None,
                 label_dict={},
                 device=None,
                 **block_kwargs):
    """
   Given a bag of features, pass them through attention network

   Returns:
   - patient_results: List of result dictionaries, where each dictionary contains features/scores per block
   """
    numOfslides = len(feats_path_list)

    patient_results = []
    counter = [0]  # List to store start/end indices for each block
    feats_patient = []
    coords_patient = []

    for feats_path in feats_path_list:
        print("\nLoading pre-extracted features from {}".format(feats_path))
        assert os.path.isfile(feats_path), "{} does not exist!".format(feats_path)
        with h5py.File(feats_path, 'r') as f:
            features = f['features'][()]
            coords = f['coords'][()]
        counter.append(counter[-1] + len(features))

        feats_patient.append(features)
        coords_patient.append(coords)

    feats_patient = np.concatenate(feats_patient)
    print("\nAttending to total of {} patches".format(len(feats_patient)))
    feats_patient = torch.from_numpy(feats_patient).unsqueeze(0).to(device)
    coords_patient = torch.from_numpy(np.concatenate(coords_patient))

    ## Compute attention scores from the extracted features
    Y_probs, attn_dict = infer_block(attn_model,
                                     feats_patient,
                                     coords_patient)

    ## Distribute features according to each block
    # Feature encoding with small MLP after pre-extraction
    features_enc = encode_features(attn_model, feats_patient)
    # Attend to the encoded features
    features_agg = attend_features(attn_model, features_enc, coords)

    for idx in range(numOfslides):
        features = feats_patient[:, counter[idx]: counter[idx + 1], ...]
        coords = coords_patient[counter[idx]: counter[idx + 1]]

        # Feature encoding with small MLP after pre-extraction
        features_enc = encode_features(attn_model, features)
        # Attend to the encoded features
        # features_agg = attend_features(attn_model, features_enc, coords)

        attn_dict_slide = {'inter': attn_dict['inter'],
                           'intra': attn_dict['intra'][counter[idx]: counter[idx + 1]],
                           'ig': attn_dict['ig'][counter[idx]: counter[idx + 1]]}

        results = {'attn_scores': attn_dict_slide,
                   'coords': coords.numpy(),
                   'features_agg': features_agg.detach().cpu().numpy(),
                   'features': features_enc.detach().squeeze().cpu().numpy(),
                   'Y_probs': Y_probs,
                   }

        patient_results.append(results)

    return patient_results


def attend_indiv(feats_path_list=[],
                 attn_model=None,
                 label=None,
                 label_dict={},
                 device=None,
                 **block_kwargs):
    """
        Given a set of features, process each slice.
        For ease of processing, currently allows only one block per patient

        Returns:
        - patient_results: List of result dictionaries, where each dictionary contains features/scores per block
        """
    patient_results = []

    for feats_path in feats_path_list:
        print("\nLoading pre-extracted features from {}".format(feats_path))
        assert os.path.isfile(feats_path), "{} does not exist!".format(feats_path)
        with h5py.File(feats_path, 'r') as f:
            features = f['features'][()]
            coords = f['coords'][()]

        attn_dict = {'inter': [], 'intra': [], 'ig': []}
        features_agg_list = []
        features_enc_list = []
        print("\nAttending to total of {} patches sequentially".format(len(features)))

        unique_levs = np.unique(coords[:, 0])

        for lev in tqdm(unique_levs):
            indices = np.flatnonzero(coords[:, 0] == lev)
            feats = torch.from_numpy(features[indices]).unsqueeze(0).to(device) # (numOfinstances, latent_dim) -> (1, numOfinstances, latent_dim)
            coord = torch.from_numpy(coords[indices])

            Y_probs, attn_dict_temp = infer_block(attn_model,
                                                  feats,
                                                  coord)
            attn_dict['inter'].append(attn_dict_temp['inter'])
            attn_dict['intra'].append(attn_dict_temp['intra'])
            attn_dict['ig'].append(attn_dict_temp['ig'])

            # Feature encoding with small MLP after pre-extraction
            features_enc = encode_features(attn_model, feats)
            # Attend to the encoded features
            features_agg = attend_features(attn_model, features_enc, coords)
            features_enc_list.append(features_enc.detach().cpu().squeeze(0).numpy())
            features_agg_list.append(features_agg.detach().cpu().numpy())

        attn_dict['inter'] = np.concatenate(attn_dict['inter'])
        attn_dict['intra'] = np.concatenate(attn_dict['intra'])
        attn_dict['ig'] = np.concatenate(attn_dict['ig'])
        features_agg_list = np.concatenate(features_agg_list)   # (numOfslices, latent_dim)
        features_enc_list = np.concatenate(features_enc_list)   # (numOffeatures, latent_dim)

        results = {'attn_scores': attn_dict,
                   'coords': coords,
                   'features_agg': features_agg_list,
                   'features': features_enc_list,
                   'Y_probs': Y_probs,
                   }

        patient_results.append(results)

    return patient_results

def attend_blocks(feats_path_list=[],
                  attn_model=None,
                  label=None,
                  label_dict={},
                  device=None,
                  attn_mode='whole',
                  **block_kwargs):
    """

    """
    if attn_mode == 'whole':
        patient_results = attend_whole(feats_path_list,
                                       attn_model,
                                       label, label_dict,
                                       device,
                                       **block_kwargs)
    elif attn_mode == 'indiv':
        patient_results = attend_indiv(feats_path_list,
                                       attn_model,
                                       label, label_dict,
                                       device,
                                       **block_kwargs)
    else:
        raise NotImplementedError("Not implemented for attn mode ", attn_mode)

    return patient_results


def write_video(img_list, video_length=10, save_fpath=None):
    """
    Create a video for the list of heatmaps

    Args:
    - img_list (list of PIL Images): list of heatmap images
    - video_length (int): Video time duration
    - save_fpath (str)

    Returns:
    - None
    """

    if len(img_list) == 1:
        print("Only a single heatmap image! Aborting video creation")
    else:
        w, h = np.array(img_list[0]).shape[:2]

        fps = len(img_list) // video_length
        video = cv2.VideoWriter(save_fpath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (w, h))

        for img in img_list:
            video.write(np.array(img))

        cv2.destroyAllWindows()
        video.release()

## Sampling ROIs
def sample_indices(scores, k, start=0.48, end=0.52, convert_to_percentile=False, seed=1):
    np.random.seed(seed)
    if convert_to_percentile:
        end_value = np.quantile(scores, end)
        start_value = np.quantile(scores, start)
    else:
        end_value = end
        start_value = start
    score_window = np.logical_and(scores >= start_value, scores <= end_value)
    indices = np.where(score_window)[0]
    if len(indices) < 1:
        return -1
    else:
        return np.random.choice(indices, min(k, len(indices)), replace=False)

def screen_coords(scores, coords, top_left, bot_right):
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

def top_k(scores, k, invert=False):
    if invert:
        top_k_ids=scores.argsort()[:k]
    else:
        top_k_ids=scores.argsort()[::-1][:k]
    return top_k_ids


def sample_rois(scores,
                coords,
                features=None,
                k=5,
                mode='range_sample',
                seed=1,
                score_start=0.45,
                score_end=0.55,
                top_left=None,
                bot_right=None):

    if len(scores.shape) == 2:
        scores = scores.flatten()

    # scores = to_percentiles(scores)
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)

    if mode == 'topk':
        sampled_ids = top_k(scores, k, invert=False)
    elif mode == 'reverse_topk':
        sampled_ids = top_k(scores, k, invert=True)
    elif mode == 'range_sample':
        sampled_ids = sample_indices(scores, start=score_start, end=score_end, k=k, convert_to_percentile=False, seed=seed)
        # print(sampled_ids)
        # if sampled_ids == -1:
        #     return {}
    else:
        raise NotImplementedError

    coords = coords[sampled_ids]
    scores = scores[sampled_ids]
    if features is not None:
        features = features[sampled_ids]

    asset = {'sampled_coords': coords, 'sampled_scores': scores, 'sampled_features': features}
    return asset


def save_patches(img_obj,
                 score,
                 coord,
                 patch_idx,
                 patch_save_dir=None,
                 patch_level=0,
                 clip_min=0,
                 clip_max=255,
                 patch_size=(96,96,96)):
    """
    Read and save patches
    """

    patches = img_obj.wsi.read_region(coord, patch_level, patch_size)
    patches = clip_and_normalize_img(np.array(patches), clip_min, clip_max)
    patches = (patches * 255).astype(np.uint8)
    patch_save_subdir = os.path.join(patch_save_dir, '{0}_z_{1}_x_{2}_y_{3}_a{4:.3f}'.format(patch_idx,
                                                                                             coord[0],
                                                                                             coord[1],
                                                                                             coord[2],
                                                                                             score))
    os.makedirs(patch_save_subdir, exist_ok=True)
    # write_video(patches, 10, os.path.join(patch_save_subdir, 'patch.avi'))

    for offset in range(len(patches)):
        if patches[offset].shape[-1] == 1:
            img = Image.fromarray(patches[offset][..., -1])
        else:
            img = Image.fromarray(patches[offset][..., ::-1])
        img_path = os.path.join(patch_save_subdir, '{0}_z_{1}_x_{2}_y_{3}_a{4:.6f}.png'.format(patch_idx,
                                                                                               coord[0] + offset,
                                                                                               coord[1],
                                                                                               coord[2],
                                                                                               score))
        img.save(img_path)

    return patches


def save_top_patches(img_obj,
                     scores,
                     coords,
                     features,
                     slide_id,
                     sample_args={},
                     sample_save_dir='.',
                     label=None,
                     patch_level=0,
                     patch_size=(96, 96, 96),
                     clip_min=0,
                     clip_max=1,
                     save=True,
                    ):
    """
    Save patches, coordinates, and scores for top k patches

    Args:
    -

    Returns:
    - None
    """

    print("\n\nSampling patches for ", slide_id)
    samples = sample_args.samples
    for sample in samples:
        if sample['sample']:
            save_dir = os.path.join(sample_save_dir, "label_{}".format(label))
            os.makedirs(save_dir, exist_ok=True)
            patch_save_dir = os.path.join(save_dir, slide_id, "{}_{}_{}".format(sample['name'],
                                                                                sample_args.sample_patch_score,
                                                                                sample['field']))
            os.makedirs(patch_save_dir, exist_ok=True)

            patch_list, coords_list, scores_list, features_list = process_patches(img_obj,
                                                                                  scores, coords,
                                                                                  features,
                                                                                  sample,
                                                                                  patch_save_dir=patch_save_dir,
                                                                                  patch_level=patch_level,
                                                                                  patch_size=patch_size,
                                                                                  clip_min=clip_min,
                                                                                  clip_max=clip_max,
                                                                                  save=save)

            ## Save patch features
            path_new = os.path.join(sample_save_dir,
                                    '{}_{}_{}_samples.h5'.format(slide_id, sample['name'], sample['field']))
            hdf5_dict = {'features': features_list,
                         'coords': coords_list,
                         'scores': scores_list
                         }
            if save:
                hdf5_dict['patches'] = patch_list

            save_hdf5(path_new, hdf5_dict, mode='w')


def process_patches(img_obj,
                    scores,
                    coords,
                    features,
                    sample_args,
                    patch_save_dir=None,
                    patch_level=0,
                    patch_size=(96, 96, 96),
                    clip_min=0,
                    clip_max=1,
                    save=True):
    """
    Sample relevant patches and save them
    """

    patch_list, coords_list, scores_list, features_list = [], [], [], []

    if sample_args['field'] == 'volume':
        print("\nSampling {} {} from entire volume".format(sample_args['k'], sample_args['name']))
        sample_results = sample_rois(scores, coords, features,
                                     k=sample_args['k'],
                                     mode=sample_args['mode'],
                                     seed=sample_args['seed'],
                                     score_start=sample_args.get('score_start', 0),
                                     score_end=sample_args.get('score_end', 1))

        if len(sample_results) == 0:
            return patch_list, coords_list, scores_list, features_list

        for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'],
                                                     sample_results['sampled_scores'])):
            print('coord: {} score: {:.3f}'.format(s_coord, s_score))

        if save:
            # Save patches
            for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'],
                                                         sample_results['sampled_scores'])):

                patch = save_patches(img_obj, s_score, s_coord, idx, patch_save_dir, patch_level, clip_min=clip_min,
                                     clip_max=clip_max,
                                     patch_size=patch_size)
                patch_list.append(patch)

            patch_list = np.stack(patch_list)

        coords_list = sample_results['sampled_coords']
        scores_list = sample_results['sampled_scores']
        features_list = sample_results['sampled_features']

    else:
        raise NotImplementedError("Not implemented")

    return patch_list, coords_list, scores_list, features_list

## Auxiliary functions
def identify_ckpt(slide_name, fold_dict, ckpt_name=None):
    """
    Idntify correct CV fold corresponding to the given slide

    Args:
    - slide_name (str)
    - fold_dict (dict): Dictionary with with key: slide_name, val: CV test fold index

    Returns:
    - ckpt (int): CV test fold index
    """
    if ckpt_name is not None:
        ckpt = ckpt_name
    else:
        if slide_name.split('-')[0] not in fold_dict.keys():
            ckpt = None
        else:
            ckpt = fold_dict[slide_name.split('-')[0]]
    return ckpt


def normalize_scores(scores, coords=None):
    if coords is None:
        scores_new = F.softmax(torch.from_numpy(scores), dim=0).numpy()
    else:
        scores_new = np.zeros_like(scores)
        z_unique_list = np.unique(coords[:, 0])

        for z_level in z_unique_list:
            indices = np.flatnonzero(coords[:, 0] == z_level)
            scores_new[indices] = F.softmax(torch.from_numpy(scores[indices]), dim=0).numpy()

    return scores_new

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

def load_params(df_entry, params):
    """
    Load pandas dataframe parameters onto params dict
    """
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
    return params
