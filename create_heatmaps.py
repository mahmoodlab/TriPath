"""
Heatmap generation script that performs three tasks

1. Save top attended patches
2. Create heatmap without overlap
3. Create heatmap with overlap
"""
from __future__ import print_function
import numpy as np
import argparse
import time
import torch
import os
import pandas as pd
import yaml
import pickle

from preprocess.wsi_core.wsi_utils import initialize_df
from models.feature_extractor import get_extractor_model
from models.head import get_decoder_input_dim
from utils.heatmap_utils import initialize_img, drawHeatmap, load_params, \
                                save_top_patches, initiate_attn_model, \
                                attend_blocks, identify_ckpt, saveHeatmap, \
                                normalize_ig_scores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--config', type=str, default="config_heatmap.yaml")
parser.add_argument('--mode', type=str, choices=['fast', 'full'], default='full')
args = parser.parse_args()

if __name__ == '__main__':
    config_path = args.config
    process_mode = args.mode
    config_dict = yaml.safe_load(open(config_path, 'r'))
    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n'+key)
            for value_key, value_value in value.items():
                print(value_key + " : " + str(value_value))
        else:
            print('\n'+key + " : " + str(value))

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    model_args = args['model_arguments']
    model_args.update({'n_classes': args['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args)
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])
    config_exp = yaml.safe_load(open(os.path.join(exp_args.exp_dir, 'conf.yaml'), 'r'))

    if patch_args.patch_mode == '3D':
        patch_size_z = patch_args.patch_size_z if patch_args.patch_size_z is not None else patch_args.patch_size
    else:
        patch_size_z = 1

    patch_size = tuple([patch_size_z, patch_args.patch_size, patch_args.patch_size])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print("Patch size: ", patch_size, " Step size: ", step_size)
    print(heatmap_args)

    ##########################
    # Set default parameters #
    ##########################
    def_seg_params = {'seg_level': 0, 'sthresh': 30, 'mthresh': 15, 'close': 4, 'use_otsu': False,
                      'keep_ids': 'none', 'exclude_ids': 'none'}
    def_filter_params = {'a_t': 5, 'a_h': 1, 'max_n_holes': 10}
    def_vis_params = {'vis_level': 1, 'line_thickness': 10}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt_easy'}
    def_img_params = {'black_thresh': 20000, 'clip_min': 26000, 'clip_max': 36000}

    ###################################
    # Create lists of slides to process
    ###################################
    if data_args.process_list is None:
        if isinstance(data_args.data_dir, list):
            slides = []
            for data_dir in data_args.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            slides = sorted(os.listdir(data_args.data_dir))
        slides = [slide for slide in slides if data_args.slide_ext in slide]

        df = initialize_df(slides,
                           seg_params=def_seg_params,
                           filter_params=def_filter_params,
                           vis_params=def_vis_params,
                           patch_params=def_patch_params,
                           img_params=def_img_params)

    else:
        df = pd.read_csv(data_args.process_list, dtype={'patient_id': str})

    mask = df['process'] == 1
    process_stack = df[mask]

    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    production_save_dir = os.path.join(exp_args.save_dir, exp_args.production_save_dir)
    raw_save_dir = os.path.join(exp_args.save_dir, exp_args.raw_save_dir)
    sample_save_dir = os.path.join(exp_args.save_dir, 'sampled_patches')

    os.makedirs(production_save_dir, exist_ok=True)
    os.makedirs(raw_save_dir, exist_ok=True)

    feature_extractor = get_extractor_model(model_args.extractor_name)
    channel = feature_extractor.get_channel_dim()

    label_dict = data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))}

    result_path = os.path.join(exp_args.exp_dir, 'result.pkl')
    with open(result_path, 'rb') as f:
        info = pickle.load(f)
    # Dictionary mapping between subject and the CV fold it is associated with
    fold_dict = {subj: fold for subj, fold in zip(info['subject'], info['fold'])}

    ## Loop through all the patients in the list
    asset_dict = {'features': {}, 'features_agg': {}, 'scores': {}, 'coords': {}}
    for i in process_stack.index.values:

        patient_id = process_stack.loc[i, 'patient_id']
        slide_ids = [s.strip() for s in process_stack.loc[i, 'slide_id'].split(',')]

        print("===============================")
        print('\n\nprocessing: ', patient_id, slide_ids)

        s = time.time()

        try:
            label = process_stack.loc[i, 'label']
        except KeyError:
            print("Label does not exist! Defaulting to Unspecified...")
            label = 'Unspecified'

        if not isinstance(label, str):
            grouping = reverse_label_dict[label]
        else:
            grouping = label

        r_slide_save_dir_list, p_slide_save_dir_list = [], []

        for slide_id in slide_ids:
            slide_name = str(patient_id) + '-' + slide_id

            # Production results
            p_slide_save_dir = os.path.join(production_save_dir, exp_args.save_exp_code, str(grouping))
            os.makedirs(p_slide_save_dir, exist_ok=True)

            # Raw results
            r_slide_save_dir = os.path.join(raw_save_dir, exp_args.save_exp_code, str(grouping), slide_name)
            os.makedirs(r_slide_save_dir, exist_ok=True)

            r_slide_save_dir_list.append(r_slide_save_dir)
            p_slide_save_dir_list.append(p_slide_save_dir)

        # A folder for saving temporary files for heatmap creation
        save_path_temp = os.path.join(exp_args.save_dir, 'temp')
        os.makedirs(save_path_temp, exist_ok=True)

        top_left = None
        bot_right = None

        ## Load attention model
        attn_model_dict = {"dropout": config_exp['dropout'],
                           # "out_dim": exp_args.n_classes,
                           "out_dim": 1,
                           'attn_latent_dim': config_exp['attn_latent_dim'],
                           'decoder': config_exp['decoder'],
                           'decoder_enc': config_exp['decoder_enc'],
                           'decoder_enc_dim': config_exp['decoder_enc_dim'],
                           'context': config_exp['context'],
                           'context_network': config_exp['context_network'] if 'context_network' in config_exp else 'GRU',
                           'input_dim': get_decoder_input_dim(model_args.extractor_name)}

        ckpt_name = identify_ckpt(patient_id, fold_dict, ckpt_name=model_args.ckpt_path)
        if ckpt_name is None:   # If appropriate ckpt_name cannot be found, just skip
            print("Cannot identify ckpt. Skipping!")
            continue
        ckpt_path = os.path.join(exp_args.exp_dir, 'checkpoints', 'ckpt_split--{}.pt'.format(str(ckpt_name)))
        attn_model = initiate_attn_model(model_dict=attn_model_dict, ckpt_path=ckpt_path)
        attn_model.to(device)

        # Load segmentation and filter parameters
        seg_params = load_params(process_stack.loc[i], def_seg_params.copy())
        filter_params = load_params(process_stack.loc[i], def_filter_params.copy())
        vis_params = load_params(process_stack.loc[i], def_vis_params.copy())
        img_params = load_params(process_stack.loc[i], def_img_params.copy())

        # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
        # vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))
        vis_patch_size = patch_size

        feats_path_list = []
        ## Construct feature list for the patient
        for slide_id in slide_ids:
            slide_name = patient_id + '-' + slide_id
            # If extracted features already exist, supply the path
            feats_path = os.path.join(exp_args.feats_path_block,
                                      slide_name + '.h5') if exp_args.feats_path_block is not None else None
            feats_path_list.append(feats_path)

        block_raw_kwargs = {'top_left': top_left, 'bot_right': bot_right,
                            'patch_size': patch_size, 'step_size': patch_size,
                            'patch_level': patch_args.patch_level,
                            'use_center_shift': heatmap_args.use_center_shift}

        heatmap_vis_args = {'convert_to_percentiles': heatmap_args.convert_to_percentiles,
                            'vis_level': 0,
                            'blur': heatmap_args.blur,
                            'custom_downsample': heatmap_args.custom_downsample}

        results_list = attend_blocks(feats_path_list=feats_path_list,
                                     attn_model=attn_model,
                                     label=label,
                                     label_dict=reverse_label_dict,
                                     device=device,
                                     attn_mode=heatmap_args.attn_mode,
                                     **block_raw_kwargs)

        ## Construct img_obj list for the patient
        img_obj_list, clip_min_list, clip_max_list = [], [], []
        for slide_id in slide_ids:
            slide_name = patient_id + '-' + slide_id

            if isinstance(data_args.data_dir, str):
                slide_path = os.path.join(data_args.data_dir, slide_name)
                if not os.path.exists(slide_path):
                    raise FileNotFoundError("{} doesn't exist!".format(slide_path))
            else:
                raise NotImplementedError

            if process_mode == 'fast':
                img_obj = None
                clip_min = img_params['clip_min']
                clip_max = img_params['clip_max']
                save = False
            else:
                false_color = process_stack.loc[i, 'false_color']
                save = True

                img_obj = initialize_img(slide_path,
                                         patch_mode=patch_args.patch_mode,
                                         binarize=True,
                                         segment=True,
                                         seg_mask_path=None,
                                         img_params=img_params,
                                         seg_params=seg_params,
                                         filter_params=filter_params)

                clip_min = img_params['clip_min']
                clip_max = img_params['clip_max']

                # If false color overlays exist, load them
                if false_color:
                    contours_tissue = img_obj.contours_tissue
                    holes_tissue = img_obj.holes_tissue

                    print("\nLoading false colored image for overlaying")
                    false_color_path = os.path.join(data_args.data_overlay_dir, slide_name)
                    if not os.path.exists(false_color_path):
                        raise FileNotFoundError('{} does not exist!'.format(false_color_path))

                    img_obj = initialize_img(path=false_color_path,
                                             patch_mode='2D',
                                             binarize=False,
                                             segment=False,
                                             img_params={'black_thresh': -1})

                    img_obj.set_contours(contours_tissue, holes_tissue)

                    clip_min = 0
                    clip_max = 255

            img_obj_list.append(img_obj)
            clip_min_list.append(clip_min)
            clip_max_list.append(clip_max)

        for idx, (slide_id, img_obj, results, clip_min, clip_max) in enumerate(zip(slide_ids,
                                                                                   img_obj_list,
                                                                                   results_list,
                                                                                   clip_min_list,
                                                                                   clip_max_list)):
            slide_name = patient_id + '-' + slide_id
            results = results_list[idx]

            scores_ig = results['attn_scores']['ig']
            scores_ig = normalize_ig_scores(torch.tensor(scores_ig)).numpy()

            scores = {'ig': scores_ig,
                      'attn': results['attn_scores']['intra']}

            features_agg = results['features_agg']
            features = results['features']
            coords = results['coords']

            asset_dict['features'][slide_name] = features
            asset_dict['scores'][slide_name] = scores
            asset_dict['coords'][slide_name] = coords
            asset_dict['features_agg'][slide_name] = features_agg

            ## Save top samples
            if sample_args.sample_patch_score == 'ig':
                scores_sample = scores_ig
            elif sample_args.sample_patch_score == 'attn':
                scores_sample = results['attn_scores']['intra']
            else:
                raise NotImplementedError

            save_top_patches(img_obj=img_obj,
                             scores=scores_sample,
                             coords=coords,
                             features=features,
                             slide_id=slide_name,
                             sample_args=sample_args,
                             sample_save_dir=sample_save_dir,
                             label=label,
                             patch_level=patch_args.patch_level,
                             patch_size=patch_size,
                             clip_min=clip_min,
                             clip_max=clip_max,
                             save=save)



        ####################################
        # Create heatmaps (non-overlapping) #
        ####################################
        if heatmap_args.draw_blocky_heatmap:
            assert process_mode != 'fast', "Cannot create heatmap with fast mode"
            for idx, (slide_id, img_obj, results) in enumerate(zip(slide_ids, img_obj_list, results_list)):
                slide_name = patient_id + '-' + slide_id
                r_slide_save_dir = r_slide_save_dir_list[idx]
                p_slide_save_dir = p_slide_save_dir_list[idx]
                results = results_list[idx]

                scores = {'ig': results['attn_scores']['ig'],
                          'attn': results['attn_scores']['intra']}

                if heatmap_args.heatmap_score == 'ig':
                    scores_vis = scores['ig']
                elif heatmap_args.heatmap_score == 'attn':
                    scores_vis = scores['attn']
                else:
                    raise NotImplementedError("Not implemented for {}".format(heatmap_args.heatmap_score))

                features = results['features']
                coords = results['coords']

                print("\nDrawing blocky heatmap...")
                heatmap_list, z_levels_list = drawHeatmap(scores_vis,
                                                          coords,
                                                          save_path_temp=save_path_temp,
                                                          clip_min=clip_min,
                                                          clip_max=clip_max,
                                                          img_obj=img_obj,
                                                          segment=True,
                                                          cmap=heatmap_args.cmap,
                                                          alpha=heatmap_args.alpha,
                                                          binarize=heatmap_args.binarize,
                                                          blank_canvas=heatmap_args.blank_canvas,
                                                          thresh=heatmap_args.binary_thresh,
                                                          patch_size=vis_patch_size,
                                                          overlap=patch_args.overlap,
                                                          cmap_normalize=heatmap_args.cmap_normalize,
                                                          cmap_min=heatmap_args.cmap_min,
                                                          cmap_max=heatmap_args.cmap_max,
                                                          top_left=top_left, bot_right=bot_right,
                                                          **heatmap_vis_args
                                                          )
                saveHeatmap(r_slide_save_dir, slide_name, heatmap_list, z_levels_list, down_factor=2, video=False)

        ##################################################
        # Create production-level heatmaps (overlapping) #
        ##################################################
        if heatmap_args.draw_fine_heatmap:
            assert process_mode != 'fast', "Cannot create heatmap with fast mode"
            print("\nGenerating heatmaps with overlap...")
            ref_scores = None

            # Recompute attention scores based on overlap
            block_prod_kwargs = {'top_left': top_left, 'bot_right': bot_right,
                                 'patch_size': patch_size, 'step_size': step_size,
                                 'patch_level': patch_args.patch_level,
                                 'use_center_shift': heatmap_args.use_center_shift}

            ## Construct img_obj and features list for the patient
            feats_path_list = []
            for slide_id in slide_ids:
                slide_name = patient_id + '-' + slide_id
                # If extracted features already exist, supply the path
                feats_path = os.path.join(exp_args.feats_path_fine,
                                          slide_name + '.h5') if exp_args.feats_path_block is not None else None
                feats_path_list.append(feats_path)

            results_list = attend_blocks(img_obj_list=img_obj_list,
                                         feats_path_list=feats_path_list,
                                         attn_model=attn_model,
                                         label=label,
                                         label_dict=reverse_label_dict,
                                         device=device,
                                         attn_mode=heatmap_args.attn_mode,
                                         **block_prod_kwargs)

            for idx, (slide_id, img_obj, results) in enumerate(zip(slide_ids, img_obj_list, results_list)):
                slide_name = patient_id + '-' + slide_id
                r_slide_save_dir = r_slide_save_dir_list[idx]
                p_slide_save_dir = p_slide_save_dir_list[idx]
                results = results_list[idx]

                scores = {'ig': results['attn_scores']['ig'],
                          'attn': results['attn_scores']['intra']}

                if heatmap_args.heatmap_score == 'ig':
                    scores_vis = scores['ig']
                elif heatmap_args.heatmap_score == 'attn':
                    scores_vis = scores['attn']
                else:
                    raise NotImplementedError

                features = results['features']
                coords = results['coords']

                heatmap_list, z_levels_list = drawHeatmap(scores_vis,
                                                          coords,
                                                          save_path_temp=save_path_temp,
                                                          clip_min=clip_min,
                                                          clip_max=clip_max,
                                                          img_obj=img_obj,
                                                          segment=True,
                                                          cmap=heatmap_args.cmap,
                                                          alpha=heatmap_args.alpha,
                                                          binarize=heatmap_args.binarize,
                                                          blank_canvas=heatmap_args.blank_canvas,
                                                          thresh=heatmap_args.binary_thresh,
                                                          patch_size=vis_patch_size,
                                                          overlap=patch_args.overlap,
                                                          top_left=top_left, bot_right=bot_right,
                                                          cmap_normalize=heatmap_args.cmap_normalize,
                                                          **heatmap_vis_args
                                                          )

                saveHeatmap(p_slide_save_dir,
                            slide_name,
                            heatmap_list,
                            z_levels_list, down_factor=1,
                            video=False)

        e = time.time()
        print("Took ", e-s)

    ## Save cohort-level data
    with open(os.path.join(exp_args.save_dir, 'cohort_asset.pkl'), 'wb') as f:
        pickle.dump(asset_dict, f)
