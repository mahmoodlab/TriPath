# TriPath <img src=docs/images/tripath.jpeg width="250" align="right"> 

### Analysis of 3D pathology samples using weakly supervised AI
Andrew H. Song, Mane Williams, Drew F.K. Williamson, Sarah S.L. Chow, Guillaume Jaume, ..., Jonathan T.C. Liu*, Faisal Mahmood*  
*Cell*

[[Article]](https://www.cell.com/cell/fulltext/S0092-8674(24)00351-9) | [[Interactive Demo]](https://tripath-demo.github.io/demo/) | [[Video]](https://www.youtube.com/watch?v=JQh5FFmc6kY) | [[Cite]](#cite)

**TriPath** is a deep-learning-based computational pipeline for volumetric image analysis that can perform weakly-supervised patient prognostication based on 3D morphological features without the need for manual annotations by pathologists.
With the rapid growth and adoption of 3D spatial biology and pathology techniques by researchers and clinicians, MAMBA provides a general and efficient framework for 3D weakly supervised learning for clinical decision support and to reveal novel 3D morphological biomarkers and insights for prognosis and therapeutic response.  

<img src=docs/images/Github_heatmap.gif align="center"> 

<!-- <div>
<img src=docs/images/CT.png width="500" align="left">   
<div>
   <img src=docs/images/white.png width="100" align="right">
   <img src=docs/images/CT_heatmaps.gif width="300" align="right"> 
</div>
</div>

<div>
<img src=docs/images/OTLS.png width="400" align="left">
<div>
   <img src=docs/images/white.png width="250" align="right">
   <img src=docs/images/white.png width="200" align="right">
   <img src=docs/images/OTLS_heatmap.gif width="350" align="right"> 
</div>
</div> -->

<br><br>
© This code and corresponding models are made available for non-commercial academic purposes and is licenced under [the Creative Commons Attribution Non Commercial No Derivatives 4.0 International license](https://creativecommons.org/licenses/by-nc-nd/4.0/). Commercial entities may contact us or the Mass General Brigham Innovations office.

## Updates
**(05/09/24)** The TriPath article is now published in _**Cell**_ ! The codebase will be continually updated.\
**(07/27/23)** The github is now live.

## Installation
Run the following code to install all dependencies.
```
pip install -r requirements.txt
```

## TriPath Walkthrough
<img src=docs/images/workflow_github_preprocessing.jpg> 

### Step 1: Tissue segmentation & patching
We treat the volumetric image as a stack of 2D images and perform tissue segmentation serially on the stack. 
To perform segmenting/patching/stitching, run the following command (change the paths accordingly.)

```
cd preprocess
python create_patches_3D.py --patch_mode 3D --patch --seg --stitch --source DATA_PATH --save_dir SAVE_PATH --patch_size 128 --step_size 128 --save_mask --slice_mode all --thresh_mode global --process_list PROCESS_LIST.csv
```
Some flags include:
* `--slice_mode` (if `--patch_mode 2D`)
    * **single** (default): patch within single 2D slice at the depth with largest tissue contour area (indicated by best_slice_idx)
    * **all**: patch across all slices
    * **step**: patch within 2D slices that are certain steps apart (specify with `--step_z` flag)
* `--slice_mode` (if `--patch_mode 3D`)
    * **single** (default): patch within single 3D slice at the depth with largest tissue contour area (indicated by best_slice_idx)
    * **all** (recommended): patch across all 3D slices
* `--thresh_mode`
  * **fixed** (default): Uses csv-supplied clip_min, clip_max to threshold the images for all subjects. For CT, use this.
  * **global** (recommended): Automatically identifies adaptive upper threshold for each subject (Top 1%). Lower threshold is set to csv-supplied clip_min. This mode will take longer than fixed mode.
* `--process_list`: This csv contains relevant segmentation/patching parameters
  * **process**: 1 if the slide needs to be processed, 0 otherwise.
  * **sthresh**: tissue segmentation threshold for binarized volume (0 ~ 255) 
  * **mthresh**: median filtering threshold to create smooth contours
  * **a_t**: segmented tissue with area below *a_t x 256 x 256* will be discarded 
  * **a_h**: segmented hole with area below *a_h x 256 x 256* will be discarded
  * **black_thresh**: threshold below which slices with smaller mean intensity gets discarded. This is to identify a subset of image stack containing air.
  * **clip_min**: lower threshold for image binarization. Voxel intensity below clip_min will be cast to clip_min (required for tissue segmentation)
  * **clip_max**: upper threshold for image binarization. Voxel intensity above clip_max will be cast to clip_max (required for tissue segmentation)



The h5 files are saved in SAVE_PATH/EXTRACTED_DIR, where EXTRACTED_DIR will be determined by patching settings. With the above command, EXTRACTED_DIR=*patch_128_step_128_3D_all_global*.
The h5 files have the following format (e.g., subject name: '00001', block name: 'A')

**filename**: SAVE_PATH/EXTRACTED_DIR/0001-A_patches.h5
* **imgs**
  * If patch_mode=3D, numpy array of (numOfpatches, C, Z, W, H)
  * If patch_mode=2D, numpy array of (numOfpatches, C, W, H)
* **coords**
  * list of (z, x, y)

### Step 2: Feature extraction
<img src=docs/images/computational_processing.gif> 

To perform feature extraction, run the following command
```
CUDA_VISIBLE_DEVICES=0 python extract_features.py --dataroot SAVE_PATH --extracted_dir EXTRACTED_DIR --batch_size 50 --patch_mode 3D --process_list process_list_extract.csv --encoder 2plus1d --augment_fold 5 --config conf/config_extraction.yaml --data_mode CT
```

Some flags include:
* `--process_list`: csv file with subject, image lower/upper threshold information. The segmentation/patching operation produces SAVE_PATH/EXTRACTED_DIR/process_list_extract.csv automatically.
* `--patch_mode`: '2D' for stacks of 2D slices processing and '3D' for 3D processing
* `--batch_size`: Batch size (number of patches) for feature extraction
* `--data_mode`: Input data modality
  * **CT** (applicable for 2D/3D data)
  * **OTLS** (applicable for 2D/3D data)
* `--encoder`: Feature extractor of choice (Please refer to models/feature_extractor/)
  * If 2D encoders are chosen for 3D patches, the encoder will encode feature for each patch slice and average the features within each 3D patch. 
* `--augment_fold`: Number of augmentations to perform (if 0, no augmentation performed)

As more 3D vision encoders become public, these will be integrated into **TriPath**.

### Step 3: Training
To run binary classification, run the following command
```
CUDA_VISIBLE_DEVICES=0 python train_cv.py --config conf/config_MIL.yaml --sample_prop 0.5 --split_mode kf
--attn_latent_dim 64 --encoder 2plus1d --numOfaug 5 --seed_data 10 --task clf --prop_train 1.0
--feats_path /path/to/features_folder
```

Some flags include:
* `--config`: YAML file that contains training parameters. These can be overriden with command-line arguments.
* `--split_mode`: 'loo' for leave-one-out CV and 'kf' for k-fold CV
* `--prop_train`: Proportion of training dataset to use for training (rest is used for validation)
* `--sample_prop`: Proportion of patches to sample from each bag of patches
* `--numOfaug`: Number of augmentations
* `--seed_data`: Random seed for data splits
* `--encoder`: Feature extractor. Must match the extractor used in the previous step
* `--task`: Classification (clf) or survival (surv)
* `--decoder_enc`: If specified, Add shallow MLP encoder on top of the features for further encoding (Useful for learning more discriminative features at the risk of overfitting due to increased number of parameters)

### (Optional) Testing
The trained models can be used to perform inference on a new sample
```
CUDA_VISIBLE_DEVICES=0 python inference.py --config conf/config_inference.yaml --mode external
```
Some flags include:
* `--mode`: Whether test data is external or internal (used in CV analysis). Required for selection of model checkpoints
  * 'internal': If the dataset was part of CV analysis, identify the CV-fold for which the test data was not part of the training dataset.
  * 'external': If not part of the cv-analysis, all models can be used to perform inference

### (Optional) Phantom Dataset Generation
If you wish to generate a phantom dataset of cell-like structures with which to analyze in a pipeline, you can run the following script:
```
python gen_artificial_data.py SAVE_PATH 50 --n_classes 2 --n_obj 300 --type cells --h 256 --w 256 --d 256 --mode RGB --size 20 --resize 2.0 4.0 4.0 --prefix gen-img
```
This will generate 50 artificial 3D images populated with cell-like structures, whose properties are determined by statistical distributions that can be manually modified via the gen_3d_img() function in utils/image_gen_utils.py. This command also specifies that the 3D images are generated as 256x256x256 but then scaled to 512x1024x1024, and that the images are RGB.

## Visualization
To visualize a 3D image in Napari, you can run the following command:
```
python visualize.py IMAGE_PATH --rgb
```
This will open the image slices in napari, which then enables easy 3D visualization as well as the ability to generate animations with the 3d image.


### Post-hoc interpretation
To create interpretable heatmap imposed on the raw volumetric image
```
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config conf/config_heatmap.yaml --mode full
```
The example heatmaps can be visualized at our interactive demo. [[Interactive Demo]](https://tripath-demo.github.io/demo/)

## Contact
For any suggestions or issues, please contact Andrew H. Song <asong@bwh.harvard.edu>

## Cite
If you find our work useful in your research or if you use parts of this code please cite our paper:
```bibtext
@article{Song2024analysis,
    title={Analyis of 3D pathology samples using weakly supervised AI},
    author = {Song, Andrew H and Williams, Mane and Williamson, Drew FK and Chow, Sarah SL and Jaume, Guillaume and Gao, Gan and Zhang, Andrew and Chen, Bowen and Baras, Alexander S and Serafin, Robert and Colling, Richard and Downes, Michelle R and Farre, Xavier and Humphrey, Peter and Verrill, Clare and True, Lawrence D and Parwani, Anil V and Liu, Jonathan TC and Mahmood, Faisal},
    journal = {Cell},
    volume={187},
    year = {2024},
    publisher = {Cell Press}
}
```

## License and Usage 
[Mahmood Lab](https://faisal.ai) - This code is made available under the CC-BY-NC-ND 4.0 License and is available for non-commercial academic purposes.

<img src=docs/images/joint_logo.png> 
