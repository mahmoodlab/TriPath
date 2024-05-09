import argparse
from utils.image_gen_utils import *
from tqdm import tqdm
import pandas as pd
from pathlib import Path

def gen_data(type, n_samples, save_prefix, mode='RGB', n_classes=2, n_obj=100, d=360, w=360, h=360, size=24, resize=[1.0], filetype='tiff'):
    """
    Generates a series of images for a simulated volume of tissue. The folder of generated images can then
    be loaded or processed as a 3D image.

    Inputs:
    - type ['cells', 'polygons', 'spheres', 'cubes']: The shape of generated items in the volume
    - n_samples (int): The number of generated volumes (dataset size)
    - save_prefix (string): The prefix for saved image files and the folder name
    - mode ['L', 'RGB']: The mode for image generation, 'L' is for black and white images and 'RGB' is for 
        color
    - n_classes (int): The number of different cases
    - n_obj (int): The number of items to simulate in each 3D image generated.
    - d (int): the depth of each generated 3D image (z-axis length, i.e. the number of 2D images generated)
    - w (int): the width of each generated 3D image
    - h (int): the height of each generated 3D image
    - size (int): The average size in pixels for each simulated shape (side length for cubes, diameter for 
        spheres)
    - resize (list(float)): The scale factor for each dimension (length 3 list) to rescale at the end by. If length=1 then the 
        same rescaling factor is used for all three image dimensions.
    - filetype (string): The filetype to save each 2D image as
    """
    n_digits = len(str(int(n_samples)))
    samples_per = n_samples / n_classes
    for i in tqdm(range(n_samples)):
        case = math.floor(i/samples_per)
        cells_phantom = gen_3d_img(n_obj, d, w, h, size, type, case, n_classes, mode=mode)
        img_path = save_prefix+'_'+type+'_'+str(i).zfill(n_digits)
        save_generated_img(cells_phantom, img_path, mode=mode, filetype=filetype, resize=resize)


def create_csv(csv_type, class_types, class_counts, prefixes, label_counts, save_path):
    """
    Generates a mock csv file that can be used to process the generated 3D image.
    Note that all inputs are lists, and each index corresponds to different 'type' of generated images.
    So, if you generated 20 gray spheres and 20 color polygons to represent two cases of patient, you 
    could have as inputs: 
    create_csv('patching', ['spheres', 'polygons'], [20, 20], ['L', 'RGB'], [20, 20], '../gray_phantom_clinical_list.csv')
    If only one type of cell (with the same prefix) was used (as will usually be the case), all input lists will be length=1
    except for label_counts (whose length will match the number of labels among the images).

    Inputs:
    - csv_type ['patching', 'clinical']: Determines the columns generated in the resulting csv.
    - class_types (list(string)): The input 'type' used in image generation with the gen_data() function.
    - class_counts (list(int)): the number of each class generated
    - prefixes (list(string)): A list of the save prefixes used when generated the images (e.g. prefix1_00.tiff)
    - label_counts (list(int)): The counts for each label value in order. For example, [10, 20, 10] will have the first ten 
        images have label=0, the next 20 have label=1, and the final 10 have label=2.
    -save_path (string): The full or relative filepath to save the generated csv file as.
    """
    d = {}
    names = []
    classes = []
    class_nums = []
    total = sum(class_counts)
    n_labels = label_counts[0]
    x = 0
    label = 0
    for i in range(len(class_types)):
        c_t, count, prefix = class_types[i], class_counts[i], prefixes[i]
        n_digits = len(str(count))
        for j in range(count):
            x += 1
            if x > n_labels:
                label += 1
                n_labels += label_counts[label]
            name = prefix+'_'+c_t+'_'+str(j).zfill(n_digits)
            names.append(name)
            classes.append(c_t)
            class_nums.append(label)
    if csv_type == 'patching':
        d['patient_id'] = names
        d['slide_id'] = names
        d['process'] = ['1'] * total
        d['status'] = ['tbp'] * total
        d['seg_level'] = ['0'] * total
        d['sthresh'] = ['4'] * total
        d['mthresh'] = ['1'] * total
        d['close'] = ['4'] * total
        d['use_otsu'] = ['FALSE'] * total
        d['a_t'] = ['0'] * total
        d['a_h'] = ['0'] * total
        d['max_n_holes'] = ['8'] * total
        d['vis_level'] = ['0'] * total
        d['line_thickness'] = ['8'] * total
        d['use_padding'] = ['TRUE'] * total
        d['contour_fn'] = ['all'] * total
        d['black_thresh'] = ['-1'] * total
        d['clip_min'] = ['0'] * total
        d['clip_max'] = ['255'] * total
    elif csv_type == 'clinical':
        d['patient_id'] = names
        d['slide_id'] = [''] * total
        d['type'] = classes
        d['label'] = class_nums
    df = pd.DataFrame(data=d)
    df.to_csv(save_path, header=True, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument('save_dir_path', type=str, default='.',
                        help='Path to the directory in which to save the generated 3D images.')
    parser.add_argument('n_samples', type=int, default=1,
                        help='Number of 3D images to generate.')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of classes in the dataset.')
    parser.add_argument('--prefix', type=str, default='phantom',
                        help='Prefix for saved images and image directory.')
    parser.add_argument('--n_obj', type=int, default=500,
                        help='Number of objects populating each 3D image.')
    parser.add_argument('--size', type=int, default=16,
                        help='Average radius in pixels of each object.')
    parser.add_argument('--type', type=str, choices=['cells', 'spheres', 'cubes', 'polygons'], default='cells',
                        help='Chooses the type of objects to populate the images with.')
    parser.add_argument('--h', default=360, type=int,
                        help='Generated image height.')
    parser.add_argument('--w', default=360, type=int,
                        help='Generated image width.')
    parser.add_argument('--d', default=360, type=int,
                        help='Generated image depth.')
    parser.add_argument('--resize', default=[1.0], type=float, nargs="+",
                        help='Values to multiply the (D,H,W) original image size by before saving. If only one value exists'
                        ' in the list it will be used for all three, otherwise the first three values will be utilized.')
    parser.add_argument('--mode', default='RGB', type=str, choices=['L', 'RGB'],
                        help='Generated image mode (how colors are stored in data).')
    parser.add_argument('--filetype', type=str, default='tiff',
                        help='What filetype to save the individual slice images as.')
    args = parser.parse_args()


    img_dir = os.path.join(args.save_dir_path, args.prefix)
    Path(img_dir).mkdir(parents=True, exist_ok=True)
    ## Use the directory+prefix as the full filepath prefix when saving individual images
    img_path_prefix = os.path.join(img_dir, args.prefix)

    n_digits = len(str(int(args.n_samples)))

    gen_data(args.type, args.n_samples, img_path_prefix, args.mode, args.n_classes, args.n_obj, args.d, args.w, args.h, 
            args.size, args.resize, args.filetype)
    create_csv('patching', ['cells'], [args.n_samples], [args.prefix], 
                [round(args.n_samples/args.n_classes)]*args.n_classes, img_path_prefix+'_patching.csv')
    create_csv('clinical', ['cells'], [args.n_samples], [args.prefix], 
                [round(args.n_samples/args.n_classes)]*args.n_classes, img_path_prefix+'_clinical.csv')

    print('done')