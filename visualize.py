"""
Script for 3D visualizations with Napari
"""

import argparse
import napari
import sys
import os
from napari_animation import Animation
import numpy as np
from PIL import Image
from preprocess.wsi_core.img_utils import read_img, clip_and_normalize_img

parser = argparse.ArgumentParser(description="Visualize a 3D image in Napari")
parser.add_argument('img_dir_path', type=str, default='.',
                    help='Path to the image slices forming a 3D image.')
parser.add_argument('--save_dir', type=str)
parser.add_argument('--rgb', action='store_true', default=False,
                    help='If included, the image will be read as an RGB image.')
parser.add_argument('--sim', action='store_true', default=False,
                    help='If included, the image will use rendering parameters appropriate for viewing simulation data.')
parser.add_argument('--snapshot', action='store_true', default=False,
                    help='If included, all images from the supplied directory will be loaded and a snapshot of their '
                    '3D volume taken at a slanted angle will be generated for each one.')
parser.add_argument('--mode', type=str, default='OTLS', choices=['OTLS', 'CT'],
                    help='The image type (which affects the image shape and resulting camera angles for snapshots).')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='If included, the image will be clipped between top/bottom 1 percent intensity values and '
                    'normalized within the range between.')
parser.add_argument('--resolution', default=1.0, type=float,
                    help='The resolution of the image, with 1.0 indicating full resolution and 0.5 indicating a '
                    'resized image of (width*0.5, height*0.5, depth*0.5)')
parser.add_argument('--reduce_factor', default=1.0, type=float,
                    help='Same effect as --resolution, but specifying the factor by which to reduce the image size. '
                    'Inputting 2.0 gives a resized image of (width/2, height/2, depth/2)')
parser.add_argument('--animation_mode', action='store_true', default=False,
                    help='If included, the Napari console will include options for capturing keyframes for an animation.')
parser.add_argument('--transparent', action='store_true', default=False,
                    help='If included, The image will replace sufficiently white pixels with transparent ones.')
parser.add_argument('--black_thresh', default=-1.0, type=float,
                    help='The black threshold used when reading and displaying the image, if -1 (default) then no thresholding is used.')
args = parser.parse_args()

if __name__ == '__main__':
    # First reduce the image size by the provided scaling if present.
    res = 1.0
    if args.resolution != 1.0:
        res = args.resolution
    elif args.reduce_factor != 1.0:
        res = 1.0 / args.reduce_factor

    img_arr, _ = read_img(args.img_dir_path, args.black_thresh, resolution=res)
    img_shape = np.shape(img_arr)
    print(f'Image shape: {img_shape}')

    # If included, clip out sufficiently white pixels (slow, only use when necessary)
    if args.transparent:
        for x in range(img_shape[0]):
            for y in range(img_shape[1]):
                for z in range(img_shape[2]):
                    if img_arr[x][y][z][0] > 220 and img_arr[x][y][z][1] > 220 and img_arr[x][y][z][2] > 220:
                        img_arr[x][y][z] = [0,0,0]
    
    if args.normalize:
        img_temp = img_arr.flatten()
        img_temp.sort()
        clip_min_adaptive = img_temp[len(img_temp) // 100]
        clip_max_adaptive = img_temp[-len(img_temp) // 100]
        img_arr = clip_and_normalize_img(img_arr, clip_min=clip_min_adaptive, clip_max=clip_max_adaptive)
        img_arr = (img_arr*256).astype(np.uint8)
        print("Image normalized!")

    # Choose the appropriate rendering parameters for viewing
    viewer = napari.Viewer(ndisplay=3)
    interp = "nearest"
    # interp='bicubic'
    render = "attenuated_mip"
    if args.rgb == True:
        render = "translucent"
    blend = "additive"
    gamma = 1.0
    iso_thresh = 1.0
    if args.sim:
        blend = "opaque"
        render = "iso"
        iso_thresh = 0.3
        gamma = 0.7

    # These manual colors are used so that pixels values of [<=0.08, <=0.08, <=0.08] (on a [0, 0, 0] to [1.0, 1.0, 1.0] scale)
    # are made to be transparent (have an alpha of 0.0), and the rest be opaque (have an alpha of 1.0)
    red_colors = np.linspace(
        start=[0, 0, 0, 1.0],
        stop=[1.0, 0, 0, 1.0],
        num=26,
        endpoint=True
    )
    red_colors[0] = np.array([0, 0, 0, 0])
    red_colors[1] = np.array([0.04, 0, 0, 0])
    red_colors[2] = np.array([0.08, 0, 0, 0])
    translucent_red = {
        'colors': red_colors,
        'name': 'translucent_red',
        'interpolation': 'linear'
    }
    green_colors = np.linspace(
        start=[0, 0, 0, 1.0],
        stop=[0, 1.0, 0, 1.0],
        num=26,
        endpoint=True
    )
    green_colors[0] = np.array([0, 0, 0, 0])
    green_colors[1] = np.array([0, 0.04, 0, 0])
    green_colors[2] = np.array([0, 0.08, 0, 0])
    translucent_green = {
        'colors': green_colors,
        'name': 'translucent_green',
        'interpolation': 'linear'
    }
    blue_colors = np.linspace(
        start=[0, 0, 0, 1.0],
        stop=[0, 0, 1.0, 1.0],
        num=26,
        endpoint=True
    )
    blue_colors[0] = np.array([0, 0, 0, 0])
    blue_colors[1] = np.array([0, 0, 0.04, 0])
    blue_colors[2] = np.array([0, 0, 0.08, 0])
    translucent_blue = {
        'colors': blue_colors,
        'name': 'translucent_blue',
        'interpolation': 'linear'
    }


    if len(img_shape) == 4:
        if img_shape[3] == 3: # Indicates RGB image
            viewer.add_image(
                img_arr, channel_axis=3, gamma=gamma, colormap=[translucent_blue, translucent_green, translucent_red],
                interpolation=[interp]*3, rendering=[render]*3, iso_threshold=iso_thresh, name=["blue", "green", "red"], 
                blending = [blend]*3)
        else: # We otherwise assume a single channel black/white image
            viewer.add_image(
                img_arr, channel_axis=3, gamma=gamma, rgb=False, interpolation=interp, rendering=render, blending=blend)
    elif len(img_shape) == 3: # Indicates there is no separate channel axis
        viewer.add_image(img_arr, rgb=False, gamma=gamma, interpolation=interp, rendering=render, blending=blend)
    else:
        sys.exit('Could not read image due to invalid image shape!')
    initial_zoom = viewer.camera.zoom

    if args.img_dir_path[-1] == '/': 
        args.img_dir_path = args.img_dir_path[:-1]

    # For snapshotting, we move to pre-specified angles and save the resulting views
    if args.snapshot:
        img_name = args.img_dir_path.split('/')[-1]

        OTLS_angles = [(-30, 30, -50)]
        # OTLS_angles = [
        #               (90, 45, 45),
        #                (-30, 30, -50),
        #                (-90, -45, -45),
                       # (150, -30, 50),
                       # (90, 45, 225),
                       # (-30, 30, 130),
                       # (-90, -45, 135),
                       # (150, -30, 230)
                       # ]

        CT_angles = [(-25, 32, -42)]
        # CT_angles = [(30, -30, -45), (30, -30, 45), (30, -30, 135), (30, -30, -135),
        #                 (-150, 30, -45), (-150, 30, 45), (-150, 30, 135), (-150, 30, -135)]

        angles = []
        if args.mode == 'OTLS':
            angles = OTLS_angles
            viewer.camera.zoom = initial_zoom * 0.6
        elif args.mode == 'CT':
            angles = CT_angles
            viewer.camera.zoom = initial_zoom * 0.75
        else:
            print(f"Unsupported image type: {args.mode}")
        best_img_arr = None
        best_avg_intensity = 300.0
        best_idx = 0
        for idx, angle in enumerate(angles):
            viewer.camera.angles = angle
            # snapshot_arr = viewer.screenshot()
            snapshot_arr = viewer.screenshot(path=None)
            snapshot_avg_intensity = np.mean(snapshot_arr)
            # We would like to choose and save the angle that has the least white-space seen, and so look for lower
            # average intensity
            if idx == 0 or snapshot_avg_intensity < best_avg_intensity:
                best_img_arr = snapshot_arr
                best_avg_intensity = snapshot_avg_intensity
                best_idx = idx

        if args.save_dir is None:
            save_path = args.img_dir_path.split('/'+img_name)[0] + "/"+img_name+"_snapshot.png"
        else:
            save_path = os.path.join(args.save_dir, '{}_snapshot.png'.format(img_name))

        if args.rgb:
            shape = np.shape(best_img_arr)
        best_img = Image.fromarray(best_img_arr)
        best_img.save(save_path)
        print(f"Saved snapshot at: {save_path}")
    else:
        if args.animation_mode:
            animation = Animation(viewer)
            viewer.update_console({'animation': animation})
        napari.run()
