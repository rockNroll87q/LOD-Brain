#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday - February 17 2021, 16:40:14

@authors:
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia

General python utility functions.
"""

import os
import sys
import nvidia_smi
import pandas as pd
import tensorflow as tf
import numpy as np
from loguru import logger
from skimage import color


def saveCsvWithDatasetList(x_paths: list, y_paths: list, filename_out: str):
    """
    Utility to save a CSV file with a list of filenames.

    :param x_paths: list of T1 filepaths
    :param y_paths: list of GT filepaths
    :param filename_out: full filename for the csv
    """
    dataset_name = (os.path.basename(filename_out)).split('_')[0]
    df = pd.DataFrame({dataset_name: x_paths,
                       'GT': y_paths})

    df.to_csv(filename_out, index=False)

    return


def selectGPUsAvailability():
    """
    I suppose we are going to use deepnet4-8: 4 x RTX8000 (45GB)
    It also tests if GPUs have free memory on.
    :return
        int {0-2}:  which GPU to use
        False:      means no GPU available -> need to exit
    """

    # Find the first available
    for i_gpu in reversed(range(4)):

        try:
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i_gpu)
            mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        except Exception:
            print('Warning: GPU did not accessed')
            break

        # If free mem is more than 35GB, then select this gpu
        if mem_res.free / (1024. ** 3) > 35.:  # free more than 35GB of VRAM -> found my GPU
            return i_gpu

    return False  # Means 'deepnet5/8' no GPU available


def findGPUtoUse():
    """
    Function that find the GPU available and set the others not visible.

    :return which_gpu: int {0-2}  which GPU to use
    """

    # Find which GPU to use
    which_gpu = selectGPUsAvailability()  # int {0-3}:  which GPU to use
    if type(which_gpu) == bool:  # False:      means no GPU available -> need to exit
        logger.info('WARNING: No GPU found!')
        sys.exit(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)  # which GPU make visible
    logger.info('GPU used: ' + str(which_gpu))

    # Configure GPUs to prevent OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    return which_gpu


def select_colors_from_gt_slice(i_gt):
    """ Finds the classes in the volume and retrieve the color needed.

    :param i_gt: segm volume
    :return colors: array with rgb colors in float
    """
    LABEL_COLORS = [(0, 0, 0),          # 0: background
                    (70, 130, 180),     # 1: WM
                    (245, 245, 245),    # 2: ganglia
                    (205, 62, 78),      # 3: GM
                    (120, 18, 134),     # 4: ventricles
                    (196, 58, 250),     # 5: cerebellum
                    (0, 148, 0),        # 6: brain stem
                    (234, 0, 31),       # 7: other (used for shepp-logan vol.s)
                    (200, 200, 200),    # 8: other
                    (61, 0, 66),        # 9: other
                    (178, 142, 77),     # 10: other
                    ]
                    
    labels_found = np.unique(i_gt)
    if len(labels_found) > len(LABEL_COLORS):           # for more than 7 labels, select random colors
        LABEL_COLORS = np.random.randint(255, size=(len(labels_found), 3))
        LABEL_COLORS[0] = (0, 0, 0)         # background
        colors = list(np.array(LABEL_COLORS)[labels_found])
    else:
        colors = list(np.array(LABEL_COLORS)[labels_found])

    # Need to convert it in floats
    for i in range(len(colors)):
        colors[i] = colors[i] / 255.

    return colors


def format_prediction_into_gif(t1, segm,
                               n_slices_to_display: int = 50,
                               alpha: float = 0.3):
    """
    Function to format the predicted volume (t1 + gt) into a single volume

    :param t1: expected tensor of shape [1, x, y, z, 1]
    :param segm: expected tensor of shape [1, x, y, z, n_classes]
    :param n_slices_to_display: number of slices to display
    :param alpha: segm transparency
    :return vol_out: t1 + gt in overlay of shape [x, y, z, RGB]
    """

    # Convert into numpy
    t1_np = t1[0, :, :, :, 0]  #  (t1.numpy())[0, :, :, :, 0]
    seg_np = (np.argmax(segm[0, ...], axis=-1)).astype(np.uint8)

    # Rescale t1 into range [0, 255], type uint8
    t1_np -= np.min(t1_np)
    t1_np /= np.max(t1_np)
    t1_np = np.round(t1_np * 255).astype(np.uint8)

    slices_to_display = np.linspace(0,  # Slices to store
                                    t1_np.shape[-1],
                                    min(n_slices_to_display, t1_np.shape[2]),
                                    endpoint=False,
                                    dtype=int)

    # Create t1 with predicted segm in overlay
    vol_out = []
    for i_slice in slices_to_display:
        
        # Give only labels present in 'i_slice'
        colors = select_colors_from_gt_slice(seg_np[:, i_slice, :])
        
        # Create image, transpose (for a better visualisation), and stack
        i_image = color.label2rgb(label=seg_np[:, i_slice, :],
                                  image=t1_np[:, i_slice, :],
                                  kind='overlay',
                                  alpha=alpha,
                                  image_alpha=1,
                                  colors=colors)
        i_image = (i_image.transpose(1, 0, 2))[::-1, :, :]      # show in proper orientation
        i_image = np.round(i_image * 255).astype(np.uint8)      # cast and set the range [0, 255]
        vol_out.append(i_image)

    # Create output in the needed format: [time, channels, width, height]
    vol_out = (np.stack(vol_out)).transpose([0, 3, 1, 2])

    return vol_out
