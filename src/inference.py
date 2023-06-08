#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tuesday - June 22 2021, 15:07:59

@author: Michele Svanera, University of Glasgow

See below for more examples

'''

## Imports
import os, sys
import argparse
from os.path import join as opj
import time
from datetime import timedelta
from pathlib import Path
from time import strftime, localtime
import yaml

import numpy as np
import nibabel as nib
from scipy import stats
from scipy.ndimage import label as scipy_label
from scipy.ndimage.filters import gaussian_filter
from nibabel.processing import conform

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
from tensorflow.keras.models import load_model

sys.path.insert(0, '/source/')
from LOD_Brain.data.volume_manager import volume_normalisation
from LOD_Brain.model import utils
from LOD_Brain.model.layers import BottleNeck, UpBottleNeck, Plain, UpPlain
from LOD_Brain.model.losses import losses_dict, metrics_dict


## Paths and Constants
Path_in_models = '/model/'
Path_out_dir = '/output/'
Shape_needed = (256, 256, 256)

## Functions

def get_largest_connected_component(mask, structure=None):
    """Function to get the largest connected component for a given input.
    :param mask: a 2d or 3d label map of boolean type.
    :param structure: numpy array defining the connectivity.
    """
    components, n_components = scipy_label(mask, structure)
    return components == np.argmax(np.bincount(components.flat)[1:]) + 1 if n_components > 0 else mask.copy()


def seg_keep_only_biggest_components(seg_in, labels = None):
    ''' Keep only the biggest component of each class (no 'background' and 'ventricles') and the mask as a whole.
    '''

    if labels is None:
        labels = np.unique(seg_in)

    # Keep only the biggest component of the seg mask
    seg_out = seg_in.copy()
    seg_out[~get_largest_connected_component(seg_out > 0)] = 0

    # For each label, keep only the biggest component
    for i_label in labels:
        if i_label == 0 or i_label == 4:        # 'background' and 'ventricles' aren't to scan
            continue
        seg_out[(~get_largest_connected_component(seg_in == i_label)) & (seg_in == i_label)] = 0        

    return seg_out


def prob_keep_only_biggest_components(prob_in, threshold = 0.25):
    """
    Keep only the biggest component of each class (no 'background' and 'ventricles') and the mask as a whole.
    :param prob_in: probability map, shape = (x,y,z,n_classes)
    :param threshold: for the biggest map as a whole
    :return prob_out probability map with only biggest components, shape = (x,y,z,n_classes)
    """

    prob_out = prob_in.copy()

    # For each label, keep only the biggest component
    for i_label in range(1, prob_in.shape[-1]):
        if i_label == 4:        # 'ventricles' isn't to scan
            continue
        mask_out = prob_out[..., i_label] > threshold
        prob_out[(~get_largest_connected_component(mask_out)) & (mask_out), i_label] = 0        

    # Keep only the biggest component of the output seg mask
    mask_out = np.argmax(prob_out, axis=-1)
    mask_out = seg_keep_only_biggest_components(mask_out, labels = [1,3,5,6]) > 0
    prob_out[~(mask_out > 0), :] = 0

    # Copy back the probabilities for 'background'
    prob_out[..., 0] = prob_in[..., 0]

    return prob_out


def findListOfAnatomical(path_in, identifier='nii.gz'):

    all_anat = []
    for root, _, files in os.walk(path_in):
        for i_file in files:
            if i_file.endswith(identifier):
                all_anat.append(root + '/' + i_file)
    
    all_anat = sorted(list(np.unique(all_anat)))
    return [i for i in all_anat if '/._' not in i]


def pad_volume(vol_in, pad_size=Shape_needed):
    ''' Pad the input volume 'vol_in' of 'pad_size'. '''
    
    padding_needed = tuple([(pad_size[i] - vol_in.shape[i]) for i in range(len(vol_in.shape))])
    padding_needed = tuple([tuple([int(np.floor(i/2)), int(np.ceil(i/2))]) for i in padding_needed])
    vol_out = np.pad(vol_in, padding_needed, 'constant', constant_values=0)
    
    return vol_out, padding_needed
    
    
def preprocessing_t1w(X_path, path_out='/output/'):
    ''' This function checks if modification on orientation and shape were made and restore the orginals. '''

    # Load the volume
    i_X = nib.load(X_path)
        
    # Check if needed to transform in 'LIA' space and paddding and save it
    vol_info = {}
    vol_info['orientation'] = nib.aff2axcodes(i_X.affine)
    vol_info['shape'] = i_X.shape
    vol_info['anat_path'] = opj(path_out, 'T1w.nii.gz')
    vol_info['padding'] = ((0, 0), (0, 0), (0, 0))
    vol_info['affine'] = i_X.affine.copy()
    vol_info['header'] = i_X.header.copy()
    nib.save(i_X, vol_info['anat_path'])
    
    # Deal with volumes with 4 dimensions (ex. (300, 320, 208, 1))
    if len(vol_info['shape']) > 3:
        i_X_data = i_X.get_fdata()[...,0]
        i_X = nib.Nifti1Image(i_X_data, affine=vol_info['affine'], header=vol_info['header'])
        nib.save(i_X, vol_info['anat_path'])
        vol_info['shape'] = i_X.shape

    # Deal with volumes bigger than 'Shape_needed' (ex. (300, 320, 208))
    if any(vol_info['shape'][i] > Shape_needed[i] for i in range(3)):
    
        # Find the new shape and find the new transformation
        new_shape = [min([Shape_needed[i], vol_info['shape'][i]]) for i in range(3)]
        padding_removed = tuple([vol_info['shape'][i] - new_shape[i]  for i in range(3)])
        new_transform = np.eye(4)
        new_transform[0:3, -1] = [-i for i in padding_removed]
        vol_info['affine'] = vol_info['affine'] @ np.linalg.inv(new_transform)

        # Apply to data and save
        i_X_data = i_X.get_fdata()[:new_shape[0], :new_shape[1], :new_shape[2]]
        i_X = nib.Nifti1Image(i_X_data, affine=vol_info['affine'], header=vol_info['header'])
        nib.save(i_X, vol_info['anat_path'])
        vol_info['shape'] = i_X.shape

    # Deal with volumes smaller than 'Shape_needed' (ex. (200, 200, 200))
    if not vol_info['shape'] == Shape_needed:
        i_X_data, vol_info['padding'] = pad_volume(i_X.get_fdata(), Shape_needed)
        new_transform = np.eye(4)
        new_transform[0:3, -1] = [j[0] for j in vol_info['padding']]
        vol_info['affine'] = vol_info['affine'] @ np.linalg.inv(new_transform)

                # Apply to data and save
        i_X = nib.Nifti1Image(i_X_data, affine=vol_info['affine'], header=vol_info['header'])
        nib.save(i_X, vol_info['anat_path'])
        vol_info['shape'] = i_X.shape

    if not vol_info['orientation'] == ('L', 'I', 'A'):       # Deal with orientation: it needs to be 'LIA'
        i_X = conform(i_X, out_shape=i_X.shape, voxel_size=vol_info['header']['pixdim'][1:4], orientation='LIA')
        nib.save(i_X, vol_info['anat_path'])
    
    return i_X, vol_info['anat_path']

    
def predict_volume(model, X_path, out_dir, normalisation_type, save_prob_map=False, save_T1w=True):
    """
    Function to make model prediction.

    :param model: trained model
    :param X_path: T1w nifti volume filepath
    :param out_dir: out folder where to store predicted volumes
    :param normalisation_type: manage normalisation as done in training
    :param save_prob_map: save or not the probability map (heavy)
    """
    
    # Load and preprocess the volume
    i_X, anat_path = preprocessing_t1w(X_path, path_out = out_dir)
    i_X_vol = np.array(i_X.get_fdata()).astype(dtype='float32')
    assert i_X_vol.shape == (256, 256, 256), f'Volume {X_path} shape is != (256, 256, 256)'

    # Normalisation
    i_X_vol = volume_normalisation(i_X_vol, normalisation_type)   
    i_X_vol = np.expand_dims(i_X_vol, axis=[0,-1])
    
    # Predict the value from the model
    y_test_prob_map = np.squeeze(model.predict(i_X_vol, verbose=False))

    # compute the argmag (hard-thresholding) of the proabability map to obtain the labelled volume
    y_test_pred = np.argmax(y_test_prob_map, axis=-1).astype(dtype='uint8')

    # Compose the output filename and headers
    i_subj_id = os.path.basename(X_path)[:-7]
    i_fullfilename_out = opj(out_dir, i_subj_id + '_t1.nii.gz')
    i_fullfilename_pred = opj(out_dir, i_subj_id + '_predicted_volume.nii.gz')
    i_fullfilename_prob = opj(out_dir, i_subj_id + '_prob_map_volume.nii.gz')

    # Save out files: gt, predicted, and probability map
    t1_np = i_X_vol[0, :, :, :, 0]
    t1_np -= np.min(t1_np)                      # return to range [0, 255]
    t1_np /= np.max(t1_np)
    t1_np = np.round(t1_np * 255).astype(np.uint8)
    t1_vol = nib.Nifti1Image(t1_np, affine=i_X.affine, header=i_X.header)
    if save_T1w:
        nib.save(t1_vol, i_fullfilename_out)

    pred_vol = nib.Nifti1Image(y_test_pred, affine=i_X.affine, header=i_X.header)
    nib.save(pred_vol, i_fullfilename_pred)

    prob_map_vol = nib.Nifti1Image(y_test_prob_map, affine=i_X.affine, header=i_X.header)
    if save_prob_map:
        nib.save(prob_map_vol, i_fullfilename_prob)

    # Cleaning
    del i_X_vol, i_X, y_test_prob_map, prob_map_vol
    os.remove(anat_path)    # remove i_X

    return i_fullfilename_pred


def load_training_model(model_keyword: str = '7im5hf6z'):
    """ 
    Search through the folders in 'Path_in_models' and returns the model path.

    :param model_keyword: keywork of the experiment  
    :return model
    """
    
    # Find the folder needed
    experiment_dir = [x for x in Path(Path_in_models).iterdir() if x.is_dir() and model_keyword in str(x)]
    assert len(experiment_dir) == 1, f'Model {model_keyword} not found! Please check.'
    #full_path_model = Path(experiment_dir[0], 'files', 'checkpoints')
    full_path_model = Path(experiment_dir[0], 'checkpoints')

    # Restore the (best) model weights (nomenclature: "model.{epoch:02d}-{val_compute_per_channel_dice:.2f}.h5")
    # last_weights_file = sorted(full_path_model.glob('model*.h5'), key=lambda x: x.name.split('.')[-2])[-1]
    last_weights_file = sorted(full_path_model.glob('model*.h5'), key=lambda x: (x.name.split('.')[-2],  x.name.split('.')[-3]))[-1]
    assert last_weights_file.is_file(), f'Model {last_weights_file} not found! Please check.'

    custom_objects = {"BottleNeck": BottleNeck, "UpBottleNeck": UpBottleNeck,
                   'Plain': Plain, 'UpPlain': UpPlain}
    custom_objects.update(losses_dict)
    custom_objects.update(metrics_dict)
    custom_objects['tversky_metric'] = custom_objects.pop('tversky_custom_metric')
    model = load_model(last_weights_file, custom_objects=custom_objects)#, compile=False)

    # Load the yaml file to read which normalisation was used 
    with open(Path(experiment_dir[0], 'config.yaml'), "r") as stream:
        try:
            config = yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            print(exc)

    if 'normalisation' in list(config['data'].keys()):
        normalisation_type = config['data']['normalisation']
    else:                                                   # old convention, with variable 'intrasite_zscoring'
        normalisation_type = 'z_score_volume'

    return model, normalisation_type


## Main
def inference_on_volume(training_name, vol_in, out_folder, POST_PROCESSING, print_outcome=True, save_T1w=False):
    start_time = time.time()

    # Search the folder and load the model
    model, normalisation_type = load_training_model(model_keyword = training_name)

    # Create an ouput folder named as the file
    local_time_str = strftime("%Y-%m-%d %H:%M:%S", localtime())
    local_time_str = (local_time_str.replace(':','-')).replace(' ','_')
    out_dir = Path(out_folder, 'testing_' + local_time_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        tf.keras.utils.plot_model(model, to_file=opj(out_dir, 'model.png'), show_shapes=True)
    except:
        pass

    # Check if 'vol_in' is a path or a file
    if os.path.isdir(vol_in):        
        all_anat = findListOfAnatomical(vol_in)
    else:
        all_anat = [vol_in]

    # Predict and save the volumes
    for i_anat in all_anat:
        # Prediction
        i_pred_path = predict_volume(model, i_anat, out_dir, normalisation_type, save_prob_map=POST_PROCESSING, save_T1w=save_T1w)

        if POST_PROCESSING:

            # Change and save probability map
            prob_map_path = i_pred_path.replace('predicted', 'prob_map')        # Load 'prob_map_volume.nii.gz'
            prob_map = nib.load(prob_map_path)
            prob_map_data = prob_map.get_fdata()

            # Smoothing the prob. maps
            prob_map_data_out = []
            for i in range(prob_map_data.shape[-1]):
                prob_map_data_out.append(gaussian_filter(prob_map_data[:, :, :, i], sigma=0.5))
            prob_map_data = np.stack(prob_map_data_out, axis=3)

            # Keep only 1 component per class
            prob_map_data = prob_keep_only_biggest_components(prob_map_data)
            prob_map = nib.Nifti1Image(prob_map_data, affine=prob_map.header.get_sform(), header=prob_map.header)
            nib.save(prob_map, prob_map_path)
            
            # Save segmentation map
            pred_mask = nib.load(i_pred_path)
            gt_pp = np.argmax(prob_map_data, axis=-1)
            pred_mask_pp = nib.Nifti1Image(gt_pp, affine=pred_mask.header.get_sform(), header=pred_mask.header)
            nib.save(pred_mask_pp, i_pred_path)

    if print_outcome:
        print('\n** Done! Total time (' + str(len(all_anat)) + ' vols.): ' + str(timedelta(seconds=(time.time() - start_time))) + ' (hh:mm:ss.ms) **\n')
    return i_pred_path, gt_pp


if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description='Inference on a single volume')

    parser.add_argument('--training_name', default='7im5hf6z', type=str, help='Trained model')
    parser.add_argument('--vol_in', type=str, default='/data/', help='Volume to test or folder')
    parser.add_argument('--out_folder', default=Path_out_dir, type=str, help='Output folder')
    # parser.add_argument('--prob_maps', type=bool, default=False, help='Probability maps in output: yes or no')
    parser.add_argument('--no_post', action='store_false', help='Apply post-processing: add flag to DO NOT apply post-proc.')
    args = parser.parse_args()
        
    # Model args
    training_name = args.training_name
    vol_in = args.vol_in
    out_folder = args.out_folder
    # prob_map_out = args.prob_maps
    POST_PROCESSING = args.no_post

    _ = inference_on_volume(training_name, vol_in, out_folder, POST_PROCESSING)



## Example
training_name = '7im5hf6z'
vol_in = '/data/'#sub-01_T1w.nii.gz'
POST_PROCESSING = True
out_folder = Path_out_dir
