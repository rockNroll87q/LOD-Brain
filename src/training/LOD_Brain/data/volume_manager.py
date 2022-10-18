#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday - February 17 2021, 16:40:14

@authors:
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia

Functions to load and manage Volume data.
"""

import nibabel as nib
import numpy as np
import os
import pandas as pd
import re
import sys
import tensorflow as tf
from loguru import logger
from os.path import join as opj
import sklearn as skl
from scipy import stats


def to_categorical_tensor(x3d, n_cls):
    """ 
    Convert numpy segm volume for categorial tf

    :param x3d: seg numpy array of shape [x, y, z, 1]
    :param n_cls: n_classes
    :return y4d: seg numpy array of shape [x, y, z, n_classes]
    """
    x, y, z = x3d.shape
    x1d = x3d.ravel()
    y1d = tf.keras.utils.to_categorical(x1d, num_classes=n_cls)
    y4d = y1d.reshape([x, y, z, n_cls])
    return y4d


def findListOfAnatomical(path_in, identifier='nii.gz'):
    all_anat = []
    for root, _, files in os.walk(path_in):
        for i_file in files:
            if i_file.endswith(identifier):
                all_anat.append(root + '/' + i_file)

    all_anat = sorted(list(np.unique(all_anat)))
    return [i for i in all_anat if '/._' not in i]


def findCorrespondingGTPath(full_path_GT, full_path_in_vol, all_volume_fullpathname, GT_to_predict):
    all_anat_GT = []
    for i_vol in all_volume_fullpathname:
        # Find the correspondent ID
        subj_id, _, anat_name = i_vol.replace(full_path_in_vol, '').split('/')
        all_anat_GT.append(full_path_GT + subj_id + '/' + GT_to_predict)

    return all_anat_GT


def excludeTestingVolume(all_volumes, testing_volume, full_path_in_vol):
    """ Given a list of all volumes, exclude the testing volume.
        Used for behavioural testing volumes.
    """

    out_list = []
    excluded_list = []
    for i_vol in all_volumes:
        i_vol_ID = (i_vol.split(full_path_in_vol)[1]).split('/')[0]
        if i_vol_ID not in testing_volume:
            out_list.append(i_vol)
        else:
            excluded_list.append(i_vol)

    return out_list, excluded_list


def excludeTestingVolumeBIDS(all_volumes, testing_volume):
    """ Given a list of all volumes, exclude the testing volume.
        Used for behavioural testing volumes.
    """

    out_list = []
    excluded_list = []
    for i_vol in all_volumes:

        if np.sum([1 for j_test in testing_volume if j_test in i_vol]) == 0:
            out_list.append(i_vol)
        else:
            excluded_list.append(i_vol)

    return out_list, excluded_list


def volumeZscoring(input_vol, voxelwise_mean, voxelwise_std):
    """
    Function to z-scoring of loaded-by-generator data
    """

    # standardize each training volume
    input_vol -= voxelwise_mean

    # Prevent division by zero
    voxelwise_std[voxelwise_std == 0] = 1
    input_vol /= voxelwise_std

    return input_vol


def checkVolumeIntegrity(input_vol, vol_name):
    """
    Function to check volume integrity
    """

    n_nan = np.sum(np.isnan(input_vol))
    n_inf = np.sum(np.isinf(input_vol))

    if n_nan != 0:
        logger.info('ERROR 23.1: %d NaN(s) found in volume "%s"!' % (n_nan, vol_name))
        sys.exit(0)

    if n_inf != 0:
        logger.info('ERROR 23.1: %d inf(s) found in volume "%s"!' % (n_inf, vol_name))
        sys.exit(0)


def checkVolumeGTnameCorrespondence(vol_name, GT_name):
    """
    Function to check that the GT and the volume match based on the ID.
    """

    # works with both: original volume and augmented
    root_path = findCommonPath(vol_name, GT_name)
    anat_folder = (vol_name.split(root_path)[1]).split('/')[0]
    gt_folder = (GT_name.split(root_path)[1]).split('/')[0]

    vol_id = (vol_name.split(anat_folder + '/')[1]).split('/')[0]
    gt_id = (GT_name.split(gt_folder + '/')[1]).split('/')[0]

    if vol_id[-7:] == '.nii.gz':  # if there is still the extension
        vol_id = vol_id[:-7]

    if vol_id != gt_id:
        logger.info('ERROR 24.1: Volume: "%s" and GT: "%s" do NOT match!' % (vol_id, gt_id))
        sys.exit(0)


def findCommonPath(str1, str2):
    """ Takes 2 strings (paths) and find the root that they have in common
    """
    if len(str1) < len(str2):
        min_str = str1
        max_str = str2
    else:
        min_str = str2
        max_str = str1

    matches = []
    min_len = len(min_str)
    for b in range(min_len):
        for e in range(min_len, b, -1):
            chunk = min_str[b:e]
            if chunk in max_str:
                matches.append(chunk)

    return os.path.dirname(max(matches, key=len))


def loadSingleVolume(vol_filename, voxelwise_mean, voxelwise_std, data_dims, th=None):
    """
    Function to load and check a single volume.
    IN:
        vol_filename: full path name of the volume to load
        th: if specified (=number) then threshold the data
    OUT:
        vol: loaded volume
    """

    # load the actual volume
    vol = np.array(nib.load(vol_filename).get_fdata()).astype(dtype='float32')

    # check if everything is ok
    if not all(True for i, y in zip(vol.shape, data_dims) if i == y):  # vol.shape != data_dims:
        logger.error('ERROR 23.2.1: volume "%s" size mismatch: exiting...' % vol_filename)
        sys.exit(0)

    if th:
        vol[vol < th] = 0

    vol = volumeZscoring(vol, voxelwise_mean, voxelwise_std)

    checkVolumeIntegrity(vol, vol_filename)

    return vol


def loadSingleGTVolume(vol_filename, data_dims):
    """
    Function to load and check a single volume.
    IN: 
        vol_filename: full path name of the volume to load
    OUT:
        vol: loaded volume 
    """

    # load the actual volume (1 single class to predict)
    vol = np.array(nib.load(vol_filename).get_fdata()).astype(dtype='uint8')

    # if vol is a probability vol, then error (to difficult to deal with it here)
    if len(vol.shape) > 3:
        logger.info('ERROR 23.2.2: volume "%s" has more than 3 dimensions: exiting...' % vol_filename)
        sys.exit(0)

    # check if everything is ok
    if not all(True if(i==y) else False for i, y in zip(vol.shape, data_dims)):  # vol.shape != data_dims:
        logger.info('ERROR 23.2.3: volume "%s" size mismatch: exiting...' % vol_filename)
        sys.exit(0)

    checkVolumeIntegrity(vol, vol_filename)

    return vol


def loadDatasetFromListFiles(full_path_volumes, y, voxelwise_mean, voxelwise_std, data_dims, num_segmask_labels=0):
    """
    Function to load a list of volumes with related GT.
    IN: 
        full_path_volumes: full path name of the volumes to load
        voxelwise_mean, voxelwise_std: matrices OR list with multiple mean and std from different sites
        y: full path name of the GT to load
    OUT:
        X_data: loaded volumes (tf format)
        Y_data: loaded GT (tf format)
    """

    X_data = []
    y_data = []

    # Load the first to find the N_classes (background included) -> WARNING: what if a class is not present!!
    if num_segmask_labels == 0:  # if not specified, compute it dinamically
        num_segmask_labels = np.unique(loadSingleGTVolume(y[0], data_dims)).shape[0]

    for vol_fullFilename, GT_fullFilename in zip(full_path_volumes, y):
        # Retrieve the mean/std volumes
        i_voxelwise_mean, i_voxelwise_std = findScoringVolume(vol_fullFilename, voxelwise_mean, voxelwise_std)

        # Load the volume with relative GT
        i_vol_loaded = loadSingleVolume(vol_fullFilename, i_voxelwise_mean, i_voxelwise_std, data_dims)
        i_GT_loaded = loadSingleGTVolume(GT_fullFilename, data_dims)

        # Prepare data for the network
        i_vol_loaded = i_vol_loaded.reshape((data_dims[0], data_dims[1], data_dims[2], 1))
        i_GT_loaded = to_categorical_tensor(i_GT_loaded, num_segmask_labels).astype(dtype='uint8')  # 'int32')

        # Populate the list
        X_data.append(i_vol_loaded)
        y_data.append(i_GT_loaded)

        # Check new classes didn't show up
        checkVolumeGTnameCorrespondence(vol_fullFilename, GT_fullFilename)

    # Concatenate data
    X_data = np.stack(X_data, axis=0)
    Y_data = np.stack(y_data, axis=0)

    #    np.save('X_data.npy',X_data)
    #    np.save('Y_data.npy',Y_data)

    return X_data, Y_data


def findScoringVolume(X_path, voxelwise_mean, voxelwise_std):
    """ Function that takes in INPUT a dict of mean/std and a volume path
        and return a couple of volumes (mean and std) to use for z-scoring.
    """
    Datasets = list(voxelwise_mean.keys())

    # Take care of the case in which 
    if type(voxelwise_mean) == dict:
        # Discover which mean and std use based on the name of the db
        filename = os.path.basename(str(X_path))
        i_dbs = [i for i in Datasets if i in filename]  # Search for the dataset name

        # Check that I find only one
        if len(i_dbs) == 1:
            i_db = i_dbs[0]
        else:
            logger.warning('ERROR 25.1: databases found "%s"...' % i_dbs)
            sys.exit(0)

        i_voxelwise_mean = voxelwise_mean[i_db]
        i_voxelwise_std = voxelwise_std[i_db]
    else:
        i_voxelwise_mean = voxelwise_mean
        i_voxelwise_std = voxelwise_std

    return i_voxelwise_mean, i_voxelwise_std


def loadVolumeAndGT(X_path, Y_path, data_dims, num_segmask_labels, normalisation='z_score_volume'):
    """
    Function to load a single volume with related GT. Used by TF2.x.
    IN: 
        X_path: full path of the volume to load
        Y_path: full path of the GT to load
        voxelwise_mean, voxelwise_std: matrices OR list with multiple mean and std from different sites
        normalisation: If  'z_score_site', it applies intra-site z-scoring. If 'z_score_volume', volume-based z-scoring. If 'rescaling' or 'rescaling_a_b', range [0, 1] or [a, b] (with clipping)
    OUT:
        X_data: loaded volumes (tf format)
        Y_data: loaded GT (tf format)
    """

    # Convert paths in strings
    X_path_str = str(bytes.decode(X_path))
    Y_path_str = str(bytes.decode(Y_path))

    # Load the volume with relative GT
    X_data = np.array(nib.load(X_path_str).get_fdata()).astype(dtype='float32')
    i_GT_loaded = loadSingleGTVolume(Y_path_str, data_dims)

    # Check shapes
    if not (X_data.shape == (data_dims[0], data_dims[1], data_dims[2])):
        logger.info('ERROR 25.1.1: volume "%s" has shape != (256, 256, 256): exiting...' % X_path_str)
        sys.exit(0)

    # Normalisation
    if normalisation == 'z_score_site':                     # If 'z_score_volume', it applies intra-site z-scoring (done in 'TFDatasetGenerator')
        if 'MindBoggle101' in X_path_str or 'MRBrainS' in X_path_str:   # By volume for these datasets
            X_data = stats.zscore(X_data, axis=None)
    elif normalisation == 'z_score_volume':                 # If 'z_score_volume', volume-based z-scoring
        X_data = stats.zscore(X_data, axis=None)
    elif normalisation == 'rescaling':                      # If 'rescaling', rescale in range [0, 1]
        X_data = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data)) * 1.         
    elif normalisation == 'rescaling_a_b':                  # If 'rescaling_a_b', rescale in range a-b normalization (with clipping)
        X_data = np.clip(X_data, np.percentile(X_data, 0.5), np.percentile(X_data, 99.5))
        X_data = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data) + 0.001) * 1.     # back to [0, 1]
    else:
        raise ValueError('Error 235: normalisation value not recognised.')

    if not (normalisation == 'z_score_site'):
        X_data = np.reshape(X_data, [data_dims[0], data_dims[1], data_dims[2], 1])

    # Prepare data for the network
    Y_data = to_categorical_tensor(i_GT_loaded, num_segmask_labels).astype(dtype='uint8')

    # Check new classes didn't show up
    checkVolumeGTnameCorrespondence(X_path_str, Y_path_str)

    return X_data, Y_data


def fromGTtoYlabels(GT, value_to_classify, vol_names):
    """
    Function to extract the 'y' from the 'GT' file.
    It matches the 'session' in the 'GT'.
    IN:
        GT: pandas file with the GT
    OUT:
        vol_names_with_GT: list of volumes of which we have the GT required
        y: GT
    """

    # Create a pandas file starting from 'vol_names'
    filename_of_volumes = [os.path.basename(i_fullpath) for i_fullpath in vol_names]
    df_subject_names = pd.DataFrame({'filename': filename_of_volumes})

    # Try to match the 'df_subject_names' with the 'GT' loaded
    df = pd.merge(GT,
                  df_subject_names,
                  how='inner',
                  on=['filename'])

    # Parse output
    y = list(df[value_to_classify])

    vol_names_with_GT = list(df['filename'])
    vol_names_with_GT = [os.path.dirname(vol_names[0]) + '/' + i for i in vol_names_with_GT]

    return vol_names_with_GT, y


def convertGTtoNumerical(y, GT_to_predict):
    """
    Convert string GT (ex. 'M' or 'F') to numerical (ex. 0, 1)
    
    """

    unique_values = np.unique(y)
    n_classes_classification = unique_values.shape[0]

    y_out = np.zeros((len(y),), dtype=int)
    dictionary_GT = {}

    for i_indx, i_value in enumerate(unique_values):
        dictionary_GT[i_value] = i_indx
        y_out[[i == i_value for i in y]] = i_indx

    return y_out, n_classes_classification, dictionary_GT


def checkNanValues(all_fullnames, y):
    """ Check if there are NaN values in the GT and delete the row.
    IN:
        all_fullnames
        y
    OUT:
        all_fullnames_out
        y_out
    """

    all_fullnames_out = list(np.copy(all_fullnames))
    y_out = np.copy(y)

    indx_to_keep = np.where(~np.isnan(y))
    all_fullnames_out = list(np.array(all_fullnames)[indx_to_keep])
    y_out = list(np.array(y)[indx_to_keep])

    return all_fullnames_out, y_out


def verifyExistenceListOfVolume(vol_paths):
    """ Given a list of paths in input, check if all volumes exist. """

    for i_path in vol_paths:

        # check the volume exist
        if not os.path.exists(i_path):
            logger.info('ERROR 39.1: volume "%s" size mismatch: exiting...' % (i_path))
            sys.exit(0)

    return

def computeClassWeights(Y_val):
    """ 
    Compute class weights to feed the fit function. 

    :param Y_val: seg numpy array of shape [x, y, z, n_classes]
    :return dict_weigths: dict with weights
    """

    if len(Y_val.shape) == 4:
        y_val = np.argmax(Y_val, axis=-1)
    else:
        y_val = Y_val
    classes = np.unique(y_val)
    weigths = skl.utils.class_weight.compute_class_weight('balanced',classes,y_val.flatten())

    return dict((i,v) for i,v in enumerate(weigths))


def volume_normalisation(i_X_vol, normalisation):
    """
    Function to normalise the data

    :param i_X_vol: input volume
    :param normalisation: normalisation type, can be: [z_score_volume, z_score_site, rescaling, rescaling_a_b]
    :param i_X_vol_out: output normalised volume
    """

    # Normalisation
    if (normalisation == 'z_score_site') or (normalisation == 'z_score_volume'):        # z-score the volume before testing
        i_X_vol_out = stats.zscore(i_X_vol, axis=None)                  
    elif (normalisation == 'rescaling'):                                                # If 'rescaling', rescale in range [0, 1]
        i_X_vol_out = (i_X_vol - np.min(i_X_vol)) / (np.max(i_X_vol) - np.min(i_X_vol)) * 1.   
    elif (normalisation == 'rescaling_a_b'):                                            # If 'rescaling_a_b', rescale in range a-b normalization (with clipping)
        i_X_vol_out = np.clip(i_X_vol, np.percentile(i_X_vol, 0.5), np.percentile(i_X_vol, 99.5))
        i_X_vol_out = (i_X_vol - np.min(i_X_vol_out)) / (np.max(i_X_vol_out) - np.min(i_X_vol_out) + 0.001) * 1.     # back to [0, 1]
    else:
        raise ValueError('Error 236: normalisation value not recognised.')

    return i_X_vol_out