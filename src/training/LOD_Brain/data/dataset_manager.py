#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday - February 17 2021, 16:40:14

@authors:
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia

Support function to manage multi-site dataset.
The goal of this file is to manage the dataset and return only the matrices (or filepaths),
needed for training, valid, and testing:
    X_train, X_valid, X_test
    Y_train_paths, Y_valid_pathsid_paths, Y_test_paths

"""

from os.path import join as opj
import os
import sys
import wandb
import nibabel as nib
import numpy as np
from scipy import stats
import pandas as pd
import sklearn as skl
import tensorflow as tf
from loguru import logger
from LOD_Brain import python_utils
from LOD_Brain.data import volume_manager
from LOD_Brain.data import augmentation
from LOD_Brain.config import Config


def load_csv_with_fullpaths(paths_dict):
    """ 
    Load the CSV with volume filepaths and return lists for train/valid/test sets.

    :param paths_dict: dict with all root paths
    :return X_{}/Y_{} for {train, valid, test}
    :return datasets: list of str of dataset names
    """

    # Load csv
    csv = pd.read_csv(opj(paths_dict['Path_in_csv'], paths_dict['Filename_csv']))
    csv = csv.sample(frac=1).reset_index(drop=True)  # shuffle the DataFrame

    # Train
    df_train = csv[csv['set'] == 'train']
    X_train = list(df_train['T1Path'])
    Y_train = list(df_train['GTPath'])
    assert len(X_train) == len(Y_train)

    # Valid
    df_valid = csv[csv['set'] == 'valid']
    X_valid = list(df_valid['T1Path'])
    Y_valid = list(df_valid['GTPath'])
    notes_valid = list(df_valid['notes'])
    assert len(X_valid) == len(Y_valid) and len(notes_valid) == len(Y_valid)

    # Test
    df_test = csv[csv['set'] == 'test']
    X_test = list(df_test['T1Path'])
    Y_test = list(df_test['GTPath'])
    notes_test = list(df_test['notes'])
    assert len(X_test) == len(Y_test) and len(notes_test) == len(Y_test)

    datasets = list(csv['database'])

    return X_train, Y_train, X_valid, Y_valid, notes_valid, X_test, Y_test, notes_test, list(np.unique(datasets))


def createDictOfMeanStd(full_path_mean_subject, full_path_std_subject):
    """ Create a dictionary with all 71 mean and std volumes for z-scoring. """

    voxelwise_mean = {}
    voxelwise_std = {}

    for i in range(len(full_path_mean_subject)):
        # Find the name of the db from filename
        root = (
            full_path_mean_subject[i].split('z_scoring/')[
                1])  # ex. '1000_FCP_FCON1000_ICBM_nii_256_average_subject.nii.gz'
        i_db = root.split('_nii')[0]  # '1000_FCP_FCON1000_ICBM'

        if os.path.exists(full_path_mean_subject[i]):
            voxelwise_mean[i_db] = np.array(nib.load(full_path_mean_subject[i]).get_fdata()).astype(dtype='float32')
            voxelwise_std[i_db] = np.array(nib.load(full_path_std_subject[i]).get_fdata()).astype(dtype='float32')
        else:
            logger.error(f"The zscoring file -{full_path_mean_subject[i]}- do not exist.")
            sys.exit()

    return voxelwise_mean, voxelwise_std


def loadInhomogeneityVolume(fullpath_inho_volume):
    """
    Load the augmentation volume 'inhomogeneity_volume.npy'

    :param fullpath_inho_volume: str with the fullpath of the volume to load
    :return inhomogeneity_volume: numpy volume of size 512^3 used in data augmentation
    """
    if fullpath_inho_volume:
        inhomogeneity_volume = np.load(fullpath_inho_volume)
        logger.info('Loaded inhomogeneity_volume (shape): ' + str(inhomogeneity_volume.shape))
    else:
        logger.warning('inhomogeneity_volume not present!')
        inhomogeneity_volume = None

    return inhomogeneity_volume


def prepareDataset(config: Config):
    """
    Function that load all the csv for training, validation, and testing setting up all
    the filenames and loading 'voxelwise_mean' and 'voxelwise_std'.
    In addition, it creates the csv files with train, val, and test volumes used.

    :param config: Config object
    :return dataset: dict with loaded data (paths only) and few attributes
    """

    # Retrieve needed arguments and set up the input paths
    path_out_folder = config.training.exp_path.as_posix()       # path where everything is saved
    paths_dict = config.data.dict()                             # dict with main paths on where data are
    full_path_in_vol = opj(paths_dict['Path_in_data'], 'T1/')
    full_path_GT = opj(paths_dict['Path_in_data'], 'GT/')
    logger.info('full_path_in_vol: ' + full_path_in_vol)
    logger.info('full_path_GT: ' + full_path_GT)

    # Import all the volume' filenames from the csv
    X_train_paths, Y_train_paths, X_valid_paths, Y_valid_paths, notes_valid, X_test_paths, \
    Y_test_paths, notes_test, Datasets = load_csv_with_fullpaths(paths_dict)

    # Load mean and std (for every database found in 'Datasets')
    if config.data.normalisation == 'z_score_site':
        main_path_zscores = opj(paths_dict['Path_in_data'], 'analysis/z_scoring/')
        scoring_datasets = sorted(list(set(Datasets) - set(['MRBrainS', 'MindBoggle101'])))
        full_path_mean_subject = [main_path_zscores + i_dataset + '_nii_256_average_subject.nii.gz' for
            i_dataset in scoring_datasets]
        full_path_std_subject = [main_path_zscores + i_dataset + '_nii_256_stdDev_subject.nii.gz' for
            i_dataset in scoring_datasets]
        voxelwise_mean, voxelwise_std = createDictOfMeanStd(full_path_mean_subject, full_path_std_subject)

    # Load validation only (actual data - 1 valid vol only)
    data_dims = tuple([config.network.size] * 3)
    num_segmask_labels = config.network.num_classes
    Y_valid_0 = np.array(nib.load(Y_valid_paths[0]).get_fdata())
    assert Y_valid_0.shape == data_dims
    assert len(np.unique(Y_valid_0)) == num_segmask_labels
    
    # Compute class weight
    class_weights = volume_manager.computeClassWeights(Y_valid_0)

    # Update 'experiment_dict' and 'training_log'
    experiment_dict = {}
    experiment_dict['path'] = dict()
    experiment_dict['path'].update({'full_path_GT': full_path_GT})
    if config.data.normalisation == 'z_score_site':
        experiment_dict['path'].update({'full_path_mean_subject': full_path_mean_subject})
        experiment_dict['path'].update({'full_path_std_subject': full_path_std_subject})
    experiment_dict['data'] = dict()
    experiment_dict['data'].update({'X_val.shape': Y_valid_0.shape})
    experiment_dict['data'].update({'data_dims': data_dims})
    experiment_dict['data'].update({'num_segmask_labels': num_segmask_labels})
    experiment_dict['data'].update({'len(X_train_paths)': len(X_train_paths)})
    experiment_dict['data'].update({'len(X_val_paths)': len(X_valid_paths)})
    experiment_dict['data'].update({'len(X_test_paths)': len(X_test_paths)})
    experiment_dict['data'].update({'class_weights': class_weights})
    wandb.log(experiment_dict)

    logger.info('Volume shapes: ' + str(data_dims))
    logger.info('All volumes number: ' + str(len(Y_train_paths) + len(Y_valid_paths) + len(Y_test_paths)))
    logger.info('Training set (#): ' + str(len(X_train_paths)) + ' vols.')
    logger.info('Validation set (#): ' + str(len(X_valid_paths)) + ' vols.')
    logger.info('Testing set (#): ' + str(len(X_test_paths)) + ' vols.')

    # Save CSV with list of files
    python_utils.saveCsvWithDatasetList(X_train_paths, Y_train_paths, opj(path_out_folder, 'training_files.csv'))
    python_utils.saveCsvWithDatasetList(X_valid_paths, Y_valid_paths, opj(path_out_folder, 'validation_files.csv'))
    python_utils.saveCsvWithDatasetList(X_test_paths, Y_test_paths, opj(path_out_folder, 'testing_files.csv'))

    # Load inhomogeneity volume (used for data augmentation)
    inhomogeneity_volume = loadInhomogeneityVolume(paths_dict['Inh_vol_path'])

    # Create a dict with all the material inside
    dataset = {}
    dataset['X_train_paths'] = X_train_paths
    dataset['Y_train_paths'] = Y_train_paths
    dataset['X_valid_paths'] = X_valid_paths
    dataset['Y_valid_paths'] = Y_valid_paths
    dataset['notes_valid'] = notes_valid
    dataset['X_test_paths'] = X_test_paths
    dataset['Y_test_paths'] = Y_test_paths
    dataset['notes_test'] = notes_test

    dataset['data_dims'] = data_dims
    dataset['Datasets'] = Datasets
    if config.data.normalisation == 'z_score_site':
        dataset['voxelwise_mean'] = voxelwise_mean
        dataset['voxelwise_std'] = voxelwise_std
    dataset['num_segmask_labels'] = num_segmask_labels
    dataset['class_weights'] = class_weights

    dataset['inhomogeneity_volume'] = inhomogeneity_volume

    return dataset


def TFDatasetGenerator(config: Config, dataset: dict):
    """
    Manage the Dataset for TF2.x training.
    It creates the datasets type for training and validation sets.

    :param config: Config object
    :param dataset: dict with loaded data (paths only) and few attributes

    :return ds_train, ds_valid: tf.data for training and validation
    """

    def tf_preprocessing(x, y, data_dims, num_labels, normalisation):
        """ Wrapper to call loading functions. Needs to be zscored. Load volume and label."""
        if type(normalisation) is not str:                  # tf.numpy_function converts to <class 'bytes'>
            normalisation = str(normalisation.decode("utf-8"))

        X, Y = volume_manager.loadVolumeAndGT(x, y, data_dims, num_labels, normalisation)
        return X, Y

    @tf.function
    def volumeZscoring(input_vols, zscorings):
        """ Function to z-scoring of loaded-by-generator data
            IN: ((vol, gt), (mean, std))
        """

        # Parse input
        vol, gt = input_vols
        mean, std = zscorings

        # standardize each training volume
        vol = tf.math.subtract(vol, mean)

        # Prevent division by zero
        std = tf.where(tf.equal(tf.zeros(1), std), 1 * tf.ones_like(std), std)
        vol = tf.math.divide(vol, std)

        # Reshape vol
        vol = tf.reshape(vol, [dataset['data_dims'][0], dataset['data_dims'][1], dataset['data_dims'][2], 1])

        return vol, gt

    def createDatasetTF(X_paths, Y_paths):
        """
        Function that takes in input 'X_paths' and 'Y_paths'
        and creates the 'tf.data.Dataset' needed for dealing with TFx.

        :param X_paths: list with T1 filenames
        :param Y_paths: list with GT filenames
        :return tf_dataset: tf.data
        """

        tf_dataset = None
        for i_site in dataset['Datasets']:

            # Note: 'MindBoggle101' has OASIS volumes in it, but they named 'OASIS' not 'OASIS3'
            # which is the name of a dataset in 'dataset['Datasets']' (i.e., no problems)

            # Retrieve X and Y paths
            i_X_paths = [i for i in X_paths if i_site in i]
            i_Y_paths = [i for i in Y_paths if i_site in i]

            assert len(i_X_paths) == len(i_Y_paths)
            if len(i_X_paths) == 0:             # some dataset may not have train/valid/test volumes
                continue

            # Create 'i_site' dataset 
            i_dataset = tf.data.Dataset.from_tensor_slices((i_X_paths, i_Y_paths))
            i_dataset = i_dataset.map(map_func=lambda x, y: tf.numpy_function(tf_preprocessing,
                                                                                inp=[x, y,
                                                                                    dataset['data_dims'],
                                                                                    dataset['num_segmask_labels'],
                                                                                    config.data.normalisation],
                                                                                Tout=[tf.float32,
                                                                                    tf.uint8]),
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

            # If True, intra-site z-scoring ('MindBoggle101' is always volume-based). If False, volume-based
            if (config.data.normalisation == 'z_score_site') and i_site != 'MindBoggle101' and i_site != 'MRBrainS':

                # Prepare normalisation volumes (mean/std) for TF2.x
                i_voxelwise_mean = dataset['voxelwise_mean'][i_site]
                i_voxelwise_std = dataset['voxelwise_std'][i_site]
                i_dataset_zscoring = tf.data.Dataset.from_tensors((i_voxelwise_mean, i_voxelwise_std))
                i_dataset_zscoring = i_dataset_zscoring.repeat(len(i_X_paths))  # .cache()

                # Apply intra-site normalisation
                i_dataset = tf.data.Dataset.zip((i_dataset, i_dataset_zscoring))
                i_dataset = i_dataset.map(volumeZscoring)

            # Update 'dataset' with the i_dataset
            if tf_dataset is None:  # first iteration
                tf_dataset = i_dataset
            else:
                tf_dataset = tf_dataset.concatenate(i_dataset)

        return tf_dataset

    # Call the function with data augmentation pipeline
    if config.augment.augmentation_AF:
        augm_transform = augmentation.get_augm_transforms_AF(dataset['inhomogeneity_volume'], config)
    else:
        augm_transform = augmentation.get_augm_transforms(dataset['inhomogeneity_volume'], config)

    def tf_augment(x, y):
        """ 
        Wrapper to call data augmentation function.

        :param x: t1 numpy array of shape [x, y, z, 1]
        :param y: seg numpy array of shape [x, y, z, n_classes]
        :return x_out, y_out: augmented (t1, mask) couple
        """

        # Apply data augmentation
        transformed = augm_transform(image=x[:, :, :, 0],
                                     mask=np.uint8(np.argmax(y, axis=-1)))
        
        # Reshape output as needed
        x_out = transformed['image'].reshape(x.shape)
        y_out = volume_manager.to_categorical_tensor(transformed['mask'], y.shape[-1]).astype(dtype='uint8')

        if config.augment.z_score_volume:
            x_out = stats.zscore(x_out, axis=None)          # z-score the volume

        return x_out, y_out

    ## Create generator for the training set
    ds_train = createDatasetTF(dataset['X_train_paths'], dataset['Y_train_paths'])
    if config.augment.augmentation:
        ds_train = ds_train.map(map_func=lambda x, y: tf.numpy_function(tf_augment,
                                                                        inp=[x, y],
                                                                        Tout=[tf.float32, tf.uint8]),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(25,  # 'len(dataset['X_train_paths'])' takes too much
                                reshuffle_each_iteration=True)  # Randomly shuffles the elements of this dataset.
    ds_train = ds_train.repeat().batch(config.training.batch_size)  # Combines consecutive elements of this dataset into batches.
    ds_train = ds_train.prefetch(25)  # Creates a Dataset that prefetches elements from this dataset.

    ## Create generator for the validation set   
    ds_valid = createDatasetTF(dataset['X_valid_paths'], dataset['Y_valid_paths'])
    ds_valid = ds_valid.repeat().batch(1)

    return ds_train, ds_valid

