#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday - February 17 2021, 16:40:14

@authors:
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia

Functions to load and manage Volume data.
"""

from LOD_Brain.config import Config
import os
from os.path import join as opj
from loguru import logger
import numpy as np
import pandas as pd
import nibabel as nib
import wandb
from LOD_Brain.data import volume_manager
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def predict_and_save_out_volume(model, test_data, out_dir, save_prob_map=False):
    """
    Function to test numerically the model.

    :param model: trained model
    :param test_data: struct with data needed: T1w and GT volumes, pathnames, header
    :param out_dir: out folder where to store predicted volumes
    :param save_prob_map: save or not the probability map (heavy)
    """
    
    # Predict the value from the model
    y_test_prob_map = np.squeeze(model.predict(test_data['i_t1'], verbose=False))

    # compute the argmag (hard-thresholding) of the proabability map to obtain the labelled volume
    y_test_pred = np.argmax(y_test_prob_map, axis=-1).astype(dtype='uint8')

    # Compose the output filename and headers
    i_subj_id = os.path.basename(test_data['i_t1_path'])[:-7]
    i_fullfilename_out = opj(out_dir, i_subj_id + '_t1.nii.gz')
    i_fullfilename_gt = opj(out_dir, i_subj_id + '_gt.nii.gz')
    i_fullfilename_pred = opj(out_dir, i_subj_id + '_predicted_volume.nii.gz')
    i_fullfilename_prob = opj(out_dir, i_subj_id + '_prob_map_volume.nii.gz')

    # Save out files: gt, predicted, and probability map
    t1_np = test_data['i_t1'][0, :, :, :, 0]
    t1_np -= np.min(t1_np)                      # return to range [0, 255]
    t1_np /= np.max(t1_np)
    t1_np = np.round(t1_np * 255).astype(np.uint8)
    t1_vol = nib.Nifti1Image(t1_np, affine=test_data['affine'], header=test_data['header'])
    nib.save(t1_vol, i_fullfilename_out)

    y_gt = (np.argmax(test_data['i_gt'][0, ...], axis=-1)).astype(np.uint8)
    gt_vol = nib.Nifti1Image(y_gt, affine=test_data['affine'], header=test_data['header'])
    nib.save(gt_vol, i_fullfilename_gt)

    pred_vol = nib.Nifti1Image(y_test_pred, affine=test_data['affine'], header=test_data['header'])
    nib.save(pred_vol, i_fullfilename_pred)

    prob_map_vol = nib.Nifti1Image(y_test_prob_map, affine=test_data['affine'], header=test_data['header'])
    if save_prob_map:
        nib.save(prob_map_vol, i_fullfilename_prob)

    return 


def test_model(model, ds_dataset, dataset, config: Config, path_out_folder, wand=False, save_out_volumes=False):
    """
    Function to test numerically the model.

    :param model: trained model
    :param ds_dataset: tf.data.Dataset
    :param dataset: dataset from 'dataset_manager.prepareDataset'
    :param config: Config object 
    :param path_out_folder: out folder where to store predicted volumes
    :param wand: log in wand or not  
    :param save_out_volumes: save or not the predicted volumes (and T1w + GT)
    """

    # Single run on the entire 'ds_test' 
    identifier = config.exp_name
    ds_test = ds_dataset.take(len(dataset['X_valid_paths']))
    score = model.evaluate(ds_test, verbose=False)

    _ = [logger.info(identifier + ' (' + i_metric + '): ' + str(i_score)) for i_metric, i_score in
         zip(model.metrics_names, score)]

    # Create out folder for testing (if needed)
    if save_out_volumes:
        out_dir = opj(path_out_folder, 'testing/')
        try:
            os.mkdir(out_dir)
        except Exception as e:
            logger.error(e)

    # Run testing on the entire 'dataset['X_valid_paths']'
    df_results = pd.DataFrame(columns=['name', 'metric', 'value', 'intraInter', 'manual'])
    for i_iter in zip(dataset['X_valid_paths'], dataset['Y_valid_paths'], dataset['notes_valid']):
        i_x_path, i_y_path, i_site = i_iter

        # Load X (T1w) and Y (GT) volumes
        i_X = nib.load(i_x_path)
        i_X_vol = np.array(i_X.get_fdata()).astype(dtype='float32')   
        i_Y = nib.load(i_y_path)
        i_Y_vol = np.array(i_Y.get_fdata()).astype(dtype='uint8')

        # Normalisation
        i_X_vol = volume_manager.volume_normalisation(i_X_vol, config.data.normalisation)

        # Volume handling
        i_X_vol = np.expand_dims(i_X_vol, axis=[0,-1])
        i_Y_vol = volume_manager.to_categorical_tensor(i_Y_vol, config.network.num_classes).astype(dtype='uint8')  # 'int32')
        i_Y_vol = np.expand_dims(i_Y_vol, axis=0)

        # Evaluate and parse results
        i_score = model.evaluate(x=i_X_vol, y=i_Y_vol, verbose=False)
        i_manual = 'semi-manual' if 'MindBoggle101' in i_x_path else 'manual' if 'MRBrainS' in i_x_path else 'auto'
        for j_metric, j_score in zip(model.metrics_names, i_score):
            if j_metric == 'loss' : continue
            j_row = [os.path.basename(i_x_path),j_metric, j_score, i_site, i_manual]
            df_results = df_results.append(pd.DataFrame([j_row], 
                                            columns=['name', 'metric', 'value', 'intraInter', 'manual']))
        
        # Predict and save the volume
        if save_out_volumes:
            test_data = {'i_t1': i_X_vol, 'i_gt': i_Y_vol, 'i_t1_path': i_x_path, 'i_gt_path': i_y_path,
                        'header': i_X.header, 'affine': i_X.affine}
            predict_and_save_out_volume(model, test_data, out_dir, save_prob_map=False)

    # Save csv for future analyses
    df_results.to_csv(opj(config.training.exp_path.as_posix(), 'results_valid.csv'), index=False)

    # Create one plot for every condition
    def plot_metric_hue(csv, metric=None, hue='all metrics', save=True):
        """ Function that manages the plots.
        """
        df = csv[csv['metric'] == metric] if metric else csv

        ax = sns.boxplot(x=hue, y='value', data=df, showfliers=False)
        ax = sns.swarmplot(x=hue, y='value', data=df, color='.35', alpha=0.5, s=3)
        title = 'validation results - ' + hue
        plt.title(title)
        plt.ylabel(metric)
        plt.xlabel('')
        plt.show()
        if save:
            plt.savefig(opj(config.training.exp_path.as_posix(), title.replace(' ','_') + '.pdf'))
        if wand:
            wandb.log({title: wandb.Image(plt)})
        plt.close()

    plot_metric_hue(df_results, metric='compute_per_channel_dice', hue='intraInter')
    plot_metric_hue(df_results, metric='compute_per_channel_dice', hue='manual')
    plot_metric_hue(df_results, hue='metric')

    
    # Log every metric (entire validation set)
    metrics = {}
    for i_metric, i_score in zip(model.metrics_names, score):
        metrics[f"test_{i_metric}"] = i_score
    
    # Log 'inter' volumes (i.e., vols from unseen sites) 
    for i_metric in model.metrics_names:
        if j_metric == 'loss' : continue

        logger.info(f"Logging metric: {i_metric}")
        logger.info('Differences in testing between sites seen in training (intra) and not seen (inter):')
        i_score_intra = df_results[(df_results['metric'] == i_metric) & (df_results['intraInter'] == 'intra')]
        i_score_inter = df_results[(df_results['metric'] == i_metric) & (df_results['intraInter'] == 'inter')]
        metrics[f"test_{i_metric}_intra"] = np.mean(i_score_intra['value'])
        metrics[f"test_{i_metric}_inter"] = np.mean(i_score_inter['value'])
        logger.info('Testing on -intra- set (' + i_metric + '): ' + str(np.mean(i_score_intra['value'])))
        logger.info('Testing on -inter- set (' + i_metric + '): ' + str(np.mean(i_score_inter['value'])))

        logger.info('Differences in testing between manual, semi-manual, and auto:')
        i_score_manual = df_results[(df_results['metric'] == i_metric) & (df_results['manual'] == 'manual')]
        i_score_semi_man = df_results[(df_results['metric'] == i_metric) & (df_results['manual'] == 'semi-manual')]
        i_score_auto = df_results[(df_results['metric'] == i_metric) & (df_results['manual'] == 'auto')]
        metrics[f"test_{i_metric}_manual"] = np.mean(i_score_manual['value'])
        metrics[f"test_{i_metric}_semi_manual"] = np.mean(i_score_semi_man['value'])
        metrics[f"test_{i_metric}_auto"] = np.mean(i_score_auto['value'])
        logger.info('Testing on -manual- set (' + i_metric + '): ' + str(np.mean(i_score_manual['value'])))
        logger.info('Testing on -semi_manual- set (' + i_metric + '): ' + str(np.mean(i_score_semi_man['value'])))
        logger.info('Testing on -auto- set (' + i_metric + '): ' + str(np.mean(i_score_auto['value'])))
        logger.info('')

    if wand:
        wandb.log(metrics)
    return

