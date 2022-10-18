#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday - October 10 2022, 15:36:49

@authors:
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia

Main.
"""

import tensorflow as tf
from LOD_Brain.training import train, train_flat, train_vanilla_unet
from LOD_Brain.testing import test_model
from LOD_Brain.config import Config, show_help, validate_config_arg
from LOD_Brain import python_utils
from LOD_Brain.data import dataset_manager
from loguru import logger
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml
import os
from os.path import join as opj
import sys
import wandb


_args_system = {}


def _process_system_args():
    global _args_system
    for num, arg in enumerate(sys.argv):
        if 'help' in arg:
            show_help()
            sys.exit()
        if arg.startswith(("-", "--")):
            try:
                current_arg = arg.split('=')
                _get_type = current_arg[0].replace('-', '')
                if not validate_config_arg(_get_type):
                    logger.warning(f'The argument "{_get_type}" is not present in the configuration. Is there a typo?')
                    sys.exit()

                get_type = _get_type.split('.')

                if get_type[0] not in _args_system:
                    _args_system[get_type[0]] = {}
                if len(get_type) == 1: 
                    _args_system[get_type[0]] = current_arg[0]
                if len(get_type) == 2:
                    if len(get_type) == 2:
                        if '[' in current_arg[1]:  # convert a list-like string in a python list.
                            _args_system[get_type[0]][get_type[1]] = eval(current_arg[1])
                        else:
                            _args_system[get_type[0]][get_type[1]] = current_arg[1]
            except ValueError:
                pass


def main():
    # Initialize a new wandb run
    wandb.init(entity="LOD_Brain", project="multisite_phase3")         # multisite_phase2    ablation_study    synthseg_trainings
    
    # Config is a variable that holds and saves hyperparameters and inputs
    logger.info(f"Updating confing from argparse: {_args_system}")
    config = Config(**_args_system)
    wandb.run.config.update(config.dict(), allow_val_change=True)
    
    wandb.run.name = f'{datetime.now().strftime("%m%d.%H%M%S")}_lv{config.network.num_levels}_' \
                     f'{"c2f_" if config.training.coarse_to_fine else ""}id{wandb.run.id}'
    wandb.run.save()
    config.exp_name = wandb.run.name if wandb.run.name else wandb.run.id
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)

    config.training.exp_path = Path(wandb.run.dir)
    with open(config.training.exp_path.parent / 'config.yaml', 'w') as f:
        yaml.dump(config.dict(), f, sort_keys=False)
    # logger file handler
    logger.add(opj(config.training.exp_path.as_posix(), config.exp_name + '.log'))
    logger.info(f"Config: {config}")

    # Find a GPU available and set others not visible
    logger.info("\n\n\n******** Run started ********")
    logger.info('%10s : %s' % ('Running on', os.uname()[1]))
    logger.info("Command line: " + sys.executable + " " + " ".join(sys.argv))
    python_utils.findGPUtoUse()

    # Retrieve all the volume pathways from the csv and other attributes
    logger.info("\n\n\n******** Dataset preparation ********")
    dataset = dataset_manager.prepareDataset(config)
    config.training.train_size = len(dataset['X_train_paths'])
    
    # Create TF datasets for train and valid sets
    ds_train, ds_valid = dataset_manager.TFDatasetGenerator(config, dataset)

    # Training
    logger.info("\n\n\n******** Training ********")

    if config.network.flat_net:
        logger.info("Training a flat unet")
        model = train_flat(config, ds_train, ds_valid, dataset['class_weights'])
    elif config.network.vanilla_unet or config.network.cerebrum_net:
        logger.info("Training a vanilla or CEREBRUM unet")
        model = train_vanilla_unet(config, ds_train, ds_valid, dataset['class_weights'])
    else:
        logger.info("Training a pyramid-unet")
        model = train(config, ds_train, ds_valid, dataset['class_weights'])

    # Log trained model info
    out_dir = config.training.exp_path.as_posix()
    tf.keras.utils.plot_model(model, to_file=opj(out_dir, 'model.png'), show_shapes=True)
    logger.info(f"Model number of parameters: {model.count_params()}")
    wandb.log({"model_count_params": model.count_params()})

    # Testing
    logger.info(f"\n\n\n******** Testing ********")
    test_model(model, ds_valid, dataset, config=config, \
            path_out_folder=out_dir, \
            wand=True, save_out_volumes=True)


if __name__ == '__main__':
    _process_system_args()
    main()
