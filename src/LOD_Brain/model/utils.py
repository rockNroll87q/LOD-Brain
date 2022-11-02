"""
Created on Feb 12 2021
@author: met

Model utility functions
"""
import os
from loguru import logger


def recompile(model):
    """
    Recompile a given model
    :param model: tf model
    """
    model.compile(model.optimizer, model.loss, model.metrics)


def freeze_level(model, level: int = 0):
    """
    Freeze a given coarse-to-fine level
    :param model: tf model
    :param level: level name
    """
    for layer in model.layers:
        if layer.variables:
            if layer.variables[0].name.split('/')[0] == f'Level_{level}':
                layer.trainable = False
                continue


def freeze_level_lte(model, level_lte: int = 0):
    """
    Freeze coarse-to-fine levels up to a given level
    :param model: tf model
    :param level_lte: level less than or equal
    """
    for k in range(level_lte+1):
        freeze_level(model, k)


def freeze_model(model):
    """
    Free all the layers in a given model
    :param model: tf model
    """
    for layer in model.layers:
        layer.trainable = False


def set_trainable(model):
    """
    Set all the layer in a given model to trainable
    :param model: tf model
    """
    for layer in model.layers:
        layer.trainable = True
    # recompile(model) Bug: raise an error if tf==2.4.1. Is it needed?


def load_last_weights(model, path, name='weights*.h5'):
    """
    Load the last updated weight file in a given directory
    :param model: tf model
    :param path: a posix path
    :param name: glob name to look for
    """
    last_weights_file = sorted(path.glob(name), key=os.path.getmtime)[-1]
    model.load_weights(last_weights_file, by_name=True)


def load_best_weights(model, path, name='model*.h5'):
    """
    Load the best weight file in a given directory
    :param model: tf model
    :param path: a posix path
    :param name: glob name to look for
    """
    last_weights_file = sorted(path.glob(name), key=lambda x: x.name.split('.')[-2])[-1]
    model.load_weights(last_weights_file, by_name=True)
    logger.info(f"Weights from checkpoint {last_weights_file} restored.")
