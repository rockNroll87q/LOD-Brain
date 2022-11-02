#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday - February 17 2021, 16:40:14

@authors:
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia

Functions to create and deal Configuration files.
Please note that all the values are overwritten by the 'config-defaults.yaml' file.
"""

from typing import List
from pydantic import BaseModel, validator, Field, PositiveInt, PositiveFloat
import tensorflow as tf
from LOD_Brain.model.losses import losses_dict


class NetConfig(BaseModel):
    num_classes: PositiveInt = Field(7, title="number of classes")
    num_levels: PositiveInt = Field(2, title="number of hierarchical levels to use on resolution dimension")
    size: PositiveInt = Field(256, title="maximum size to reach")
    num_initial_filter: PositiveInt = Field(8, title="number of filters in the first block")
    num_blocks_per_level: PositiveInt = Field(3, title="number of blocks per level (CEREBRUM had 3)")
    upskip: str = Field("UpSampling3D", title="upsampling strategy")
    conv_block: str = Field("Plain", title="conv block (either Plain or BottleNeck)")
    dropout_rate: PositiveFloat = Field(0.05, title="used dropout rate. A 0 value means no dropout")
    activation_enc: str = Field("relu", title="a registered tf2 activation function for the encoder")
    activation_dec: str = Field("relu", title="a registered tf2 activation function for the decoder")
    stride: PositiveInt = Field(2, title="applied downsampling and upsampling stride")
    bn: str = Field("GN", title="whether use batch norm (BN) or GroupNorm (GN) or None")
    kernel_initializer: str = Field("he_normal", title="kernel initializer")
    kernel_regularizer: PositiveFloat = Field(1e-4, title="l2 penalty")
    downsampling_factor: PositiveInt = Field(4, title="downsampling factor (must be power of 2)")
    skip_intermediate_result: bool = Field(False, title="Avoid the corse-to-fine final summation. Keep only the information from the last level.")
    pre_ds: int = Field(0, title="apply a pre downsampling")
    flat_net: bool = Field(False, title="build a flat architecture (still coarse to fine")
    conv_repetition: bool = Field(False, title="repeated conv layers (where possible)")
    filter_multiplicative_factors: List = Field([], title="filter_multiplicative_factors. Should be formatted like '[1,1,1]'")
    boost_first_layer: bool = Field(False, title="use 'filter_multiplicative_factors' also for the first conv layer")
    vanilla_unet: bool = Field(False, title="build a vanilla U-Net architecture")
    cerebrum_net: bool = Field(False, title="build a CEREBRUM-3T architecture")

    @property
    def num_filters(cls):
        return [cls.num_initial_filter * (2 ** i) for i in range(cls.num_blocks_per_level)]


class TrainConfig(BaseModel):
    lr: PositiveFloat = Field(0.0005, title="learning rate")
    optimizer: str = Field('Adam', title="optimizer, can be either a accepted keras optimizer string, or an optimizer object")
    loss: str = Field('per_channel_dice_loss', title="loss. See `LOD_Brain.model.losses.losses_dict` for a list on available losses")
    ft_loss: str = Field('categorical_crossentropy', title="loss used during finetuning")
    batch_size: PositiveInt = Field(1, title="batch size")
    epochs: PositiveInt = Field(50, title="number of epochs for the base model")
    epochs_ft: PositiveInt = Field(20, title="number of epochs for the fine-tunings. `coarse_to_fine` and `fine_tuning` must be enabled")
    train_size: PositiveInt = Field(None, title="Number of training samples")
    validation_steps: PositiveInt = Field(10, title="Number of steps in validation")
    validation_freq: PositiveInt = Field(1, title="Specifies how many training epochs to run before a new validation run is performed")
    weights_path: str = Field('', title="Load pretrained model. `coarse_to_fine` must be disable, and you must ensure that the model is correct")
    coarse_to_fine: bool = Field(True, title="Whether enable a coase-to-fine training")
    fine_tuning: bool = Field(False, title="Whether enable the finetuning in training (means the coarse will be unfreezed and fine-tuned)")
    exp_path: str = Field('', title="Private: experiment path")
    refinement: bool = Field(False, title="Enable a final refinement")
    ref_loss: str = Field('categorical_crossentropy', title="Loss used during refinement")
    ref_lr: PositiveFloat = Field(1e-5, title="refinement lr")
    ref_epochs: PositiveInt = Field(10, title="refinement epochs")
    ref_unfreeze: bool = Field(False, title="unfreeze or not coarse levels while refinement")
    test_best_model: bool = Field(True, title="test the best model (True) or the last (False)")
    save_train_best_model_only: bool = Field(True, title="in training, store only the best model (True) or the last (False)")

    @property
    def optimizer_obj(cls):
        if cls.optimizer == 'Adam':
            return tf.keras.optimizers.Adam(lr=cls.lr, beta_1=0.9, beta_2=0.999)
        else:
            raise ValueError('Unknown optimizer.')


class AugmentConfig(BaseModel):
    augmentation: bool = Field(True, title="use or not data augmentation")
    prob_overall: float = Field(0.9, title="overall probability to apply data augmentation (training ONLY)")

    prob_flip: float = Field(0.5, title="probability to apply 'VerticalFlip'")
    prob_inho: float = Field(1.0, title="probability to apply 'InhomogeneityNoiseAugment'")

    prob_geom: float = Field(1.0, title="probability to apply a geometric transformation (one of the following)")
    prob_grid: float = Field(1.0, title="probability to apply 'GridDistortion'")
    prob_resi: float = Field(0.0, title="probability to apply 'RandomResizedCrop'")
    prob_rota: float = Field(0.0, title="probability to apply 'RotationAugment'")
    prob_tran: float = Field(0.0, title="probability to apply 'TranslationAugment'")

    prob_colo: float = Field(1.0, title="probability to apply a color transformation (one of the following)")
    prob_blur: float = Field(1.0, title="probability to apply 'Blur'")
    prob_down: float = Field(1.0, title="probability to apply 'Downscale'")
    prob_salt: float = Field(1.0, title="probability to apply 'SaltAndPepperNoiseAugment'")
    prob_gaus: float = Field(1.0, title="probability to apply 'GaussianNoiseAugment'")
    prob_ghos: float = Field(1.0, title="probability to apply 'GhostingAugment'")
    prob_gamm: float = Field(1.0, title="probability to apply 'GammaNoiseAugment'")
    prob_neck: float = Field(0.0, title="probability to apply 'SliceRepetitionNeckNoiseAugment'")
    prob_cont: float = Field(1.0, title="probability to apply 'ContrastNoiseAugment'")
    prob_slic: float = Field(1.0, title="probability to apply 'SliceSpacingNoiseAugment'")
    prob_bias: float = Field(0.0, title="probability to apply 'BiasNoiseAugment'")

    z_score_volume: bool = Field(False, title="z-score T1w volume after augmentation")
    augmentation_AF: bool = Field(False, title="apply any augmentation available")


class DataConfig(BaseModel):
    Path_in_data: str = Field('/LOD_Brain/data/', title="root folder")
    Path_in_csv: str = Field('/LOD_Brain/data/analysis/csv/', title="csv path")
    Filename_csv: str = Field('dataset_short_training_a+a.csv', title="csv filename")
    Inh_vol_path: str = Field('/LOD_Brain/data/analysis/inhomogeneity_volume/inhomogeneity.npy', title="inhomogeneity path")
    normalisation: str = Field('z_score_volume', title="Normalisation procedure, can be: [z_score_volume, z_score_site, rescaling, rescaling_a_b]")
    
    @validator('normalisation')
    def norm_check(cls, v):
        if not v in ['z_score_volume', 'z_score_site', 'rescaling', 'rescaling_a_b']:
            raise ValueError('normalisation must be in [z_score_volume, z_score_site, rescaling, rescaling_a_b]')
        return v


class Config(BaseModel):
    exp_name: str = Field("LOD_Brain", title="Experiment base name")
    seed: PositiveInt = Field(23, title="random seed")
    network: NetConfig = Field(NetConfig(), title="Network configurations")
    data: DataConfig = Field(DataConfig(), title="Data configurations")
    training: TrainConfig = Field(TrainConfig(), title="Training configurations")
    augment: AugmentConfig = Field(AugmentConfig(), title="Augmentation configurations")


def get_help(schema, mother=''):
    """
    Create the help structure given a pydantic scheme
    :param schema: pydantic schema
    :param mother: the option group
    :return:
    """
    for c in schema["properties"]:
        if "allOf" in schema["properties"][c]:
            print(f'\n    [{c}]')
            sec_key = schema["properties"][c]["allOf"][0]["$ref"].split('/')[-1]
            get_help(schema["definitions"][sec_key], mother=f'{c}.')
        else:
            title = schema["properties"][c]["title"]
            arg_type = schema["properties"][c]["type"]
            default = schema["properties"][c].get("default", "---")
            print(f'      {title}\n      --{mother if mother else ""}{c} [{arg_type}] = {default}')


def show_help():
    """
    Shows an help message
    """
    print("""Usage: python main.py [OPTIONS]

        Script to train LOD_Brain. The help is in the form --argument [type] = default

    Options:""")
    get_help(Config.schema())


def config_flattened(schema, mother=''):
    """
    Return a list of permitted config parameters
    :param schema: pydantic schema
    :param mother: the option group (used for recursion)
    :return:
    """
    out = []
    for c in schema["properties"]:
        if "allOf" in schema["properties"][c]:
            sec_key = schema["properties"][c]["allOf"][0]["$ref"].split('/')[-1]
            out += list(config_flattened(schema["definitions"][sec_key], mother=f'{c}.'))
        else:
            out += [f'{mother if mother else ""}{c}']
    return out


def validate_config_arg(arg):
    """
    Check if an argument is in the configuration scheme
    :param arg: argument
    :return: True if arg is present, False otherwise
    """
    config_list = config_flattened(Config.schema())
    return arg in config_list
