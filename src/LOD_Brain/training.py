#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday - October 10 2022, 15:36:49

@authors:
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia

Function for training the model.
"""

from LOD_Brain.model import losses
from LOD_Brain.model.network import build_model, build_flat_model, build_vanilla_unet_model, build_cerebrum_model
from LOD_Brain.model.utils import freeze_level_lte, load_last_weights, set_trainable, load_best_weights
from LOD_Brain.config import Config
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
import tensorflow as tf
from wandb.keras import WandbCallback
import wandb
from loguru import logger
from LOD_Brain.python_utils import format_prediction_into_gif


def train(config: Config, ds_train, ds_val, class_weights, wand=True):
    """
    Performs a training following the configurations
    :param config: config object
    :param ds_train: tf.data
    :param ds_val: tf.data
    :param class_weights: class weights to feed the fit function
    :return trained tf model
    """

    # Create checkpoint folder
    (config.training.exp_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    if config.training.coarse_to_fine and config.network.num_levels > 1:
        logger.info(f"Training coarse to fine with {config.network.num_levels} step")
        final_size = config.network.size
        num_levels = config.network.num_levels
        for i in range(num_levels):
            logger.info(f"--- STEP: {i + 1}\n\n")
            config.network.num_levels = i+1
            logger.info(f"Setting size to {config.network.size} ({final_size} // 2 ** ({config.network.num_levels} - {i} -1)")
            tf.keras.backend.clear_session()    # clear session to avoid wrong names and memory issues
            model = build_model(config.network, ds_factor=config.network.downsampling_factor**(num_levels - i - 1 + config.network.pre_ds))
            logger.info(f"Model # parameters: {model.count_params()}")
            opt = model._get_optimizer(config.training.optimizer)
            opt.lr.assign(config.training.lr)
            model.compile(optimizer=opt,
                          loss=losses.losses_dict[config.training.loss],
                          metrics=list(losses.metrics_dict.values()))
            if i:
                # load prev weights if i > 0
                load_last_weights(model, config.training.exp_path / "checkpoints")
                # freeze level > i
                freeze_level_lte(model, i)
            # train newly added layer
            logger.info(f"--- [training new layers] ---")
            train_model(config, model, ds_train, ds_val, class_weights)
            if i and config.training.fine_tuning:
                # unfreeze
                set_trainable(model)
                # train whole network. Apply specific finetuning hyperparams
                ft_config = config.copy()
                ft_config.training.lr /= 10
                ft_config.training.epochs = ft_config.training.epochs_ft

                logger.info(f"--- [finetuning] ---")
                opt = model._get_optimizer(config.training.optimizer)
                opt.lr.assign(ft_config.training.lr)
                model.compile(optimizer=opt,
                              loss=losses.losses_dict[config.training.ft_loss],
                              metrics=list(losses.metrics_dict.values()))
                train_model(ft_config, model, ds_train, ds_val, class_weights)

            # save weights
            model.save_weights((config.training.exp_path / "checkpoints" / f'weights_lv{i}.h5').as_posix())

        # final refinement
        if config.training.refinement:

            # unfreeze
            if config.training.ref_unfreeze:
                set_trainable(model)

            ref_config = config.copy()
            ref_config.training.lr = config.training.ref_lr
            ref_config.training.epochs = config.training.ref_epochs

            logger.info(f"--- [refinement] ---")
            opt = model._get_optimizer(ref_config.training.optimizer)
            opt.lr.assign(ref_config.training.lr)
            model.compile(optimizer=opt,
                          loss=losses.losses_dict[ref_config.training.ref_loss],
                          metrics=list(losses.metrics_dict.values()))
            train_model(ref_config, model, ds_train, ds_val, class_weights)

            # save weights
            model.save_weights((config.training.exp_path / "checkpoints" / f'refined_weights_lv{i}.h5').as_posix())

        if config.training.test_best_model:
            load_best_weights(model, config.training.exp_path / "checkpoints")

        return model
    else:
        model = build_model(config.network, config.network.downsampling_factor ** config.network.pre_ds)
        logger.info(f"Model # parameters: {model.count_params()}")
        opt = model._get_optimizer(config.training.optimizer)
        opt.lr.assign(config.training.lr)
        # tf.keras.backend.set_value(model.optimizer.lr, config['initial_learning_rate']) # TODO: consider this
        model.compile(optimizer=opt,
                      loss=losses.losses_dict[config.training.loss],
                      metrics=list(losses.metrics_dict.values()))
        if config.training.weights_path:
            logger.info(f"Load weights from {config.training.weights_path}")
            model.load_weights(config.training.weights_path, by_name=True)

        train_model(config, model, ds_train, ds_val, class_weights)
        # save weights
        model.save_weights((config.training.exp_path / "checkpoints" / f'final_weights.h5').as_posix())

        if config.training.test_best_model:
            load_best_weights(model, config.training.exp_path / "checkpoints")

        return model


def train_flat(config: Config, ds_train, ds_val, class_weights):
    """
    Performs a training following the configurations
    :param config: config object
    :param ds_train: tf.data
    :param ds_val: tf.data
    :param class_weights: class weights to feed the fit function
    :return trained tf model
    """

    # Create checkpoint folder
    (config.training.exp_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    if config.training.coarse_to_fine and config.network.num_levels > 1:
        logger.info(f"Training coarse to fine with {config.network.num_levels} step")
        final_size = config.network.size
        num_levels = config.network.num_levels
        filters = config.network.num_filters
        num_blocks_per_level = config.network.num_blocks_per_level - num_levels
        for i in range(num_levels):
            logger.info(f"--- STEP: {i + 1}\n\n")
            logger.info(f"Setting size to {config.network.size} ({final_size} // 2 ** ({config.network.num_levels} - {i} -1)")
            tf.keras.backend.clear_session()    # clear session to avoid wrong names and memory issues

            config_tr = config.copy()
            config_tr.network.num_blocks_per_level = num_blocks_per_level + i
            config_tr.network.num_initial_filter = filters[num_levels - i - 1]
            logger.info(f"num_blocks_per_level: {config_tr.network.num_blocks_per_level}, "
                        f"num_initial_filter: {config_tr.network.num_initial_filter}, "
                        f"num_filters: {config_tr.network.num_filters}")

            model = build_flat_model(config_tr.network, ds_factor=config_tr.network.downsampling_factor**(num_levels - i - 1 + config_tr.network.pre_ds))
            opt = model._get_optimizer(config.training.optimizer)
            opt.lr.assign(config.training.lr)
            model.compile(optimizer=opt,
                          loss=losses.losses_dict[config.training.loss],
                          metrics=list(losses.metrics_dict.values()))
            if i:
                # load prev weights if i > 0
                model.summary(print_fn=lambda x: logger.info(x), line_length=120)  # log the summary of the model
                load_last_weights(model, config.training.exp_path / "checkpoints")
                # freeze level > i
                freeze_level_lte(model, i)
            # train newly added layer
            logger.info(f"--- [training new layers] ---")
            train_model(config_tr, model, ds_train, ds_val, class_weights)
            if i and config.training.fine_tuning:
                # unfreeze
                set_trainable(model)
                # train whole network. Apply specific finetuning hyperparams
                ft_config = config_tr.copy()
                ft_config.training.lr /= 10
                ft_config.training.epochs = ft_config.training.epochs_ft

                logger.info(f"--- [finetuning] ---")
                opt = model._get_optimizer(config.training.optimizer)
                opt.lr.assign(ft_config.training.lr)
                model.compile(optimizer=opt,
                              loss=losses.losses_dict[config.training.ft_loss],
                              metrics=list(losses.metrics_dict.values()))
                train_model(ft_config, model, ds_train, ds_val, class_weights)

            # save weights
            model.save_weights((config.training.exp_path / "checkpoints" / f'weights_lv{i}.h5').as_posix())

        if config.training.test_best_model:
            load_best_weights(model, config.training.exp_path / "checkpoints")

        return model


def train_vanilla_unet(config: Config, ds_train, ds_val, class_weights):
    """
    Performs a training following the configurations
    :param config: config object
    :param ds_train: tf.data
    :param ds_val: tf.data
    :param class_weights: class weights to feed the fit function
    :return trained tf model
    """

    # Create checkpoint folder
    (config.training.exp_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session()    # clear session to avoid wrong names and memory issues

    if config.network.cerebrum_net:
        logger.info(f"-- Training CEREBRUM U-Net in 1 step (no finetuning, no coarse-to-fine) --")
        logger.info(f"num_blocks_per_level: {config.network.num_blocks_per_level}, "
                    f"num_initial_filter: {config.network.num_initial_filter}"
                    f"num_filters: {config.network.num_filters}")
        model = build_cerebrum_model(config.network)
        adamopt = tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=adamopt,
                        loss=losses.losses_dict['categorical_crossentropy'],
                        metrics=list(losses.metrics_dict.values()))        
    else:
        logger.info(f"-- Training Vanilla U-Net in 1 step (no finetuning, no coarse-to-fine) --")
        logger.info(f"num_blocks_per_level: {config.network.num_blocks_per_level}, "
                    f"num_initial_filter: {config.network.num_initial_filter}"
                    f"num_filters: {config.network.num_filters}")
        model = build_vanilla_unet_model(config.network)
        opt = model._get_optimizer(config.training.optimizer)
        opt.lr.assign(config.training.lr)
        model.compile(optimizer=opt,
                        loss=losses.losses_dict[config.training.loss],
                        metrics=list(losses.metrics_dict.values()))
    train_model(config, model, ds_train, ds_val, class_weights)

    # save weights
    model.save_weights((config.training.exp_path / "checkpoints" / f'weights_unet.h5').as_posix())

    if config.training.test_best_model:
        load_best_weights(model, config.training.exp_path / "checkpoints")

    return model


def train_model(config: Config, model, ds_train, ds_val, class_weights):
    """
    Train a single pre-instantiated model
    :param config: config object
    :param model: tf model
    :param ds_train: tf.data
    :param ds_val: tf.data
    :param class_weights: class weights to feed the fit function
    :return:
    """
    model.summary(print_fn=lambda x: logger.info(x), line_length=120)           # log the summary of the model
    STEPS_PER_EPOCH = config.training.train_size // config.training.batch_size
    # Callbacks
    wandb_callback = WandbCallback(monitor='val_loss',
                                   verbose=0,
                                   mode='auto',
                                   save_weights_only=True,
                                   log_weights=True,
                                   log_gradients=False,
                                   save_model=False,
                                   log_batch_frequency=int(config.training.validation_freq * STEPS_PER_EPOCH)
                                  )
    # TODO: Add a callback to log the result volume of a segmentation test (on wandb via a LambdaCallback)
    tb_callback = TensorBoard(log_dir=(config.training.exp_path.parent / 'logs').as_posix(),
                              histogram_freq=0,
                              # profile_batch='10, 15'
                              )
    checkpointer = ModelCheckpoint((config.training.exp_path / "checkpoints" / "model.{epoch:02d}-{val_compute_per_channel_dice:.4f}.h5").as_posix(),
                                   monitor='val_compute_per_channel_dice',
                                   save_best_only=config.training.save_train_best_model_only,            #True,
                                   mode='max',
                                   save_freq='epoch',    # int(config.training.validation_freq * STEPS_PER_EPOCH),
                                   verbose=1
                                   )

    def log_wandb_mask(epoch, logs):
        if not epoch % config.training.validation_freq: 
            # Use the model to predict the values from the validation dataset.
            vol = next(ds_val.as_numpy_iterator())
            segm = model.predict(vol[0])

            out = format_prediction_into_gif(vol[0], segm)
            gt = format_prediction_into_gif(vol[0], vol[1])
        
            # log all composite images to W&B
            wandb.log({"predictions": wandb.Video(out, fps=4, format="gif")})
            wandb.log({"gt": wandb.Video(gt, fps=4, format="gif")})

    im_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_wandb_mask)

    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, min_lr=1e-7, verbose=1)

    # Training loop
    model.fit(ds_train,
              steps_per_epoch=config.training.train_size // config.training.batch_size,
              epochs=config.training.epochs,
              validation_data=ds_val,
              validation_steps=config.training.validation_steps,
              validation_freq=config.training.validation_freq,
              callbacks=[tb_callback, wandb_callback, reduce_lr, earlystopper, checkpointer, im_callback],
              # class_weight = class_weights,
              verbose=2,
              )
