#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday - February 17 2021, 16:40:14

@authors:
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia

Functions to augment training data.
"""

import numpy as np
from scipy.ndimage import affine_transform
from scipy.ndimage import rotate
import albumentations as albu
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from LOD_Brain.config import Config
from scipy import stats


def addWeighted(src1, alpha, src2, beta, gamma):
    """ 
    Calculates the weighted sum of two arrays (cv2 replaced).

    :param src1: first input array.
    :param aplha: weight of the first array elements.
    :param src2: second input array of the same size and channel number as src1.
    :param beta: weight of the second array elements.
    :param gamma: scalar added to each sum
    :return: output array that has the same size and number of channels as the input arrays.
    """

    return src1 * alpha + src2 * beta + gamma


def augmentation_salt_and_pepper_noise(X_data, amount=10. / 1000):
    """ 
    Function to add S&P noise to the volume.

    :param X_data: input volume (3D) -> shape (x,y,z)
    :param amount: quantity of voxels affected
    :return X_data_out: augmented volume
    """

    X_data_out = X_data
    salt_vs_pepper = 0.2  # Ration between salt and pepper voxels
    n_salt_voxels = int(np.ceil(amount * np.prod(X_data_out.size) * salt_vs_pepper))
    n_pepper_voxels = int(np.ceil(amount * np.prod(X_data_out.size) * (1.0 - salt_vs_pepper)))

    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(n_salt_voxels)) for i in np.squeeze(X_data_out).shape]
    X_data_out[coords[0], coords[1], coords[2]] = np.max(X_data)

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(n_pepper_voxels)) for i in np.squeeze(X_data_out).shape]
    X_data_out[coords[0], coords[1], coords[2]] = np.min(X_data)

    return X_data_out


class SaltAndPepperNoiseAugment(ImageOnlyTransform):
    def apply(self, img, **params):
        return augmentation_salt_and_pepper_noise(img)


def augmentation_gaussian_noise(X_data):
    """ 
    Function to add gaussian noise to the volume.

    :param X_data: input volume (3D) -> shape (x,y,z)
    :return X_data_out: augmented volume
    """

    # Gaussian distribution parameters
    X_data_no_background = X_data
    mean = np.mean(X_data_no_background)
    var = np.var(X_data_no_background)
    sigma = var ** 0.5

    gaussian = np.random.normal(mean, sigma, X_data.shape).astype(X_data.dtype)

    # Compose the output (src1, alpha, src2, beta, gamma)
    X_data_out = addWeighted(X_data, 0.8, gaussian, 0.2, 0)

    return X_data_out


class GaussianNoiseAugment(ImageOnlyTransform):
    def apply(self, img, **params):
        return augmentation_gaussian_noise(img)


def augmentation_inhomogeneity_noise(X_data, inhom_vol):
    """ 
    Function to add inhomogeneity noise to the volume.

    :param X_data: input volume (3D) -> shape (x,y,z)
    :param inhom_vol: inhomogeneity volume (preloaded)
    :return X_data_out: augmented volume
    """

    # Randomly select a vol of the same shape of 'X_data'
    x_1 = np.random.randint(0, int(X_data.shape[0]) - 1, size=1)[0]
    x_2 = np.random.randint(0, int(X_data.shape[1]) - 1, size=1)[0]
    x_3 = np.random.randint(0, int(X_data.shape[2]) - 1, size=1)[0]
    y_1 = inhom_vol[x_1: x_1 + X_data.shape[0],
          x_2: x_2 + X_data.shape[1],
          x_3: x_3 + X_data.shape[2]]

    # Compose the output: add noise to the original vol
    X_data_out = X_data + y_1.astype(X_data.dtype)

    return X_data_out


class InhomogeneityNoiseAugment(ImageOnlyTransform):

    def __init__(self, inhom_vol: np.array, always_apply=False, p=1.0):
        super(InhomogeneityNoiseAugment, self).__init__(always_apply, p)
        self.inhom_vol = inhom_vol

    def apply(self, img, **params):
        return augmentation_inhomogeneity_noise(img, self.inhom_vol)


def change_luminance_contrast(X_data: np.ndarray, clipping: bool=False, threshold: float=0.025):
    """ 
    Function to change the luminance and the contrast of the input volume.
    :param X_data: input volume (3D) -> shape (x,y,z)
    :param clipping: boolean, clip the the 
    :param threshold: clip at this value (0.025 is like 6 / 255)
    :return X_data_out: augmented volume
    """

    X_data_out = X_data
    gamma = (3.0 - 0.5) * np.random.RandomState().random_sample() + 0.5

    def gamma_correction_1_slide(slice, gamma):

        slice_min, slice_max = slice.min(), slice.max()
        slice_gamma = slice
        slice_gamma = ((slice_gamma - slice_min) / (slice_max - slice_min + 0.001))                 # move in the range [0, 1]
        if clipping:
            slice_gamma[slice < threshold] = 0
        slice_gamma = slice_gamma **  (1.0 / gamma)                                                 # apply gamma
        slice_gamma = slice_gamma * (slice_max - slice_min) + slice_min                             # move back in the original range

        return slice_gamma

    # Apply gamma correction for every slice in 'X_data'
    for i_slide in range(X_data.shape[2]):
        X_data_out[:,:,i_slide] = gamma_correction_1_slide(X_data[:,:,i_slide], gamma)

    return X_data_out


class GammaNoiseAugment(ImageOnlyTransform):
    def apply(self, img, **params):
        return change_luminance_contrast(img)


def augmentation_neck_slice_repetition(X_data: np.ndarray, threshold: float=0.025):
    """ 
    Function to add slices at the end of the volume (i.e., neck).

    :param X_data: input volume (3D) -> shape (x,y,z)
    :param threshold: clip at this value (0.025 is like 12 / 255)
    :return X_data_out: augmented volume
    """

    X_min, X_max = X_data.min(), X_data.max()
    X_data_out = ((X_data - X_min) / (X_max - X_min + 0.001))               # move in the range [0, 1]
    slice_repetitions = np.random.RandomState().randint(10, 30)

    # Find the last slice with no zero values (or almost zero)
    for i in reversed(range(X_data.shape[2])):
        if np.max(X_data_out[:, i, :]) > threshold:
            break

    if i == 0 or i == (X_data.shape[2] - 1):
        return X_data

    # Take 'i' (-10) slices and copy if 'slice_repetitions' times
    index_to_copy = i - 10
    slice_to_copy = X_data_out[:, index_to_copy, :]
    for j in range(index_to_copy, min(index_to_copy + slice_repetitions, X_data_out.shape[1] - 1)):
        X_data_out[:, j, :] = slice_to_copy

    X_data_out = X_data_out * (X_max - X_min) + X_min                       # move back in the original range

    return X_data_out


class SliceRepetitionNeckNoiseAugment(ImageOnlyTransform):
    def apply(self, img, **params):
        return augmentation_neck_slice_repetition(img)


def change_contrast(X_data: np.ndarray, min_alpha: float=0.5, max_alpha: float=3.0):
    """ 
    Function to change the contrast of the input volume.

    :param X_data: input volume (3D) -> shape (x,y,z)
    :param min_alpha: min value for the contrast
    :param max_alpha: max value for the contrast
    :return X_data_out: augmented volume
    """

    X_min, X_max = X_data.min(), X_data.max()
    alpha = (max_alpha - min_alpha) * np.random.RandomState().random_sample() + min_alpha       # Contrast
    beta = 0                                                                                    # Brightness

    # Apply contrast change to 'X_data'
    X_data_out = np.clip((alpha * X_data + beta), X_min, X_max)

    return X_data_out


class ContrastNoiseAugment(ImageOnlyTransform):
    def apply(self, img, **params):
        return change_contrast(img)


def slice_spacing(X_data: np.ndarray, min_slice_rep: int=2, max_slice_rep: int=5):
    """ 
    Function to add more slices of the input volume in Axial view.
    If 'slice_repetitions'=2, means every slice is repeated twice (for a total of 2, consecutive).
    

    :param X_data: input volume (3D) -> shape (x,y,z)
    :param min_slice_rep: min amount of consecutive slices in Axial view
    :param max_slice_rep: max amount of consecutive slices in Axial view
    :return X_data_out: augmented volume
    """

    slice_repetitions = np.random.RandomState().randint(min_slice_rep, max_slice_rep)

    # Apply contrast change to 'X_data'
    X_data_out = X_data[:, ::(slice_repetitions), :]                    # keep only '256/(slice_repetitions)' slices
    X_data_out = np.repeat(X_data_out, slice_repetitions, axis=1)       # repeat the slice 'slice_repetitions' times
    X_data_out = X_data_out[:, :X_data.shape[2], :]                     # take the same shape as the beginning

    assert X_data_out.shape == X_data.shape

    return X_data_out


class SliceSpacingNoiseAugment(ImageOnlyTransform):
    def apply(self, img, **params):
        return slice_spacing(img)


def augmentation_bias_noise(X_data: np.ndarray):
    """ 
    Function to add bias noise to the volume.

    :param X_data: input volume (3D) -> shape (x,y,z)
    :return X_data_out: augmented volume
    """
    
    # Extract a value for 'n_cycles' from 1 to 7, and a possible transpose choice
    n_cycles = np.random.RandomState().randint(1, 7)
    possible_transpose = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
    i_choice = np.random.RandomState().choice(range(len(possible_transpose)))
    Factor = 2.

    # Create sin wave
    x = np.linspace(-np.pi * n_cycles/Factor, np.pi * n_cycles/Factor, 256)
    y = np.linspace(-np.pi * n_cycles/Factor, np.pi * n_cycles/Factor, 256)
    z = np.linspace(-np.pi * n_cycles/Factor, np.pi * n_cycles/Factor, 256)
    xx, yy, zz = np.meshgrid(x, y, z)
    noise = np.sin(np.transpose(xx, axes=(0,2,1)) + yy + zz) + \
            np.sin(xx + np.transpose(yy, axes=(0,2,1)) + zz) + \
            np.sin(xx + yy + np.transpose(zz, axes=(0,2,1)))

    # Change (transpose) the axes to add variability
    noise = np.transpose(noise, axes=possible_transpose[i_choice])          

    # Adjust the amplitude of the noise (1/10 of the volume)
    noise = stats.zscore(noise, axis=None)
    noise = noise / 2

    # Add noise to avoid easy filter detection
    mean = np.mean(noise)
    var = np.var(noise)
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, noise.shape).astype(noise.dtype)
    noise = addWeighted(noise, 0.9, gaussian, 0.1, 0)        # Compose the output (src1, alpha, src2, beta, gamma)

    # Apply it only on non-zero, or near to zero, of 'X_data' (i.e., min after z-scoring)
    noise[X_data == X_data.min()] = 0
    
    # Compose the output: add noise to the original vol
    X_data_out = X_data + noise.astype(X_data.dtype)

    return X_data_out


class BiasNoiseAugment(ImageOnlyTransform):
    def apply(self, img, **params):
        return augmentation_bias_noise(img)


def translate_volume(image,
                     shift_x0: int, shift_x1: int, shift_x2: int,
                     padding_mode: str = 'nearest',
                     spline_interp_order: int = 1):
    """ 
    Function to apply volume translation to a single volume.

    :param image: input volume (3D) -> shape (x,y,z)
    :param shift_x0-shift_x1-shift_x2: shift in voxels
    :param padding_mode: the padding mode
    :param spline_interp_order: order for the affine transformation
    :return: augmented volume
    """

    # Set the affine transformation matrix
    M_t = np.eye(4)
    M_t[:-1, -1] = np.array([-shift_x0, -shift_x1, -shift_x2])

    return affine_transform(image, M_t,
                            order=spline_interp_order,
                            mode=padding_mode,
                            cval=0,
                            output_shape=image.shape)


class TranslationAugment(DualTransform):
    """ Class to deal with translation augmentation. """

    def __init__(self, max_shift: list = [20, 20, 20], always_apply=False, p=1.0):
        super(TranslationAugment, self).__init__(always_apply, p)
        self.max_shift = max_shift

    def get_params(self):

        # Randomly select parameters
        try:
            shifts = [(np.random.RandomState().randint(2 * i) - i) for i in self.max_shift]
            shift_x0, shift_x1, shift_x2 = shifts
        except:
            shift_x0, shift_x1, shift_x2 = [0] * 3

        return {"shift_x0": shift_x0, "shift_x1": shift_x1, "shift_x2": shift_x2}

    def apply(self, img, shift_x0: int = 0, shift_x1: int = 0, shift_x2: int = 0, **params):

        # Apply to image or mask
        if np.issubdtype(img.dtype, np.floating):  # image
            img_out = translate_volume(img,
                                       shift_x0, shift_x1, shift_x2,
                                       padding_mode='nearest',
                                       spline_interp_order=1)
        elif np.issubdtype(img.dtype, np.integer):  # mask
            img_out = translate_volume(img,
                                       shift_x0, shift_x1, shift_x2,
                                       padding_mode='constant',
                                       spline_interp_order=0)
        else:
            raise Exception('Error 23: type not supported.')

        return img_out


class RotationAugment(DualTransform):
    """ Class to deal with rotation augmentation. """

    def __init__(self,
                 max_angle: int = 10,
                 rot_spline_order: int = 3,
                 always_apply=False,
                 p=1.0):
        super(RotationAugment, self).__init__(always_apply, p)
        self.max_angle = max_angle
        self.rot_spline_order = rot_spline_order

    def get_params(self):

        # Randomly select parameters
        random_angle = np.random.RandomState().randint(2 * self.max_angle) - self.max_angle
        rot_axes = np.random.RandomState().permutation(range(3))[:2]  # random select the 2 rotation axes

        return {"random_angle": random_angle, "rot_axes": rot_axes}

    def apply(self, img, random_angle: int, rot_axes: int, **params):

        # Apply to image or mask
        if np.issubdtype(img.dtype, np.floating):  # image
            img_out = rotate(input=img,
                             angle=random_angle,
                             axes=rot_axes,
                             reshape=False,
                             order=self.rot_spline_order,
                             mode='nearest',
                             prefilter=True)
        elif np.issubdtype(img.dtype, np.integer):  # mask
            img_out = rotate(input=img,
                             angle=random_angle,
                             axes=rot_axes,
                             reshape=False,
                             order=0,
                             mode='constant',
                             prefilter=True)
        else:
            raise Exception('Error 24: type not supported.')

        return img_out


class GhostingAugment(ImageOnlyTransform):
    """ Class to deal with ghosting augmentation. """

    def __init__(self,
                 max_repetitions: int = 4,
                 always_apply=False,
                 p=1.0):
        super(GhostingAugment, self).__init__(always_apply, p)
        self.max_repetitions = max_repetitions

    def apply(self, img, **params):
        # Randomly select parameters
        repetitions = np.random.RandomState().choice(range(1, self.max_repetitions + 1))
        axis = np.random.RandomState().choice(range(len(img.shape)))

        img_out = img
        shift_value = 0
        for i_rep in range(1, repetitions + 1):
            # Compute the shift to apply to the data
            shift_value += int(img.shape[axis] / (i_rep + 1))

            # Shift the data and add to the out volume
            data_repetition = np.roll(img, shift_value, axis=axis)
            img_out = addWeighted(img_out, 0.85, data_repetition, 0.15, 0)

        return img_out


def get_augm_transforms(inho_vol, config: Config, volume_size: int = 256):
    """
    Get the transformations for volume (and mask) augmentation.

    :param inho_vol: inhomogeneity volume
    :param volume_size: size of the volume
    :param config: Config class (with all probabilities stored)
    :return: albumentation composition
    """

    return albu.Compose([

        # Default transformations
        albu.VerticalFlip(p = config.augment.prob_flip),  # sagittal plane
        InhomogeneityNoiseAugment(inho_vol, p = config.augment.prob_inho),  # Inhomogeneity noise

        # Geometric transformations
        albu.OneOf([
            albu.GridDistortion(num_steps = 5,
                                distort_limit = (-0.10, +0.10),
                                interpolation = 4,
                                border_mode = 1,
                                p = config.augment.prob_grid),
            albu.RandomResizedCrop(height = volume_size,
                                   width = volume_size,
                                   scale = (0.9, 1.0),
                                   ratio = (0.8, 1.20),
                                   interpolation = 4,
                                   p = config.augment.prob_resi),
            RotationAugment(p = config.augment.prob_rota),
            TranslationAugment(p = config.augment.prob_tran),
        ], p = config.augment.prob_geom),

        # Color transformations
        albu.OneOf([
            albu.Blur(blur_limit = (3, 3), p = config.augment.prob_blur),
            SaltAndPepperNoiseAugment(p = config.augment.prob_salt),
            GaussianNoiseAugment(p = config.augment.prob_gaus),
            GhostingAugment(p = config.augment.prob_ghos),
            albu.OneOf([                                # half 'interpolation = 0', half 'interpolation = 4'
                albu.Downscale(scale_min = 0.25, 
                                scale_max = 0.75, 
                                interpolation = 0,      # (cv2.INTER_NEAREST)
                                p = 1.),
                albu.Downscale(scale_min = 0.25, 
                                scale_max = 0.75, 
                                interpolation = 4,      # (cv2.INTER_LANCZOS4)
                                p = 1.),
            ], p=config.augment.prob_down),
            GammaNoiseAugment(p = config.augment.prob_gamm),
            SliceRepetitionNeckNoiseAugment(p = config.augment.prob_neck),
            ContrastNoiseAugment(p = config.augment.prob_cont),
            SliceSpacingNoiseAugment(p = config.augment.prob_slic),
            BiasNoiseAugment(p = config.augment.prob_bias),    
        ], p = config.augment.prob_colo),

    ], p = config.augment.prob_overall)
    

def get_augm_transforms_AF(inho_vol, config: Config, volume_size: int = 256):
    """
    Get the transformations for volume (and mask) augmentation.

    :param inho_vol: inhomogeneity volume
    :param volume_size: size of the volume
    :param config: Config class (with all probabilities stored)
    :return: albumentation composition
    """

    return albu.Compose([

        # Default transformations
        albu.VerticalFlip(p = config.augment.prob_flip),  # sagittal plane
        InhomogeneityNoiseAugment(inho_vol, p = config.augment.prob_inho),  # Inhomogeneity noise

        # Geometric transformations
        albu.GridDistortion(num_steps = 5,
                            distort_limit = (-0.10, +0.10),
                            interpolation = 4,
                            border_mode = 1,
                            p = config.augment.prob_grid),
        albu.RandomResizedCrop(height = volume_size,
                                width = volume_size,
                                scale = (0.9, 1.0),
                                ratio = (0.8, 1.20),
                                interpolation = 4,
                                p = config.augment.prob_resi),
        RotationAugment(p = config.augment.prob_rota),
        TranslationAugment(p = config.augment.prob_tran),

        # Color transformations
        albu.OneOf([
            albu.Blur(blur_limit = (3, 3), p = config.augment.prob_blur),
            GaussianNoiseAugment(p = config.augment.prob_gaus),
        ], p = 1.),
        albu.Downscale(scale_min = 0.6, 
                        scale_max = 0.99, 
                        interpolation = 4, 
                        p = config.augment.prob_down),
        # SaltAndPepperNoiseAugment(p = config.augment.prob_salt),
        albu.OneOf([
            GhostingAugment(p = config.augment.prob_ghos),
        ], p = 0.1),        # not so always.

    ], p = config.augment.prob_overall)
    
