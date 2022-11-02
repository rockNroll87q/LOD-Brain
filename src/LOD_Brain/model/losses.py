"""
Created on 12 Feb 2021

@author: Mattia Savardi, Michele Svanera
"""
import tensorflow as tf
import tensorflow.keras.backend as K


def tversky_metric(y_true, y_pred, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
    """
    Tversky metric

    The Tversky index, named after Amos Tversky, is an asymmetric similarity measure on sets that compares a variant
    to a prototype. The Tversky index can be seen as a generalization of the Sørensen–Dice coefficient and the Tanimoto
    coefficient (aka Jaccard index).

    alpha = beta = 0.5 => dice coeff
    alpha = beta = 1 => tanimoto coeff
    alpha + beta = 1 => F beta coeff

    :param y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse categorical crossentropy where shape = `[batch_size, d0, .. dN-1]`
    :param y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
    :param eps: numeric stability
    :param beta: weight for the false negatives
    :param alpha: weight for the false positives
    :return: float, tversky coefficient
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)

    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)

    return (true_pos + eps) / (true_pos + alpha * false_pos + beta * false_neg + eps)


def tversky_loss(alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
    def loss(y_true, y_pred):
        return 1 - tversky_metric(y_true, y_pred, alpha, beta, eps)

    return loss


def focal_tversky_loss(gamma=0.75):
    def loss(y_true, y_pred):
        pt_1 = tversky_metric(y_true, y_pred)
        return K.pow((1 - pt_1), gamma)

    return loss


def dice_coef_multilabel_metric(y_true, y_pred, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-6,
                                axis=(0, 1, 2, 3)):
    """
    Dice coefficient multi label
    :param y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse categorical crossentropy where shape = `[batch_size, d0, .. dN-1]`
    :param y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
    :param alpha: 0.5 for dice coefficient
    :param beta: 0.5 for dice coefficient
    :param eps: numeric stability
    :param axis: reduction axes
    :return: 0 in the best case and 1 in the worst -> sum of dc on the different labels
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    num_classes = K.cast(K.shape(y_true)[-1], float)
    ones = K.ones(K.shape(
        y_true))    # TODO: this operation fails: TypeError: An op outside of the function building code is being passed
                    #       a "Graph" tensor. It is possible to have Graph tensors leak out of the function building
                    #       context by including a tf.init_scope in your function building code.

    p1 = ones - y_pred
    g1 = ones - y_true

    num = K.sum(y_pred * y_true, axis=axis) + eps
    den = num + alpha * K.sum(y_pred * g1, axis=axis) + beta * K.sum(p1 * y_true, axis=axis)

    ratio = K.sum(num / den)

    return ratio / num_classes


def dice_coef_multilabel_loss(alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-6, use_log=True):
    def loss(y_true, y_pred):
        if use_log:
            return -tf.math.log(dice_coef_multilabel_metric(y_true, y_pred, alpha, beta, eps))
        else:
            return 1 - dice_coef_multilabel_metric(y_true, y_pred, alpha, beta, eps)

    return loss


def jaccard_metric(y_true, y_pred, axis=(1, 2, 3, 4), eps: float = 1e-6):
    """
    Calculate Jaccard similarity between labels and predictions.

    Jaccard similarity is in [0, 1], where 1 is perfect overlap and 0 is no
    overlap. If both labels and predictions are empty (e.g., all background),
    then Jaccard similarity is 1.

    If we assume the inputs are rank 5 [`(batch, x, y, z, classes)`], then an
    axis parameter of `(1, 2, 3)` will result in a tensor that contains a Jaccard
    score for every class in every item in the batch. The shape of this tensor
    will be `(batch, classes)`. If the inputs only have one class (e.g., binary
    segmentation), then an axis parameter of `(1, 2, 3, 4)` should be used.
    This will result in a tensor of shape `(batch,)`, where every value is the
    Jaccard similarity for that prediction.

    Implemented according to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/#Equ7

    :param y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse categorical crossentropy where shape = `[batch_size, d0, .. dN-1]`
    :param y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
    :param axis: reduction axes
    :param eps: numerical stability
    :return:Tensor of Jaccard similarities.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    intersection = tf.reduce_sum(y_true * y_pred, axis=axis) + eps
    union = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis)
    return intersection / (union - intersection)


def jaccard_loss(y_true, y_pred):
    return 1. - jaccard_metric(y_true, y_pred)


def compute_per_channel_dice(y_true, y_pred, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         y_true (torch.Tensor): NxSpatialxC input tensor
         y_pred (torch.Tensor): NxSpatialxC target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """
    y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
    y_true = tf.cast(y_true, "float64")
    y_pred = tf.cast(y_pred, "float64")

    # compute per channel Dice Coefficient
    intersect = tf.math.reduce_sum(y_pred * y_true, 0)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target) or extension (see V-Net) (input^2 + target^2)
    denominator = tf.math.reduce_sum(y_pred * y_pred, 0) + tf.math.reduce_sum(
        y_true * y_true, 0
    )
    return 2 * (intersect / (denominator + epsilon))


def per_channel_dice_loss(x, y, **kwargs):
    return 1 - tf.reduce_mean(compute_per_channel_dice(x, y, **kwargs))


def mixed_loss(cce=.4, dice=0.6):
    def loss(x, y, **kwargs):
        return tf.cast(dice*per_channel_dice_loss(x, y, **kwargs), "float64") + tf.cast(cce*tf.keras.losses.categorical_crossentropy(x, y, **kwargs), "float64")
    return loss


losses_dict = {'categorical_crossentropy': tf.keras.losses.categorical_crossentropy,
               'focal_tversky_loss': focal_tversky_loss(gamma=0.75),
               'tversky_custom_loss': tversky_loss(alpha=0.3, beta=0.7),
               'dice_loss': tversky_loss(alpha=0.5, beta=0.5),
               'tanimoto_loss': tversky_loss(alpha=1, beta=1),
               'jaccard_loss': jaccard_loss,
               'per_channel_dice_loss': per_channel_dice_loss,
               'mixed_dice_cce_8_2': mixed_loss(dice=0.8, cce=0.2),
               'mixed_dice_cce_6_4': mixed_loss(dice=0.6, cce=0.4),
               'mixed_dice_cce_5_5': mixed_loss(dice=0.5, cce=0.5),
               'mixed_dice_cce_4_6': mixed_loss(dice=0.4, cce=0.6),
               'mixed_dice_cce_2_8': mixed_loss(dice=0.2, cce=0.8),
               }

metrics_dict = { #'dice_coef_multilabel_metric': dice_coef_multilabel_metric, # Remove until bugfix
                'tversky_custom_metric': tversky_metric,
                'jaccard_metric': jaccard_metric,
                'compute_per_channel_dice': compute_per_channel_dice,
                }
