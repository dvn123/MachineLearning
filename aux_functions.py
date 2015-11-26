import imghdr
import math
import struct
from sklearn.preprocessing import LabelBinarizer
import sklearn.utils.validation as val
import numpy as np


def get_image_size(f_name):
    with open(f_name, 'rb') as f_handle:
        head = f_handle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(f_name) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        else:
            return
        return width, height


def get_percentage_of_list(list, percentage):
    size = math.floor(len(list) * percentage)
    return list[:size]

def _weighted_sum(sample_score, sample_weight, normalize=False):
    if normalize:
        return np.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()

def log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None):
    lb = LabelBinarizer()
    T = lb.fit_transform(y_true)
    if T.shape[1] == 1:
        T = np.append(1 - T, T, axis=1)

    # Clipping
    Y = np.clip(y_pred, eps, 1 - eps)

    # This happens in cases when elements in y_pred have type "str".
    if not isinstance(Y, np.ndarray):
        raise ValueError("y_pred should be an array of floats.")

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    # Check if dimensions are consistent.
    val.check_consistent_length(T, Y)
    T = val.check_array(T)
    Y = val.check_array(Y)
    print(T)
    print(Y)
    if T.shape[1] != Y.shape[1]:
        raise ValueError("y_true and y_pred have different number of classes "
                         "%d, %d" % (T.shape[1], Y.shape[1]))

    # Renormalize
    Y /= Y.sum(axis=1)[:, np.newaxis]
    loss = -(T * np.log(Y)).sum(axis=1)

    return _weighted_sum(loss, sample_weight, normalize)
