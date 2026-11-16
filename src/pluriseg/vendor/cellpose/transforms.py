"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

https://github.com/MouseLand/cellpose/blob/a54cb48849b7e225a81e8e43dcb042d42427f543/cellpose/transforms.py
"""
import logging

import numpy as np

transforms_logger = logging.getLogger(__name__)


def normalize99(Y, lower=1, upper=99, copy=True, downsample=False):
    """
    Normalize the image so that 0.0 corresponds to the 1st percentile and 1.0 corresponds to the 99th percentile.

    Args:
        Y (ndarray): The input image (for downsample, use [Ly x Lx] or [Lz x Ly x Lx]).
        lower (int, optional): The lower percentile. Defaults to 1.
        upper (int, optional): The upper percentile. Defaults to 99.
        copy (bool, optional): Whether to create a copy of the input image. Defaults to True.
        downsample (bool, optional): Whether to downsample image to compute percentiles. Defaults to False.

    Returns:
        ndarray: The normalized image.
    """
    X = Y.copy() if copy else Y
    X = X.astype("float32") if X.dtype!="float64" and X.dtype!="float32" else X
    if downsample and X.size > 224**3:
        nskip = [max(1, X.shape[i] // 224) for i in range(X.ndim)]
        nskip[0] = max(1, X.shape[0] // 50) if X.ndim == 3 else nskip[0]
        slc = tuple([slice(0, X.shape[i], nskip[i]) for i in range(X.ndim)])
        x01 = np.percentile(X[slc], lower)
        x99 = np.percentile(X[slc], upper)
    else:
        x01 = np.percentile(X, lower)
        x99 = np.percentile(X, upper)
    if x99 - x01 > 1e-3:
        X -= x01 
        X /= (x99 - x01)
    else:
        X[:] = 0
    return X