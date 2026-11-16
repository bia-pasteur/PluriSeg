"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

https://github.com/MouseLand/cellpose/blob/a54cb48849b7e225a81e8e43dcb042d42427f543/cellpose/plot.py
"""
from pluriseg.vendor.cellpose import transforms
import numpy as np


# modified to use sinebow color
def dx_to_circ(dP):
    """Converts the optic flow representation to a circular color representation.

    Args:
        dP (ndarray): Flow field components [dy, dx].
        
    Returns:
        ndarray: The circular color representation of the optic flow.

    """
    mag = 255 * np.clip(transforms.normalize99(np.sqrt(np.sum(dP**2, axis=0)), downsample=True), 0, 1.)
    angles = np.arctan2(dP[1], dP[0]) + np.pi
    a = 2
    mag /= a
    rgb = np.zeros((*dP.shape[1:], 3), "uint8")
    rgb[..., 0] = np.clip(mag * (np.cos(angles) + 1), 0, 255).astype("uint8")
    rgb[..., 1] = np.clip(mag * (np.cos(angles + 2 * np.pi / 3) + 1), 0, 255).astype("uint8")
    rgb[..., 2] = np.clip(mag * (np.cos(angles + 4 * np.pi / 3) + 1), 0, 255).astype("uint8")
    
    return rgb

