"""
https://github.com/stardist/stardist/blob/e80c6de700693bc228ed3c9ba1dc19c3785667ee/stardist/utils.py
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects

try:
    from edt import edt
    _edt_available = True
    try:    _edt_parallel_max = len(os.sched_getaffinity(0))
    except: _edt_parallel_max = 128
    _edt_parallel_default = 4
    _edt_parallel = os.environ.get('STARDIST_EDT_NUM_THREADS', _edt_parallel_default)
    try:
        _edt_parallel = min(_edt_parallel_max, int(_edt_parallel))
    except ValueError as e:
        warnings.warn(f"Invalid value ({_edt_parallel}) for STARDIST_EDT_NUM_THREADS. Using default value ({_edt_parallel_default}) instead.")
        _edt_parallel = _edt_parallel_default
    del _edt_parallel_default, _edt_parallel_max
except ImportError:
    _edt_available = False
    # warnings.warn("Could not find package edt... \nConsider installing it with \n  pip install edt\nto improve training data generation performance.")
    pass


def edt_prob(lbl_img, anisotropy=None):
    if _edt_available:
        return _edt_prob_edt(lbl_img, anisotropy=anisotropy)
    else:
        # warnings.warn("Could not find package edt... \nConsider installing it with \n  pip install edt\nto improve training data generation performance.")
        return _edt_prob_scipy(lbl_img, anisotropy=anisotropy)

def _edt_prob_edt(lbl_img, anisotropy=None):
    """Perform EDT on each labeled object and normalize.
    Internally uses https://github.com/seung-lab/euclidean-distance-transform-3d
    that can handle multiple labels at once
    """
    lbl_img = np.ascontiguousarray(lbl_img)
    constant_img = lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0
    if constant_img:
        warnings.warn("EDT of constant label image is ill-defined. (Assuming background around it.)")
    # we just need to compute the edt once but then normalize it for each object
    prob = edt(lbl_img, anisotropy=anisotropy, black_border=constant_img, parallel=_edt_parallel)
    objects = find_objects(lbl_img)
    for i,sl in enumerate(objects,1):
        # i: object label id, sl: slices of object in lbl_img
        if sl is None: continue
        _mask = lbl_img[sl]==i
        # normalize it
        prob[sl][_mask] /= np.max(prob[sl][_mask]+1e-10)
    return prob

def _edt_prob_scipy(lbl_img, anisotropy=None):
    """Perform EDT on each labeled object and normalize."""
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    constant_img = lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0
    if constant_img:
        lbl_img = np.pad(lbl_img, ((1,1),)*lbl_img.ndim, mode='constant')
        warnings.warn("EDT of constant label image is ill-defined. (Assuming background around it.)")
    objects = find_objects(lbl_img)
    prob = np.zeros(lbl_img.shape,np.float32)
    for i,sl in enumerate(objects,1):
        # i: object label id, sl: slices of object in lbl_img
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        # 1. grow object slice by 1 for all interior object bounding boxes
        # 2. perform (correct) EDT for object with label id i
        # 3. extract EDT for object of original slice and normalize
        # 4. store edt for object only for pixels of given label id i
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask = grown_mask[shrink_slice]
        edt = distance_transform_edt(grown_mask, sampling=anisotropy)[shrink_slice][mask]
        prob[sl][mask] = edt/(np.max(edt)+1e-10)
    if constant_img:
        prob = prob[(slice(1,-1),)*lbl_img.ndim].copy()
    return prob
