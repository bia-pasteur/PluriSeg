"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu

https://github.com/MouseLand/cellpose/blob/a54cb48849b7e225a81e8e43dcb042d42427f543/cellpose/utils.py#L621
"""

import fastremap
import fill_voids
import numpy as np
from scipy.ndimage import find_objects


def fill_holes_and_remove_small_masks(masks, min_size=15):
    """ Fills holes in masks (2D/3D) and discards masks smaller than min_size.

    This function fills holes in each mask using fill_voids.fill.
    It also removes masks that are smaller than the specified min_size.

    Parameters:
    masks (ndarray): Int, 2D or 3D array of labelled masks.
        0 represents no mask, while positive integers represent mask labels.
        The size can be [Ly x Lx] or [Lz x Ly x Lx].
    min_size (int, optional): Minimum number of pixels per mask.
        Masks smaller than min_size will be removed.
        Set to -1 to turn off this functionality. Default is 15.

    Returns:
        ndarray: Int, 2D or 3D array of masks with holes filled and small masks removed.
            0 represents no mask, while positive integers represent mask labels.
            The size is [Ly x Lx] or [Lz x Ly x Lx].
    """

    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError("masks_to_outlines takes 2D or 3D array, not %dD array" %
                         masks.ndim)

    # Filter small masks
    if min_size > 0:
        uniq, counts = fastremap.unique(masks, return_counts=True)
        # uniq[0] is background (0), so uniq[1:] are the actual mask labels
        # counts[1:] are the corresponding counts
        small_mask_indices = np.nonzero(counts[1:] < min_size)[0]
        # Get the actual label values to remove (not indices)
        labels_to_remove = uniq[1:][small_mask_indices]
        masks = fastremap.mask(masks, labels_to_remove)
        fastremap.renumber(masks, in_place=True)

    slices = find_objects(masks)
    j = 0
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            msk = fill_voids.fill(msk)
            masks[slc][msk] = (j + 1)
            j += 1

    if min_size > 0:
        uniq, counts = fastremap.unique(masks, return_counts=True)
        small_mask_indices = np.nonzero(counts[1:] < min_size)[0]
        labels_to_remove = uniq[1:][small_mask_indices]
        masks = fastremap.mask(masks, labels_to_remove)
        fastremap.renumber(masks, in_place=True)
    

    return masks