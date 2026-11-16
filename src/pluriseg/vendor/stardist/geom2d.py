"""
https://github.com/stardist/stardist/blob/e80c6de700693bc228ed3c9ba1dc19c3785667ee/stardist/geometry/geom2d.py
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

from skimage.draw import polygon # pylint: disable=E0611


def _raise(e):
    """stand-in for csbdeep's _raise"""
    raise e


def _check_label_array(y, name):
    """stand-in for csbdeep's _check_label_array"""
    if not (y.dtype.kind in 'iu' and y.min() >= 0):
        raise ValueError(f"'{name}' must be a non-negative integer label array")


def _py_star_dist(a, n_rays=32, grid=(1,1)):
    (np.isscalar(n_rays) and 0 < int(n_rays)) or _raise(ValueError())
    if grid != (1,1):
        raise NotImplementedError(grid)
    
    n_rays = int(n_rays)
    a = a.astype(np.uint16,copy=False)
    dst = np.empty(a.shape+(n_rays,),np.float32)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            value = a[i,j]
            if value == 0:
                dst[i,j] = 0
            else:
                st_rays = np.float32((2*np.pi) / n_rays)
                for k in range(n_rays):
                    phi = np.float32(k*st_rays)
                    dy = np.cos(phi)
                    dx = np.sin(phi)
                    x, y = np.float32(0), np.float32(0)
                    while True:
                        x += dx
                        y += dy
                        ii = int(round(i+x))
                        jj = int(round(j+y))
                        if (ii < 0 or ii >= a.shape[0] or
                            jj < 0 or jj >= a.shape[1] or
                            value != a[ii,jj]):
                            # small correction as we overshoot the boundary
                            t_corr = 1-.5/max(np.abs(dx),np.abs(dy))
                            x -= t_corr*dx
                            y -= t_corr*dy
                            dist = np.sqrt(x**2+y**2)
                            dst[i,j,k] = dist
                            break
    return dst


def dist_to_coord(dist, points, scale_dist=(1,1)):
    """convert from polar to cartesian coordinates for a list of distances and center points
    dist.shape   = (n_polys, n_rays)
    points.shape = (n_polys, 2)
    len(scale_dist) = 2
    return coord.shape = (n_polys,2,n_rays)
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    assert dist.ndim==2 and points.ndim==2 and len(dist)==len(points) \
        and points.shape[1]==2 and len(scale_dist)==2
    n_rays = dist.shape[1]
    phis = ray_angles(n_rays)
    coord = (dist[:,np.newaxis]*np.array([np.sin(phis),np.cos(phis)])).astype(np.float32)
    coord *= np.asarray(scale_dist).reshape(1,2,1)    
    coord += points[...,np.newaxis] 
    return coord


def polygons_to_label_coord(coord, shape, labels=None):
    """renders polygons to image of given shape

    coord.shape   = (n_polys, n_rays)
    """
    coord = np.asarray(coord)
    if labels is None: labels = np.arange(len(coord))

    _check_label_array(labels, "labels")
    assert coord.ndim==3 and coord.shape[1]==2 and len(coord)==len(labels)

    lbl = np.zeros(shape,np.int32)

    for i,c in zip(labels,coord):
        rr,cc = polygon(*c, shape)
        lbl[rr,cc] = i+1

    return lbl


def polygons_to_label(dist, points, shape, prob=None, thr=-np.inf, scale_dist=(1,1)):
    """converts distances and center points to label image

    dist.shape   = (n_polys, n_rays)
    points.shape = (n_polys, 2)

    label ids will be consecutive and adhere to the order given
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    prob = np.inf*np.ones(len(points)) if prob is None else np.asarray(prob)

    assert dist.ndim==2 and points.ndim==2 and len(dist)==len(points)
    assert len(points)==len(prob) and points.shape[1]==2 and prob.ndim==1

    n_rays = dist.shape[1]

    ind = prob>thr
    points = points[ind]
    dist = dist[ind]
    prob = prob[ind]

    ind = np.argsort(prob, kind='stable')
    points = points[ind]
    dist = dist[ind]

    coord = dist_to_coord(dist, points, scale_dist=scale_dist)

    return polygons_to_label_coord(coord, shape=shape, labels=ind)


def ray_angles(n_rays=32):
    return np.linspace(0,2*np.pi,n_rays,endpoint=False)
