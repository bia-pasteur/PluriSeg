"""
https://github.com/kevinjohncutler/omnipose/blob/15bf463d9af167d1b5323d16276a96c018260ae5/src/omnipose/core.py
"""

import logging
import time

import edt
import fastremap
import ncolor
import numpy as np
import scipy
import torch
from scipy.ndimage import affine_transform, binary_dilation, binary_opening, binary_closing, convolve, find_objects, label, mean, shift, uniform_filter
from skimage import filters, measure
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import find_boundaries  # pylint: disable=E0611
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from torch.cuda import empty_cache

from pluriseg.vendor.omnipose import utils

omnipose_logger = logging.getLogger('omnipose-core')


torch_GPU = None
torch_CPU = None
OMNI_INSTALLED = True
SKIMAGE_ENABLED = True
SKLEARN_ENABLED = True
HDBSCAN_ENABLED = False


# ## Section I: core utilities

def get_niter(dists):
    """
    Get number of iterations. 
    
    Parameters
    --------------
    dists: ND array, float
        array of (nonnegative) distance field values
        
    Returns
    --------------
    niter: int
        number of iterations empirically found to be the lower bound for convergence 
        of the distance field relaxation method
    
    """
    # isarray = type(dists) == np.ndarray
    # module = np if isarray else torch
    module = utils.get_module(dists)
    c = module.ceil(module.max(dists)*1.16)+1
    return c.astype(int) if module==np else c.int()
    
    # deprecated? only called during training flow generation it seems, not inference 

    # m = module.max(dists)
    # c = module.ceil(m*1.16)
    # # c = c.item()
    # i = c.to(torch.int32) + 1
    # return i


# minor modification to generalize to nD 
def dist_to_diam(dt_pos,n): 
    """
    Convert positive distance field values to a mean diameter. 
    
    Parameters
    --------------
    dt_pos: 1D array, float
        array of positive distance field values
    n: int
        dimension of volume. dt_pos is always 1D because only the positive values
        int he distance field are passed in. 
        
    Returns
    --------------
    mean diameter: float
        a single number that corresponds to the diameter of the N-sphere when
        dt_pos for a sphere is given to the function, holds constant for 
        extending rods of uniform width, much better than the diameter of a circle 
        of equivalent area for estimating the short-axis dimensions of objects
    
    """
    return 2*(n+1)*np.mean(dt_pos)
#     return np.exp(3/2)*gmean(dt_pos[dt_pos>=gmean(dt_pos)])

def diameters(masks, dt=None, dist_threshold=0, pill=False, return_length=False):
    
    """
    Calculate the mean cell diameter from a label matrix. 
    
    Parameters
    --------------
    masks: ND array, float
        label matrix 0,...,N
    dt: ND array, float
        distance field
    dist_threshold: float
        cutoff below which all values in dt are set to 0. Must be >=0. 
        
    Returns
    --------------
    diam: float
        a single number that corresponds to the average diameter of labeled regions in the image, see dist_to_diam()
    
    """
    if dist_threshold<0:
        dist_threshold = 0
    
    if dt is None and np.any(masks):
        dt = edt.edt(np.int32(masks))
    dt_pos = np.abs(dt[dt>dist_threshold])
          
    # omnipose_logger.info(dt_pos.shape)
    A = np.count_nonzero(dt_pos)
    D = np.sum(dt_pos)
    
    if np.any(dt_pos):
        if not pill:
            diam = dist_to_diam(np.abs(dt_pos),n=masks.ndim)
            if return_length:
                return diam, A/diam
        else:
            return pill_decomposition(A,D)
    else:
        diam = 0
        
    return diam
    
def pill_decomposition(A,D):
    R = np.sqrt((np.sqrt(A**2 + 24*np.pi*D) - A) / (2*np.pi))
    L = (3*D - np.pi*(R**4)) / (R**3)
    return R, L
    

    
# ## Section II: ground-truth flow computation  


# @torch.no_grad() # try to solve memory leak in mps
def masks_to_flows(masks, affinity_graph=None, dists=None, coords=None, links=None, use_gpu=True, device=None, 
                   omni=True, dim=2, smooth=False, normalize=False, n_iter=None, verbose=False):
    """Convert masks to flows. 
    
    First, we find the scalar field. In Omnipose, this is the distance field. In Cellpose, 
    this is diffusion from center pixel. 
    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask.
    
    The flow components are then found as the gradient of the scalar field. 

    Parameters
    -------------
    masks: int, ND array
        labeled masks, 0 = background, 1,2,...,N = mask labels   
    dists: ND array, float
        array of (nonnegative) distance field values
    affinity_graph: ND array, bool
        hypervoxel affinity array, alternative to providing overseg labels and links
        the most general way to compute flows, and can represent internal boundaries 
    links: list of label links
        list of tuples used for treating label pairs as the same  
    use_gpu: bool
        flag to use GPU for speedup. Note that Omnipose fixes some bugs that caused the Cellpose GPU implementation
        to have different behavior compared to the Cellpose CPU implementation. 
    device: torch device
        what compute hardware to use to run the code (GPU VS CPU)
    omni: bool
        flag to generate Omnipose flows instead of Cellpose flows
    dim: int
        dimensionality of image data

    Returns
    -------------
    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].
    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 

    """
    if links is not None and dists is not None:
        print('Your dists are probably wrong...')
        
    if coords is None:
        coords = np.nonzero(masks) 
    # Generalize method of computing affinity graph for flow
    # as well as boundary, even with self-contact. Self-contact
    # requires mutilabel masks and link files. 
    
    steps, inds, idx, fact, sign = utils.kernel_setup(dim)
    
    case = [affinity_graph is None, 
             affinity_graph is not None and affinity_graph.shape[1] != len(coords[0])]
    if np.any(case):
        affinity_graph = masks_to_affinity(masks, coords, steps, inds, 
                                           idx, fact, sign, dim, links=links)
        if case[1]:
            print('Warning: passed affinity does not match mask coordinates. Recomputing.')

    boundaries = affinity_to_boundary(masks,affinity_graph,coords)
    
    if dists is None:
        # formatting reshuffles indices, so only do this
        # when no links are present 
        if (links is None or len(links)==0):# and (affinity_graph is None):
            masks = ncolor.format_labels(masks)
            dists = edt.edt(masks,parallel=-1)
        else:
            # this distance field is not completely accurate, but the point of it
            # is to estimate the number of iterations needed only, so close enough
            # better this than have self-contact boundaries mess up the distance field
            # and therefore completely overestimate the number of iterations required 
            # (Need to test to see if checking for convergence is faster...)
            dists = edt.edt(masks-boundaries,parallel=-1)+(masks>0)

    if device is None:
        if use_gpu:
            device = torch_GPU
        else:
            device = torch_CPU
    
    # masks_to_flows_device/cpu deprecated. Running using torch on CPU is still 2x faster
    # than the dedicated, jitted CPU code thanks to it being parallelized I think.
    
    if masks.ndim==3 and dim==2:
        # this branch preserves original 3D approach 
        print('Sorry, this branch has not yet been updated - do not use omnipiose for this')
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_torch(masks[z], dists[z], boundaries[z], 
                                        device=device, omni=omni)[0]
            mu[[1,2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_torch(masks[:,y], dists[:,y], boundaries[:,y], 
                                        device=device, omni=omni)[0]
            mu[[0,2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_torch(masks[:,:,x], dists[:,:,x], boundaries[:,:,x], #<<< will want to fix this 
                                        device=device, omni=omni)[0]
            mu[[0,1], :, :, x] += mu0
        return masks, dists, None, mu #consistency with below
    
    else:
        T, mu = masks_to_flows_torch(masks, affinity_graph, coords, dists, device=device,
                                     omni=omni, smooth=smooth, normalize=normalize, n_iter=n_iter, 
                                     verbose=verbose)
        return masks, dists, boundaries, T, mu



#Now fully converted to work for ND.
# @torch.no_grad() # try to solve memory leak in mps
def masks_to_flows_torch(masks, affinity_graph, coords=None, dists=None, device=torch.device('cpu'), omni=True,
                         affinity_field=False, smooth=False, normalize=False, n_iter=None, weight=1,
                         return_flows=True, edges=None, initialize=False, verbose=False):
    """Convert ND masks to flows. 
    
    Omnipose find distance field, Cellpose uses diffusion from center of mass.

    Parameters
    -------------
    masks: int, ND array
        labelled masks, 0 = background, 1,2,...,N = mask labels
    dists: ND array, float
        array of (nonnegative) distance field values
    device: torch device
        what compute hardware to use to run the code (GPU VS CPU)
    omni: bool
        flag to generate Omnipose flows instead of Cellpose flows
    smooth: bool
        use relaxation to smooth out distance and therby flow field
    n_iter: int
        override number of iterations 

    Returns
    -------------
    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z or T = mu[0].
    dist: float, 2D or 3D array
        scalar field representing temperature distribution (Cellpose)
        or the smooth distance field (Omnipose)

    """
    if np.any(masks):
        # the padding here is different than the padding added in masks_to_flows(); 
        # for omni, I used to reflect across the edge like a barbarian to simulate the mask extending past the edge, then crop
        # now I just use the affinity graph and force connections to the boundary!

        centers = np.array([])  
        if not omni: #do original centroid projection algrorithm
            unique_labels = fastremap.unique(masks)[1:]
            # get mask centers
            centers = np.array(scipy.ndimage.center_of_mass(masks, 
                                                            labels=masks, 
                                                            index=unique_labels)).astype(int).T
                        
            # check mask center inside mask
            valid = masks[tuple(centers)] == unique_labels
            for i in np.nonzero(~valid)[0]:
                crds = np.array(np.nonzero(masks==unique_labels[i]))
                meds = np.median(crds,axis=0)
                imin = np.argmin(np.sum((crds-meds)**2,axis=0))
                centers[:,i]=crds[:,imin]
                
        # set number of iterations
        if n_iter is None:
            if omni and OMNI_INSTALLED:
                if dists is not None:
                    # omni version requires fewer iterations 
                    n_iter = get_niter(dists) ##### omnipose.core.get_niter
            else:
                slices = scipy.ndimage.find_objects(masks)
                ext = np.array([[s.stop - s.start + 1 for s in slices[i-1]] for i in unique_labels])
                n_iter = 2 * (ext.sum(axis=1)).max()
            

        out = _extend_centers_torch(masks, centers, affinity_graph, coords,
                                    n_iter=n_iter, device=device, omni=omni, smooth=smooth, 
                                    weight=weight, return_flows=return_flows, affinity_field=affinity_field,
                                    edges=edges, initialize=initialize, verbose=verbose)
        
        if return_flows:
            T, mu = out
            if normalize:
                mu = utils.normalize_field(mu,use_torch=True,cutoff=0 if not smooth else 0.15) ##### transforms.normalize_field(mu,omni) 
                if verbose:
                    print('normalizing field')
            return T, mu
        else:
            return out
    else:
        return torch.zeros(masks.shape), torch.zeros((masks.ndim,)+masks.shape)



# @njit() cannot compute fingerprint of empty set
def masks_to_affinity(masks, coords, steps, inds, idx, fact, sign, dim,
                      neighbors=None,
                      links=None, edges=None, dists=None, cutoff=np.sqrt(2), 
                      spatial=False):
    """
    Convert label matrix to affinity graph. Here the affinity graph is an NxM matrix,
    where N is the number of possible hypercube connections (3**dimension) and M is the
    number of foreground hypervoxels. Self-connections are set to 0. 
    
    idx is the central index of the kernel, inds[0]. 
    edges is a list of tuples (y1,y2,y3,...),(x1,x2,x3,...) etc. to which all adjacent pixels should be connected
    concatenated masks should be paddedby 1 to make sure that doesn't cause unextpected label merging 
    dist can be used instead for edge connectivity 
    """

    # only reason to pad with edgemode  is to leverage duplicating labels to connect to boundary
    # must pad with 1 to allow for simple neighbor indexing 
    # There is much larger prior padding to handle edge artifacts, but we could avoid this with more sophisticated edge handling
    # need two things to ask the question: 1. is_background 2. is_edge 
    # if we are looking at an edge, we ask if we are connected to any background in any direction
    # if so, we do not connect to an edge 
    # that would leave single pixels connected to an edge, so need to check its neighbors for its edge connections
    
    shape = masks.shape
    # dim x steps x npix array of pixel coordinates 
    if neighbors is None: 
        
        neighbors = utils.get_neighbors(coords,steps,dim,shape,edges)
        
    # print('masks_to_affinity',masks.shape,coords[0].shape,neighbors.shape)
    
    # define where edges are, may be in the middle of concatenated images 
    is_edge = np.logical_and.reduce([neighbors[d]==neighbors[d][idx] for d in range(dim)]) 
    
    # extract list of neighbor label values
    piece_masks = masks[tuple(neighbors)]
    
    # see where the neighbor matches central pixel
    is_self = piece_masks == piece_masks[idx]

    # Pixels are linked if they share the same label or are next to an edge...
    conditions = [is_self,
                  is_edge
                 ] 
    # print([c.shape for c in conditions],len(links))
    # ...or they are connected via an explicit list of labels to be linked. 
    if links is not None and len(links)>0:
        raise NotImplementedError() # [pluriseg]
        # is_link = np.zeros(piece_masks.shape, dtype=np.bool_)
        # is_link = get_link_matrix(links, piece_masks, np.concatenate(inds), idx, is_link)
        # conditions.append(is_link)
        
    affinity_graph = np.logical_or.reduce(conditions) 
    affinity_graph[idx] = 0 # no self connections
    
    # We may not want all masks to be reflected across the edge. Thresholding by distance field
    # is a good way to make sure that cells are not doubled up along their boundary. 
    if dists is not None:
        print('debug: check this')
        affinity_graph[is_edge] = dists[tuple(neighbors)][idx][np.nonzero(is_edge)[-1]]>cutoff
    
    return affinity_graph

# @njit() error 
def affinity_to_boundary(masks,affinity_graph,coords, dim=None):
    """Convert affinity graph to boundary map.
    
    Internal hypervoxels are those that are fully connected to all their 3^D-1 neighbors, 
    where D is the dimension. Boundary hypervoxels are those that are connected to fewer 
    than this number and at least 1 other hypervoxel. Correct boundaries should have >=D connections,
    but the lower bound here is set to 1. 
    
    Parameters:
    -----------
    masks: ND array, int or binary 
        label matrix or binary foreground mask
    
    affinity_graph: ND array, bool
        hypervoxel affinity array, <3^D> by <number of foreground hypervoxels>
    
    coords: tuple or ND array
        coordinates of foreground hypervoxels, <dim>x<npix>
    
    Returns:
    --------
    
    boundary
    """
    if dim is None:
        dim = masks.ndim       
    csum = np.sum(affinity_graph,axis=0)
    boundary = np.logical_and(csum<(3**dim-1),csum>0) # check this latter condition
    
    # check if spatial or npix
    # if spatial, no need to convert to mask coordinates 
    if boundary.shape == masks.shape:
        return boundary
    else:
        bd_matrix = np.zeros(masks.shape,int)
        bd_matrix[tuple(coords)] = boundary 
        return bd_matrix
    
def spatial_affinity(affinity_graph, coords, shape):
    """
    Convert affinity graph in (S,N) format to (S,*DIMS) format. 
    """
    nsteps,npix = affinity_graph.shape
    affinity = np.zeros((nsteps,)+shape)
    affinity[(Ellipsis,)+tuple(coords)] = affinity_graph
    return affinity

# @torch.no_grad() # try to solve memory leak in mps
def _extend_centers_torch(masks, centers, affinity_graph, coords=None, n_iter=200, 
                          device=torch.device('cpu'), omni=True, smooth=False, 
                          weight=1, return_flows=True, affinity_field=False, 
                          edges=None, initialize=False, verbose=False):
    """ runs diffusion on GPU to generate flows for training images or quality control
    PyTorch implementation is faster than jitted CPU implementation, therefore only the 
    GPU optimized code is being used moving forward. 
    
    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    centers: int, 2D or 3D array
        array of center coordinates [[y0,x0],[x1,y1],...] or [[t0,y0,x0],...]
    n_inter: int
        number of iterations
    device: torch device
        what compute hardware to use to run the code (GPU VS CPU)  
    omni: bool
        whether to generate Omnipose field (solve Eikonal equation) 
        or the Cellpose field (solve heat equation from "center") 
        
    Returns
    -------------
    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z (or T) = mu[0].
    dist: float, 2D or 3D array
        the smooth distance field (Omnipose)
        or temperature distribution (Cellpose)
    boundaries: bool, 2D or 3D array
        binary field representing 1-connected boundary 
         
    """
    d = masks.ndim
    shape = masks.shape
    npix = affinity_graph.shape[-1]
    steps, inds, idx, fact, sign = utils.kernel_setup(d)

    if coords is None:
        coords = np.nonzero(masks>0) # >0 to handle -1 labels at edge; do I use that anymore? check...
    else:
        coords = tuple(coords)
    # we want to index the flatened pixel list T will of shape (npix,)
    neighbors = utils.get_neighbors(coords,steps,d,shape,edges) # shape (d,3**d,npix)   
    indexes, neigh_inds, ind_matrix = utils.get_neigh_inds(tuple(neighbors),coords,shape)

    central_inds = ind_matrix[tuple(neighbors[:,idx])]
    centroid_inds = ind_matrix[tuple(centers)] if len(centers) else np.zeros(0)

    if verbose:
        print('affinity_graph',affinity_graph.shape,affinity_graph.dtype)
        print('index shape',indexes.shape)
        print('neighbors shape',neighbors.shape)
        print('neigh_inds shape',neigh_inds.shape)
        print('central_inds shape',central_inds.shape)
        print('centroid_inds shape',centroid_inds.shape)

    # previous neighbor-finding code has been replaced with affinity_graph code 
    # this is always precomputed by this stage 

    dtype = torch.float
    # T = torch.zeros(npix, dtype=dtype, device=device)
    T =  torch.ones(npix, dtype=dtype, device=device)

    d = torch.tensor(d)
    idx = torch.tensor(idx)
    fact = torch.tensor(fact)
    steps = torch.tensor(steps,device=device)        
    inds = tuple([torch.tensor(i) for i in inds])
    omni = torch.tensor(omni)
    smooth = torch.tensor(smooth)
    verbose = torch.tensor(verbose)

    isneigh = torch.tensor(affinity_graph,device=device,dtype=torch.bool) # isneigh shape (3**d,npix)
    neigh_inds = torch.tensor(neigh_inds,device=device)
    central_inds = torch.tensor(central_inds,device=device,dtype=torch.long)
    centroid_inds = torch.tensor(centroid_inds,device=device,dtype=torch.long)

    if affinity_field:
        # experimenting with using the connectivity graph to define the scalar field precition class
        T = torch.tensor(affinity_graph,device=device,dtype=dtype).sum(axis=0)
    else:
        if initialize and d<=3:
            T = torch.tensor(edt.edt(masks)[coords],device=device) 

        if n_iter is None:
            n_iter = torch.tensor(50)
        else:
            n_iter = torch.tensor(n_iter)

        T = _iterate(T,neigh_inds,central_inds,centroid_inds,
                     idx,d,inds,fact,isneigh,n_iter,omni,smooth,verbose)

    ret = []
    
    if return_flows:
        # calculate gradient with contributions along cardinal, ordinal, etc. 
        # new implementation is 30x faster than an earlier version 
        n_axes = len(fact)-1
        s = [n_axes,d,isneigh.shape[-1]]
        mu_ = torch.zeros((d,)+shape,device=device,dtype=dtype)
        mu_[(Ellipsis,)+coords] = _gradient(T,d,steps,fact,inds,isneigh,neigh_inds,central_inds,s)
        if verbose:
            print('mu',mu_.shape)
        ret += [mu_] # .detach() adds a lot of time? 
    
    # put back into ND
    T_ = torch.zeros(shape,device=device,dtype=dtype)
    T_[coords] = T
    
    # put it first 
    ret = [T_]+ret
    
    return (*ret,)


@torch.jit.script # saves maybe 10%
def update_torch(a,f,fsq):
    # Turns out we can just avoid a ton of individual if/else by evaluating the update function
    # for every upper limit on the sorted pairs. I do this by pieces using cumsum. The radicand
    # being nonegative sets the upper limit on the sorted pairs, so we simply select the largest 
    # upper limit that works. I also put a couple of the indexing tensors outside of the loop. 
    """Update function for solving the Eikonal equation. """
    a,_ = torch.sort(a,dim=0) # sorting was the source of the small artifact bug 
    am = a*((a-a[-1])<f)
    sum_a = am.sum(dim=0)
    sum_a2 = (am**2).sum(dim=0)
    # return (1/d)*(sum_a+torch.sqrt(torch.clamp((sum_a**2)-d*(sum_a2-fsq),min=0)))
    # return (1/d)*(sum_a+torch.clamp((sum_a**2)-d*(sum_a2-fsq),min=0)**0.5)
    # return (1/d)*(am.sum(dim=0)+torch.clamp((am.sum(dim=0)**2)-d*((am**2).sum(dim=0)-fsq),min=0)**0.5)
    # return (1/d)*(sum_a+torch.sqrt(torch.clamp((sum_a**2)-d*(sum_a2-fsq),min=0)))
    
    d = a.shape[0] # d acutally needed to be the number of elements being compared, not dimension 
    return (1/d)*(sum_a+torch.sqrt(torch.clamp((sum_a**2)-d*(sum_a2-fsq),min=0)))
    
    

@torch.jit.script
def eikonal_update_torch(Tneigh: torch.Tensor,
                         r: torch.Tensor,
                         d: torch.Tensor,
                         index_list: list[torch.Tensor],
                         factors: torch.Tensor):
    """Update for iterative solution of the eikonal equation on GPU."""
    # preallocate array to multiply into to do the geometric mean
    # Tneigh always has shape 1 x nconnections x npix
    geometric = 1
    phi_total = torch.ones_like(Tneigh[0,:]) if geometric else torch.zeros_like(Tneigh[0,:])
    
    # loop over each index list + weight factor 
    n = len(factors) - 1
    w = 0.

    for inds,f,fsq in zip(index_list[1:],factors[1:],factors[1:]**2):    
    
        # find the minimum of each hypercube pair along each axis
        npair = len(inds)//2
        
        # mins = torch.stack([torch.fmin(Tneigh[inds[i],:],Tneigh[inds[-(i+1)],:]) for i in range(npair)])
        mins = torch.stack([torch.minimum(Tneigh[inds[i],:],Tneigh[inds[-(i+1)],:]) for i in range(npair)])
        
        # apply update rule using the array of mins, 
        update = update_torch(mins,f,fsq)
        
        # put into storage array
        if geometric:
            phi_total *= update
        else:
            phi_total += update
            
    phi_total = torch.pow(phi_total,1/n) if geometric else phi_total/n
    
    return phi_total


@torch.jit.script
def _iterate(T: torch.Tensor, # 1D tensor of scalar values at each pixel
             neigh_inds: torch.Tensor, 
             central_inds: torch.Tensor, 
             centroid_inds: torch.Tensor, 
             idx: torch.Tensor,
             d: torch.Tensor,
             inds: list[torch.Tensor],
             fact: torch.Tensor,
             isneigh: torch.Tensor,
             n_iter: torch.Tensor,
             omni: torch.Tensor,
             smooth: torch.Tensor, 
             verbose: torch.Tensor):
    
    T0 = T.clone()
    eps = 1e-3 if not smooth else 1e-8
    # eps = 1e-5
    
    # n_iter = 200
    if verbose:
        print('eps is ', eps, 'n_iter is', n_iter)
    
    # I wonder if it is possible to reduce the update grid after points converge 
    t = torch.tensor(0)
    not_converged = torch.tensor(True)
    error = torch.tensor(1)
    npix = isneigh.shape[-1]

    
    # r = torch.arange(0,npix)
    r = central_inds 
    
    while not_converged:
        if omni:# and OMNI_INSTALLED:
            Tneigh = T[neigh_inds]
            Tneigh *= isneigh #zeros out any elements that do not belong in convolution
            T = eikonal_update_torch(Tneigh,r,d,inds,fact) # now central_inds = 0,1,2,3,...
        else:
            T[centroid_inds] += 1

        # error = mse(T,T0)
        error = (T-T0).square().mean() #faster than mse function
        
        if omni:
            not_converged = torch.logical_and(error>eps, t<n_iter)
            # not_converged = torch.logical_and(torch.tensor(error>eps), torch.tensor(t<n_iter))
            # not_converged = torch.logical_and(error>eps, torch.tensor(t<n_iter))
            
        else:
            not_converged = t<n_iter
            
        # helps to do a bit of smoothing to start get the signal propagated
        if not omni or t<1 or smooth:  #  or not not_converged
            Tneigh = T[neigh_inds]
            Tneigh *= isneigh
            T = Tneigh.mean(dim=0) # mean along the <3**d>-element column does the box convolution 
            
        # update the old one 
        T0.copy_(T) # faster than T0 = T.clone() or  T0[:] = T

        t+=1

    if verbose:
        print('iter: ',t,'{:.10f}'.format(error))
    # There is still a fade out effect on long cells, not enough iterations to diffuse far enough I think 
    # The log operation does not help much to alleviate it, would need a smaller constant inside. 
    if not omni:
        T = torch.log(1.+ T)

    return T


@torch.jit.script 
def _gradient(T,d,steps,fact,
              inds: list[torch.Tensor],
              isneigh,
              neigh_inds: torch.Tensor,
              central_inds: torch.Tensor, 
              s: list[int]
             ):

    finite_differences = torch.zeros(s,device=T.device,dtype=T.dtype)
    cvals = T[central_inds]
    for ax,(ind,f) in enumerate(zip(inds[1:],fact[1:])):

        vals = T[neigh_inds[ind]] # maybe go back to passing neigh_vals
        vals[~isneigh[ind]] = 0 # T[]*mask prevent bleedover / boundary issues, big problem in stock Cellpose that got reverted!

        mid = len(ind)//2
        r = torch.arange(mid)
        # unit vectors 
        vecs = steps[ind].float()
        uvecs = (vecs[-(r+1)] - vecs[r]).T #/(2*f) #move normalization to end for speed

        # calculate differences along each axis with directional pairs  
        diff = (vals[-(r+1)]-vals[r]) # /(2*f)

        # dot products, project differences onto cardinal coorinate system 
        finite_differences[ax] = torch.matmul(uvecs,diff) / (2*f)**2
        # finite_differences[ax] = torch.einsum('ij,jk->ik', uvecs, diff)  / (2*f)**2

        
    mu = torch.mean(finite_differences,dim=0) 

    # do some averaging with neighbors, but weighted by dot product so that magnitude does not fall off
    weight = torch.sum(mu[:,neigh_inds]*(mu[:,central_inds].unsqueeze(1)),dim=0).abs() # A.B
    weight[~isneigh] = 0
    wsum = weight.sum(dim=0)
    return torch.where(wsum!=0,
                       (mu[:,neigh_inds]*weight).sum(dim=1) / wsum,
                       torch.zeros_like(wsum))






# ## Section II: mask recontruction

def compute_masks(dP, dist, affinity_graph=None, bd=None, p=None, coords=None, iscell=None, niter=None, rescale=1.0, resize=None, 
                  mask_threshold=0.0, diam_threshold=12.,flow_threshold=0.4, 
                  interp=True, cluster=False, boundary_seg=False, affinity_seg=False, do_3D=False, 
                  min_size=None, max_size=None, hole_size=None, omni=True, 
                  calc_trace=False, verbose=False, use_gpu=False, device=None, nclasses=2, 
                  dim=2, eps=None, hdbscan=False, flow_factor=6, debug=False, override=False, suppress=None, despur=False):
    """
    Compute masks using dynamics from dP, dist, and boundary outputs.
    Called in cellpose.models(). 
    
    Parameters
    -------------
    dP: float, ND array
        flow field components (2D: 2 x Ly x Lx, 3D: 3 x Lz x Ly x Lx)
    dist: float, ND array
        distance field (Ly x Lx)
    bd: float, ND array
        boundary field
    p: float32, ND array
        initial locations of each pixel before dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].  
    coords: int32, 2D array
        non-zero pixels to run dynamics on [npixels x D]
    niter: int32
        number of iterations of dynamics to run
    rescale: float (optional, default None)
        resize factor for each image, if None, set to 1.0   
    resize: int, tuple
        shape of array (alternative to rescaling)  
    mask_threshold: float 
        all pixels with value above threshold kept for masks, decrease to find more and larger masks 
    flow_threshold: float 
        flow error threshold (all cells with errors below threshold are kept) (not used for Cellpose3D)
    interp: bool 
        interpolate during dynamics
    cluster: bool
        use sub-pixel DBSCAN clustering of pixel coordinates to find masks
    do_3D: bool (optional, default False)
        set to True to run 3D segmentation on 4D image input
    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1
    omni: bool 
        use omnipose mask recontruction features
    calc_trace: bool 
        calculate pixel traces and return as part of the flow
    verbose: bool 
        turn on additional output to logs for debugging 
    use_gpu: bool
        use GPU of flow_threshold>0 (computes flows from predicted masks on GPU)
    device: torch device
        what compute hardware to use to run the code (GPU VS CPU)
    nclasses:
        number of output classes of the network (Omnipose=3,Cellpose=2)
    dim: int
        dimensionality of data / model output
    eps: float
        internal epsilon parameter for (H)DBSCAN
    hdbscan: 
        use better, but much SLOWER, hdbscan clustering algorithm (experimental)
    flow_factor:
        multiple to increase flow magnitude (used in 3D only, experimental)
    debug:
        option to return list of unique mask labels as a fourth output (for debugging only)

    Returns
    -------------
    mask: int, ND array
        label matrix
    p: float32, ND array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. 
    tr: float32, ND array
        intermediate locations of each pixel during dynamics,
        size [axis x niter x Ly x Lx] or [axis x niter x Lz x Ly x Lx]. 
        For debugging/paper figures, very slow. 
    bd: float32, ND array
        boundary map
    augmented_affinity: float32, ND array
        concatenated coordinates and affinity graph, hence (d+1,3**d,npix)
    """

    # do everything in padded arrays for boundary/affinity functions - not needed anymore?
    pad = 0 ##
    # pad = 1 ##
    # print('pad',pad)
    if do_3D:
        dim = 3 
    pad_seq = [(0,)*2]+[(pad,)*2]*dim
    unpad = tuple([slice(pad,-pad) if pad else slice(None,None)]*dim) # works in case pad is zero

    if hole_size is None:
        hole_size = 3**(dim//2) # just a guess

    labels = None
    
    if verbose:
        startTime0 = time.time()
        omnipose_logger.info(f'mask_threshold is {mask_threshold}')
        if omni and (not SKIMAGE_ENABLED):
             omnipose_logger.warning('Omni enabled but skimage not enabled')
    
    # inds very useful for debugging and figures; allows us to easily specify specific indices for Euler integration
    if iscell is None:
        if coords is not None:
            iscell = np.zeros_like(dist,dtype=np.int32)
            iscell[tuple(coords)] = 1
        else:
            if (omni and SKIMAGE_ENABLED) or override:
                if verbose:
                    omnipose_logger.info('Using hysteresis threshold.')
                iscell = filters.apply_hysteresis_threshold(dist, mask_threshold-1, mask_threshold) # good for thin features

            else:
                iscell = dist > mask_threshold # analog to original iscell=(cellprob>cellprob_threshold)
    
    
    # if nclasses>1, we can do instance segmentation. 
    if np.any(iscell) and nclasses>1: 

        iscell_pad = np.pad(iscell,pad) # I should get rid of all padding commands, padding is zero now 
        coords = np.array(np.nonzero(iscell_pad)).astype(np.int32)       
        shape = iscell_pad.shape
        
        
        # for boundary later, also for affinity_seg option
        # steps = utils.get_steps(dim) # perhaps should factor this out of the function 

        if suppress is None:
            suppress = omni and not affinity_seg # Euler suppression ON with omni unless affinity seg 

        #preprocess flows
        if omni and OMNI_INSTALLED:

            # Euler suppression may be bad in 3D in general, fyi 
            if suppress:# and not affinity_seg:
                # dP_ = div_rescale(dP,iscell) / rescale ##### omnipose.core.div_rescale
                # print('testing something new')
                # dP_ = utils.normalize_field(dP)
                # dP_ *= (1-utils.rescale(dist))
                
                # this is the winner I think 
                dP_ = div_rescale(dP,iscell) / rescale ##### omnipose.core.div_rescale
                # dP_ /= np.clip(dist,1,np.inf) # this is a problem in some places, 06/13/2023
            else:
                dP_ = dP.copy()/5.
            

            # else:
            #     dP_ = utils.normalize_field(dP)
            # dP_ = bd_rescale(dP,mask, 4*bd) / rescale ##### omnipose.core.div_rescale
                
            # dP_ = dP.copy()
            if dim>2 and suppress:
                dP_ *= flow_factor
                print('dP_ times {} for >2d, still experimenting'.format(flow_factor))

        else:
            dP_ = dP * iscell / 5.
        
        dP_pad = np.pad(dP_,pad_seq)
        dt_pad = np.pad(dist,pad)
        bd_pad = np.pad(bd,pad)
        bounds = None        
        
        # boundary seg can be stupid fast but it is a little broken
        if boundary_seg: # new tactic is to use flow to compute boundaries, including self-contact ones
            raise NotImplementedError() # [pluriseg]
            # if verbose:
            #     omnipose_logger.info('doing new boundary seg')
            # bd = get_boundary(np.pad(dP,pad_seq),iscell_pad)
            # labels, bounds, _ = boundary_to_masks(bd,iscell_pad) 

            # hole_size = 0 # turn off small hole filling, still do area threhsolding 
            
            # # compatibility 
            # p = np.zeros([2,1,1])
            # tr = []
            
        else: # do the ol' Euler-integration + clustering 

            # the clustering algorithm requires far fewer iterations because it 
            # can handle subpixel separation to define blobs, whereas the thresholding method
            # requires blobs to be separated by more than 1 pixel 
            # new affinity_seg does not do Euler supression and benefits from moderate point clustering 
            if (cluster or affinity_seg or not suppress) and niter is None:
                # niter = int(diameters(iscell,dist))
                # dividing by two is sometimes necessary, but it seems like it might be generally more harm than good
                niter = int(diameters(iscell,dist)/(1+affinity_seg))

                # if verbose:
                #     omnipose_logger.info('niter is now {}'.format(niter))
                
            if p is None:
                p, coords, tr = follow_flows(dP_pad, dt_pad, coords, niter=niter, interp=interp,
                                            use_gpu=use_gpu, device=device, omni=omni, 
                                            suppress= suppress,
                                            calc_trace=calc_trace, verbose=verbose)
            else:
                tr = []
                if verbose:
                    omnipose_logger.info('p given')
                # print('a2',shape,p.shape,coords.shape,p.max(), p[:,~iscell_pad].shape)
                
                # set the points that are background to not move
                p[:,~iscell_pad] = np.stack(np.nonzero(~iscell_pad))
                    

            #calculate masks
            if (omni and OMNI_INSTALLED) or override:
                steps, inds, idx, fact, sign = utils.kernel_setup(dim)
                if affinity_seg:
                    hole_size = 0 # turn off small hole filling, still do area threhsolding 
                    if affinity_graph is None:       
                        if verbose:
                            omnipose_logger.info('computing affinity graph')
                        # assuming we have no passed in the affinity graph, we need to compute it  
                        # affinity_graph, neighbors, neigh_inds = _get_affinity(steps,
                        #                                                     iscell_pad,
                        #                                                     dP_pad,
                        #                                                     dt_pad,
                        #                                                     p,
                        #                                                     coords, 
                        #                                                     pad=pad)
                        
                        initial_points = np.stack(meshgrid(iscell_pad.shape))
                        final_points = p
                        supporting_inds = utils.get_supporting_inds(steps)

                        # print('p',final_points.shape, initial_points.shape)
                        # if we can keep the points and predictions on GPU, we could make this a lot faster...
                        # especially if we can optmize euler+affinity together
                        affinity_graph = _get_affinity_torch(initial_points, 
                                                            final_points, 
                                                            dP_pad, #<<<<<<<<<<< add support for other options here 
                                                            dt_pad, 
                                                            iscell_pad, 
                                                            steps,
                                                            fact,
                                                            inds,
                                                            supporting_inds,
                                                            niter,
                                                            device=device # would default to torch GPU otherwise?
                                                            )
                        affinity_graph = affinity_graph.squeeze().cpu().numpy()
                        affinity_graph = affinity_graph[(Ellipsis,)+tuple(coords)]
                    
                    
                    # despur really not needed anymore, handled by the affinity grpah torch version
                    # elif despur:
                        # if it is passed in, we need the neigh_inds to compute masks 
                        # (though eventually we will want this to also be in parallel on GPU...)
                    
                    neighbors = utils.get_neighbors(tuple(coords),steps,dim,shape, pad=pad) # shape (d,3**d,npix)
                    indexes, neigh_inds, ind_matrix = utils.get_neigh_inds(tuple(neighbors),tuple(coords),shape)
                    
                    despur = dim==2 and despur # only do despur in 2D 
                    if verbose and not despur:
                        omnipose_logger.info('despur disabled')
                    
                    if despur:
                        non_self = np.array(list(set(np.arange(len(steps)))-{inds[0][0]})) # I need these to be in order
                        cardinal = np.concatenate(inds[1:2])
                        ordinal = np.concatenate(inds[2:])
                        
                        affinity_graph = _despur(affinity_graph, 
                                                neigh_inds, 
                                                indexes, 
                                                steps, 
                                                non_self, 
                                                cardinal, 
                                                ordinal, 
                                                dim)
                        
                        # I need to make sure that the masks/coords also get updated... that's what affinity_to_masks does 
                        # altertnatiely, the affinity_to_boundary then boundary_to_masks does this 
                  
                    bounds = affinity_to_boundary(iscell_pad,affinity_graph,tuple(coords))
                        
                    if cluster:
                        labels = affinity_to_masks(affinity_graph,neigh_inds,iscell_pad,coords,verbose=verbose)
                        # move bounds here, out of get affinity 
                    else:
                        # maybe faster version that skips connected components using the affinity graph
                        #  and instead uses the boundary output to define masks (implict connected components)
                        if verbose:
                            omnipose_logger.info('doing affinity seg without cluster.')
                        labels, bounds, _ = boundary_to_masks(bounds,iscell_pad) 
                                        
                else:
                    labels, _ = get_masks(p, bd_pad, dt_pad, iscell_pad, coords, nclasses, cluster=cluster,
                                             diam_threshold=diam_threshold, verbose=verbose, 
                                             eps=eps, hdbscan=hdbscan) ##### omnipose.core.get_masks
                    affinity_graph = None # could replace with masks to affinity 
                    coords = np.nonzero(labels)                
            else:
                labels = get_masks_cp(p, iscell=iscell_pad,
                                      flows = dP_pad if flow_threshold>0 else None, 
                                      use_gpu=use_gpu) ### just get_masks

        # flow thresholding factored out of get_masks
        # still could be useful for boundaries! TODO: Need to put in the self-contact boundaries as input <<<<<<
        # also can now turn on for do_3D... 
        
        if not do_3D: 
            flows = np.pad(dP,pad_seq) # original flow
            shape0 = flows.shape[1:]
            if labels.max()>0 and flow_threshold is not None and flow_threshold > 0 and flows is not None:
                # print('aaa',np.count_nonzero(labels),np.array(coords).shape,affinity_graph.shape)
                labels = remove_bad_flow_masks(labels, flows, 
                                               coords=coords, 
                                               affinity_graph=affinity_graph, 
                                               threshold=flow_threshold,
                                               use_gpu=use_gpu, 
                                               device=device, 
                                               omni=omni)
                _,labels = np.unique(labels, return_inverse=True)
                labels = np.reshape(labels, shape0).astype(np.int32)
        
        
        # need to reconsider this for self-contact... ended up just disabling with hole size 0
        # print('dd',iscell_pad.shape,labels.shape)
        masks = fill_holes_and_remove_small_masks(labels, min_size=min_size, max_size=max_size, ##### utils.fill_holes_and_remove_small_masks
                                                 hole_size=hole_size, dim=dim)*iscell_pad 
        # masks = labels
        # Resize mask, semantic or instance 
        resize_pad = np.array([r+2*pad for r in resize]) if resize is not None else labels.shape
        if tuple(resize_pad)!=labels.shape:
            if verbose:
                omnipose_logger.info(f'resizing output with resize = {resize_pad}')
            # mask = resize_image(mask, resize[0], resize[1], interpolation=0).astype(np.int32) 
            ratio = np.array(resize_pad)/np.array(labels.shape)
            masks = zoom(masks, ratio, order=0).astype(np.int32) 
            iscell_pad = masks>0
            dt_pad = zoom(dt_pad, ratio, order=1)
            dP_pad = zoom(dP_pad, np.concatenate([[1],ratio]), order=1) # for boundary 
            
            # affinity_seg not compatible with rescaling after euler integration
            # would need to upscale predcitons first 
            if verbose and affinity_seg:
                omnipose_logger.info('affinity_seg not compatible with rescaling, disabling')
            affinity_seg = False
            
        if not affinity_seg or boundary_seg:
            bounds = find_boundaries(masks,mode='inner',connectivity=dim)

        # If using default omnipose/cellpose for getting masks, still try to get accurate boundaries 
        if bounds is None:
            if verbose:
                print('Default clustering on, finding boundaries via affinity.')
            print('TO-DO: replace with _get_affinity_torch')
            affinity_graph, neighbors, neigh_inds, bounds = _get_affinity(steps,masks,dP_pad,dt_pad,p,inds, pad=pad)

            # boundary finder gets rid of some edge pixels, remove these from the mask 
            gone = neigh_inds[3**dim//2,np.sum(affinity_graph,axis=0)==0]
            # coords = np.argwhere(masks)
            crd = coords.T 
            masks[tuple(crd[gone].T)] = 0 
            iscell_pad[tuple(crd[gone].T)] = 0 
        else:
            # ensure that the boundaries are consistent with mask cleanup
            # only small masks would be deleted here, no changes otherwise to boundaries 
            bounds *= masks>0

        fastremap.renumber(masks,in_place=True) #convenient to guarantee non-skipped labels

        # moving the cleanup to the end helps avoid some bugs arising from scaling...
        # maybe better would be to rescale the min_size and hole_size parameters to do the
        # cleanup at the prediction scale, or switch depending on which one is bigger... 
        
        masks_unpad = masks[unpad] if pad else masks 
        bounds_unpad = bounds[unpad] if pad else bounds
        
        if affinity_seg:

            # I also want to return the raw affinity graph
            # the problem there is that it is computed on the padded array
            # besides unpadding, I need to delete columns for missing pixels 

            # Idea here is that I index everything corresponding to the affinity graph first


            # then I figure out which of these columns correspond to pixels that are in the final masks
            # this works by looking at an array of indices the same size as the image, and any pixels not part
            # of the original affinity graph do not participate, i.e. hole filling does not work 
            coords_remaining = np.nonzero(masks)
            inds_remaining = ind_matrix[coords_remaining]
            affinity_graph_unpad = affinity_graph[:,inds_remaining]
            neighbors_unpad = neighbors[...,inds_remaining] - pad

            # I also want to package the affinity graph with the pixel coordinates 
            # then there is no ambiguity and can extract a binary mask
            # thus the augmented affinity graph would be (d+1,3**d,npix)

            augmented_affinity = np.vstack((neighbors_unpad,affinity_graph_unpad[np.newaxis]))


            # # newer version that takes care of mask cleanup as well
            # # NOTE: without padding, this subsample affinity may be very overkill
            # # all I need to do is truncate the affinity graph so that neighbors and affinity are deleted where cleanup occurred 
            # slc = tuple([slice(pad,shape[d]-pad) for d in range(dim)]) 
            # augmented_affinity = np.vstack((neighbors,affinity_graph[np.newaxis]))
            # augmented_affinity = utils.subsample_affinity(augmented_affinity,slc,masks)


            # this also applied to the traced pixels
            if calc_trace:
                # tr = tr[:,inds_remaining]-pad
                print('warning calc trace not cropped')
                
        else:
            augmented_affinity = []
        
        ret = [masks_unpad, p, tr, bounds_unpad, augmented_affinity]
        
    else: # nothing to compute, just make it compatible
        omnipose_logger.info('No cell pixels found.')
        ret = [iscell, np.zeros([2,1,1]), [], iscell, []]
    
    if debug:
        ret += [labels] # also return the version of labels are prior to filling holes etc. 


    if verbose:
        executionTime0 = (time.time() - startTime0)
        omnipose_logger.info('compute_masks() execution time: {:.3g} sec'.format(executionTime0))
        if labels is not None:
            omnipose_logger.info('\texecution time per pixel: {:.6g} sec/px'.format(executionTime0/np.prod(labels.shape)))
            omnipose_logger.info('\texecution time per cell pixel: {:.6g} sec/px'.format(np.nan if not np.count_nonzero(labels) else executionTime0/np.count_nonzero(labels)))
        else:
            omnipose_logger.info('\tno objects found')

    return (*ret,)


# Omnipose requires (a) a special suppressed Euler step and (b) a special mask reconstruction algorithm. 

# no reason to use njit here except for compatibility with jitted fuctions that call it 
# this way, the same factor is used everywhere (CPU with/without interp, GPU)
# @njit()
def step_factor(t):
    """ Euler integration suppression factor.
    
    Conveneient wrapper function allowed me to test out several supression factors. 
    
    Parameters
    -------------
    t: int
        time step
    """
    return (1+t)

def div_rescale(dP,mask,p=1):
    """
    Normalize the flow magnitude to rescaled 0-1 divergence. 
    
    Parameters
    -------------
    dP: float, ND array
        flow field 
    mask: int, ND array
        label matrix
        
    Returns
    -------------
    dP: float, ND array
        rescaled flow field
    
    """
    dP = dP.copy()
    dP *= mask 
    dP = utils.normalize_field(dP)
    if p>0:
        # div = utils.normalize99(likewise(dP))
        div = utils.normalize99(divergence(dP))**p
        dP *= div
    return dP


def divergence(f,sp=None):
    """ Computes divergence of vector field
    
    Parameters
    -------------
    f: ND array, float
        vector field components [Fx,Fy,Fz,...]
    sp: ND array, float
        spacing between points in respecitve directions [spx, spy, spz,...]
        
    """
    num_dims = len(f)
    if any(f.shape[1+i] < 2 for i in range(num_dims-1)):
        return np.zeros_like(f[0])
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])


def get_masks(p, bd, dist, mask, inds, nclasses=2,cluster=False,
              diam_threshold=12., eps=None, min_samples=5, hdbscan=False, verbose=False):
    """Omnipose mask recontruction algorithm.
    
    This function is called after dynamics are run. The final pixel coordinates are provided, 
    and cell labels are assigned to clusters found by labeling the pixel clusters after rounding
    the coordinates (snapping each pixel to the grid and labeling the resulting binary mask) or 
    by using DBSCAN or HDBSCAN for sub-pixel clustering. 
    
    Parameters
    -------------
    p: float32, ND array
        final locations of each pixel after dynamics
    bd: float, ND array
        boundary field
    dist: float, ND array
        distance field
    mask: bool, ND array
        binary cell mask
    inds: int, ND array 
        initial indices of pixels for the Euler integration [npixels x ndim]
    nclasses: int
        number of prediciton classes
    cluster: bool
        use DBSCAN clustering instead of coordinate thresholding
    diam_threshold: float
        mean diameter under which clustering will be turned on automatically
    eps: float
        internal espilon parameter for (H)DBSCAN
    hdbscan: bool
        use better, but much SLOWER, hdbscan clustering algorithm
    verbose: bool
        option to print more info to log file
    
    Returns
    -------------
    mask: int, ND array
        label matrix
    labels: int, list
        all unique labels 
    """
    if nclasses > 1:
        dt = np.abs(dist[mask]) #abs needed if the threshold is negative
        d = dist_to_diam(dt,mask.ndim) 

    else: #backwards compatibility, doesn't help for *clusters* of thin/small cells
        d = diameters(mask,dist)
    
    if eps is None:
        eps = 2**0.5
    # The mean diameter can inform whether or not the cells are too small to form contiguous blobs.
    # My first solution was to upscale everything before Euler integration to give pixels 'room' to
    # stay together. My new solution is much better: use a clustering algorithm on the sub-pixel coordinates
    # to assign labels. It works just as well and is faster because it doesn't require increasing the 
    # number of points or taking time to upscale/downscale the data. Users can toggle cluster on manually or
    # by setting the diameter threshold higher than the average diameter of the cells. 
    if verbose:
        omnipose_logger.info('Mean diameter is %f'%d)

    if d <= diam_threshold: #diam_threshold needs to change for 3D
        cluster = True
        if verbose and not cluster:
            omnipose_logger.info('Turning on subpixel clustering for label continuity.')
    
    cell_px = tuple(inds)
    coords = np.nonzero(mask)
    newinds = p[(Ellipsis,)+cell_px].T
    mask = np.zeros(p.shape[1:],np.uint32)
    
    # the eps parameter needs to be opened as a parameter to the user
    if verbose:
        omnipose_logger.info('cluster: {}, SKLEARN_ENABLED: {}'.format(cluster,SKLEARN_ENABLED))
        
    
    if cluster and SKLEARN_ENABLED:
        if verbose:
            startTime = time.time()
            alg = ['','H']
            omnipose_logger.info('Doing {}DBSCAN clustering with eps={}, min_samples={}'.format(alg[hdbscan],eps,min_samples))

        if hdbscan and not HDBSCAN_ENABLED:
            omnipose_logger.warning('HDBSCAN clustering requested but not installed. Defaulting to DBSCAN')
        
        if hdbscan and HDBSCAN_ENABLED:
            #sklearn dbscan and hdbscan are really slow compared to the dbscan package below, 
            # cosndier depricating this option
            clusterer = HDBSCAN(cluster_selection_epsilon=eps,
                                # allow_single_cluster=True,
                                min_samples=min_samples)
            
            clusterer.fit(newinds)
            labels = clusterer.labels_
        else:
            # [pluriseg] using sklearn since no dbscane for numpy > 2.0
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            clusterer.fit(newinds)
            labels = clusterer.labels_


        # filter out small clusters
        # unique_labels = set(labels) - {-1,0}
        # for l in unique_labels:
        #     hits = labels==l
        #     if np.sum(hits)<9:
        #         labels[hits] = -1 # make outliers
        
        if verbose:
            executionTime = (time.time() - startTime)
            omnipose_logger.info('Execution time in seconds: ' + str(executionTime))
            omnipose_logger.info('{} unique labels found'.format(len(np.unique(labels))-1))

        #### snapping outliers to nearest cluster
        # there was a bug where small clusters counted as outliers, and were snapped to a very distant cluster
        # I will fix this by enfocing a limit on distance to the nearest cluster
        # min_samples could also be reduced... 
        snap = 1
        if snap:
            nearest_neighbors = NearestNeighbors(n_neighbors=5) # maybe should be 5 instead of 50 
            neighbors = nearest_neighbors.fit(newinds)
            o_inds = np.where(labels==-1)[0]
            if len(o_inds):
                outliers = [newinds[i] for i in o_inds]
                nearest_dists, nearest_indices = neighbors.kneighbors(outliers)
                
                # get the labels of the nearest neighbors
                nearest_labels = labels[nearest_indices]
                
                # find the first instance that is not in reference to other ouliers
                nearest_idx = [np.where(n!=-1)[0][0] if np.any(n!=-1) else 0 for n in nearest_labels]
                dist_thresh = eps
                l = [nl[i] if nd[i]<dist_thresh else -1 for i,nl,nd in zip(nearest_idx,nearest_labels,nearest_dists)]
                labels[o_inds] = l
                if verbose:
                    omnipose_logger.info(f'Outlier cleanup with dist threshold {dist_thresh:.2f}:')
                    distances = [nd[i] for i,nd in zip(nearest_idx,nearest_dists)]
                    omnipose_logger.info(f'\tmin and max distance to nearest cluster: {np.min(distances):.2f},{np.max(distances):.2f}')
                    omnipose_logger.info('\tSnapped {} of {} outliers to nearest cluster'.format(np.sum(np.array(l)!=-1),len(o_inds)))

        ###
        mask[cell_px] = labels+1 # outliers have label -1
        
    else: #this branch can have issues near edges 
        newinds = np.rint(newinds.T).astype(int)
        new_px = tuple(newinds)
        skelmask = np.zeros_like(dist, dtype=bool)
        skelmask[new_px] = 1

        #disconnect skeletons at the edge, 5 pixels in 
        border_mask = np.zeros(skelmask.shape, dtype=bool)
        border_px =  border_mask.copy()
        border_mask = binary_dilation(border_mask, border_value=1, iterations=5)

        border_px[border_mask] = skelmask[border_mask]
        if verbose:
             omnipose_logger.info('nclasses: {}, mask.ndim: {}'.format(nclasses,mask.ndim))
        if nclasses == mask.ndim+2: #can use boundary to erase joined edge skelmasks 
            border_px[bd>-1] = 0
            if verbose:
                omnipose_logger.info('Using boundary output to split edge defects.')
        # else: #otherwise do morphological opening to attempt splitting 
        #     # border_px = binary_opening(border_px,border_value=0,iterations=3)

        skelmask[border_mask] = border_px[border_mask]

        if SKIMAGE_ENABLED:
            cnct = skelmask.ndim #-1
            labels = measure.label(skelmask,connectivity=cnct) #is this properly generalized to ND? seems like it works
        else:
            labels = label(skelmask)[0]
        mask[cell_px] = labels[new_px]
        
    if verbose:
        omnipose_logger.info('Done finding masks.')
    return mask, labels


# Generalizing to ND. Again, torch required but should be plenty fast on CPU too compared to jitted but non-explicitly-parallelized CPU code.
# also should just rescale to desired resolution HERE instead of rescaling the masks later... <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# grid_sample will only work for up to 5D tensors (3D segmentation). Will have to address this shortcoming if we ever do 4D. (see my pull request to torchvf for this
# I got rid of the map_coordinates branch, I tested execution times and pytorch implemtation seems as fast or faster
# @torch.jit.script

# deleted steps interp in favor of just using steps_batch as a unified function in ND.
# can use nearest interpolation if needed. 

def steps_batch(p, dP, niter, omni=True, suppress=True, interp=True, 
                 calc_trace=False, calc_bd=False, verbose=False):
    """Euler integration of pixel locations p subject to flow dP for niter steps in N dimensions. 
    
    Parameters
    ----------------
    p: float32, tensor
        pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)
    dP: float32, ND array
        flows [axis x Lz x Ly x Lx]
    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------
    p: float32, ND array
        final locations of each pixel after dynamics

    """
    align_corners = True
    # mode = 'nearest' if (omni and not suppress) else 'bilinear'
    
    # we want to use bilinear interpolation if using Euler suppression 
    # Affinity reconstruction does not require Euler suppression
    # and we want to also be able to toggle this globally with interp arg
    # (omni and and not suppress) is false when affinity is on
    interp = interp and not suppress
    mode = 'bilinear' if interp else 'nearest'
    if verbose:
        omnipose_logger.info(f'interp is {interp}, interpolation mode is {mode}')
    
    d = dP.shape[1] # number of components = number of dimensions 
    shape = dP.shape[2:] # shape of component array is the shape of the ambient volume 
    inds = list(range(d))[::-1] # grid_sample requires a particular ordering

    # print('inds', inds,p.shape, p.min(), p.max()) 

    device = dP.device # get the device from dP tensor

    shape = np.array(shape)[inds]-1.  # dP is d.Ly.Lx, inds flips this to flipped X-1, Y-1, ...
    # print('SHAPE',shape)
    B,D,I = p.shape
    # print('p...',p.shape,inds,shape)
    # pt = p[:,inds].permute(0,2,1).unsqueeze(1).float()
    pt = p[:,inds].permute(0,2,1).view([B]+[1]*(D-1)+[I,D]).float()
    
    # print('pt_new',pt.shape)

    pt0 = pt.clone() # save first
    flow = dP[:,inds] # inds is just flipping the spatial component ordering from TYX to XYT
    
    # print('point, flow shape',pt.shape,flow.shape)

    for k in range(d): 
        pt[...,k] = 2*pt[...,k]/shape[k] - 1 
        flow[:,k] = 2*flow[:,k]/shape[k]
    
    if calc_trace:
        dims = [-1,niter]+[-1]*(pt.ndim-1)
        trace = torch.clone(pt).detach().unsqueeze(1).expand(*dims) # add time 

    if omni and OMNI_INSTALLED and suppress:
        dPt0 = torch.nn.functional.grid_sample(flow, pt, mode=mode, align_corners=align_corners)

    for t in range(niter):
        if calc_trace and t>0:
            trace[:,t].copy_(pt)
            
        # print('aa',flow.shape,pt.shape)
        dPt = torch.nn.functional.grid_sample(flow, pt, mode=mode,
                                              align_corners=align_corners)

        if omni and OMNI_INSTALLED and suppress:
                dPt = (dPt+dPt0) / 2. # average with previous flow 
                dPt0.copy_(dPt) # update old flow 
                dPt /= step_factor(t) # suppression factor 
                
        for k in range(d): #clamp the final pixel locations
            pt[...,k] = torch.clamp(pt[...,k] + dPt[:,k], -1., 1.)
        
    pt = (pt+1)*0.5
    for k in range(d): 
        pt[...,k] *= shape[k]

    if calc_trace:
        trace = (trace+1)*0.5
        for k in range(d): 
            trace[...,k] *= shape[k]

    if calc_trace:
        # tr =  trace[...,inds].permute(0,1,-1,2,3)
        tr =  trace[...,inds].transpose(-1,1).contiguous()
        
    else:
        tr = None
    # p =  pt[...,inds].permute(0,-1,1,2)
    p =  pt[...,inds].transpose(-1,1).contiguous()
    
    empty_cache()
    return p, tr

# now generalized and simplified. Will work for ND if dependencies are updated. 
def follow_flows(dP, dist, inds, niter=None, interp=True, use_gpu=True, 
                 device=None, omni=True, suppress=False, calc_trace=False, verbose=False):
    """ define pixels and run dynamics to recover masks in 2D
    
    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------
    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
    inds: int, ND array 
        initial indices of pixels for the Euler integration 
    niter: int 
        number of iterations of dynamics to run
    interp: bool 
        interpolate during dynamics 
    use_gpu: bool 
        use GPU to run interpolated dynamics (faster than CPU)   
    omni: bool 
        flag to enable Omnipose suppressed Euler integration etc. 
    calc_trace: bool 
        flag to store and retrun all pixel coordinates during Euler integration (slow)

    Returns
    ---------------
    p: float32, ND array
        final locations of each pixel after dynamics
    inds: int, ND array
        initial indices of pixels for the Euler integration [npixels x ndim]
    tr: float32, ND array
        list of intermediate pixel coordinates for each step of the Euler integration

    """
    if verbose:
        omnipose_logger.info(f'niter: {niter}, interp: {interp}, suppress: {suppress}, calc_trace: {calc_trace}')
    
    if niter is None:
        niter = 200
    
    niter = np.uint32(niter)
    cell_px = (Ellipsis,)+tuple(inds)

   # got rid of the interp vs not interp branch in favor of just using nearest interpolation in the
   # interp code; make single batch compatible with batched integrator with unsqueezing 
    
    flow_pred = torch.tensor(dP,device=device).unsqueeze(0) 
    shape = flow_pred.shape
    B = shape[0] # this should be 1 in this branch, from unsqueezing
    dim = shape[1]
    dims = shape[-dim:] #spatial dims

    coords = [torch.arange(0, l, device=device) for l in dims]
    mesh = torch.meshgrid(coords, indexing = "ij")
    init_shape = [B, 1] + ([1] * len(dims))
    initial_points = torch.stack(mesh, dim = 0) # torchvf flips with mesh[::-1]
    initial_points = initial_points.repeat(init_shape).float()

    final_points = initial_points.clone()

    
    
    if inds.ndim < 2 or inds.shape[0] < dim:
        omnipose_logger.warning('WARNING: no mask pixels found')
        tr = None
    else:
        final_p, tr = steps_batch(initial_points[cell_px], 
                                flow_pred, 
                                niter,
                                omni=omni, # omni controls the momentum term I have 
                                suppress=suppress, # Euler suppression can be independent, i.e. with agginity_seg 
                                interp=interp,
                                calc_trace=calc_trace, 
                                verbose=verbose)
        
        final_points[cell_px] = final_p.squeeze()
    
    p = final_points.squeeze().cpu().numpy()
    if verbose:
        omnipose_logger.info('done follow_flows')
    return p, inds, tr

def remove_bad_flow_masks(masks, flows, coords=None, affinity_graph=None, threshold=0.4, use_gpu=False, device=None, omni=True):
    """ remove masks which have inconsistent flows 
    
    Uses metrics.flow_error to compute flows from predicted masks 
    and compare flows to predicted flows from network. Discards 
    masks with flow errors greater than the threshold.

    Parameters
    ----------------
    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    flows: float, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
    threshold: float
        masks with flow error greater than threshold are discarded

    Returns
    ---------------
    masks: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    merrors, _ =  flow_error(masks, flows, coords, affinity_graph, use_gpu, device, omni) ##### metrics.flow_error
    badi = 1+(merrors>threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks

def flow_error(maski, dP_net, coords=None, affinity_graph=None, use_gpu=False, device=None, omni=True):
    """ error in flows from predicted masks vs flows predicted by network run on image

    This function serves to benchmark the quality of masks, it works as follows
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted

    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
    changed in Cellpose.eval or CellposeModel.eval.

    Parameters
    ------------
    maski: ND-array (int) 
        masks produced from running dynamics on dP_net, 
        where 0=NO masks; 1,2... are mask labels
    dP_net: ND-array (float) 
        ND flows where dP_net.shape[1:] = maski.shape

    Returns
    ------------
    flow_errors: float array with length maski.max()
        mean squared error between predicted flows and flows from masks
    dP_masks: ND-array (float)
        ND flows produced from the predicted masks
    
    """
    if dP_net.shape[1:] != maski.shape:
        omnipose_logger.info('ERROR: net flow is not same size as predicted masks')
        return

    # ensure unique masks
    # maski = np.reshape(np.unique(maski.astype(np.float32), return_inverse=True)[1], maski.shape)
    fastremap.renumber(maski,in_place=True)

    # flows predicted from estimated masks and boundaries
    idx = -1 # flows are the last thing returned now
    dim = maski.ndim
    dP_masks = masks_to_flows(maski, dim=dim, coords=coords, affinity_graph=affinity_graph, 
                              use_gpu=use_gpu, device=device, omni=omni)[idx].cpu().numpy() ##### dynamics.masks_to_flows
    # difference between predicted flows vs mask flows
    flow_errors = np.zeros(maski.max())
    
    for i in range(dP_masks.shape[0]):
        flow_errors += mean((dP_masks[i] - dP_net[i]/5.)**2, maski, #the /5 is to compensate for the *5 we do for training
                            index=np.arange(1, maski.max()+1))
        
    return flow_errors, dP_masks

def scale_to_tenths(x):
    eps = 1e-12  # Small epsilon to prevent log issues
    scale_factor = 10 ** (-torch.floor(torch.log10(torch.abs(x) + eps)) - 1)
    return x * scale_factor
    
def scale_to_tenths(x, max_gain=10):
    eps = 1e-12
    sf  = 10 ** (-torch.floor(torch.log10(torch.abs(x)+eps)) - 1)
    sf  = torch.clamp(sf, 1/max_gain, max_gain)   # cap between 0.1× and 10×
    return x * sf

def fill_holes_and_remove_small_masks(masks, min_size=None, max_size=None, hole_size=3, dim=2):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)
    
    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes
    
    Parameters
    ----------------
    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    min_size: int (optional, default 3**dim)
        minimum number of pixels per mask (exclusive), can turn off with -1
    max_size: int (optional, default None)
        maximum number of pixels per mask (exclusive)
    hole_size: int (optional, default 3)
        holes bigger than this are NOT filled
    dim: int (optional, default 2)
        dimension of the masks

    Returns
    ---------------
    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    # Min size taken to be an N-cube in ND (9 pixels, 27 voxels, ...) if not specified 
    if min_size is None:
        min_size = 3**dim # N cube 

    # if masks.ndim==2 or dim>2:
        # formatting to integer is critical
        # need to test how it does with 3D
    masks = ncolor.format_labels(masks, min_area=min_size)#, clean=True)
    fill_holes = hole_size>0 # toggle off hole filling by setting hole size to 0
    
    slices = find_objects(masks)
    j = 0
    for i,slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i+1)
            npix = msk.sum()

            too_small = npix < min_size
            too_big = False if max_size is None else npix > max_size

            if (min_size > 0) and (too_small or too_big):
                masks[slc][msk] = 0
            elif fill_holes:   
                hsz = np.count_nonzero(msk)*hole_size/100 #turn hole size into percentage
                #eventually the boundary output should be used to properly exclude real holes vs label gaps
                # for not I just toggle it off 
                if SKIMAGE_ENABLED: # Omnipose version (passes 2D tests)
                    pad = 1
                    unpad = tuple([slice(pad,-pad)]*dim) 
                    # [pluriseg] max_size to supress warnings
                    padmsk = remove_small_holes(np.pad(msk,pad,mode='constant'),max_size=hsz)
                    msk = padmsk[unpad]
                else: #Cellpose version
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = (j+1)
                j+=1
    return masks

def divergence_torch(y):
    """
    Divergence for a batched D-vector field stored as ``(B, D, *spatial)``.

    * **GPU / MPS** -> use a single call to ``torch.gradient`` (fast, parallel).
    * **CPU**       -> compute only the gradients actually needed, one component
      at a time, to avoid the unnecessary D^2 work that the vectorised call
      performs on the CPU.

    Returns
    -------
    div : torch.Tensor
        Shape ``(B, *spatial)`` - divergence of ``y``.
    """
    B, D, *spatial = y.shape

    # Guard against degenerate spatial dims that are too small for gradient
    # computation (edge_order >= 1). When any spatial dimension is < 2 the
    # divergence is undefined; return zeros to avoid runtime errors and keep
    # CPU/GPU parity checks focused on the network output.
    if any(s < 2 for s in spatial):
        return torch.zeros((B, *spatial), dtype=y.dtype, device=y.device)
    if y.device.type == 'cpu':
        # Allocate output once and fill in-place
        div = torch.zeros((B, *spatial), dtype=y.dtype, device=y.device)
        for d in range(D):                          # loop over spatial axes
            comp   = y[:, d]                        # (B, *spatial)
            axis   = d + 1                          # 0=batch, 1=first spatial …
            grad_d = torch.gradient(comp, dim=axis)[0]
            div   += grad_d
        return div
    else:
        spatial_axes = list(range(-len(spatial), 0))      # e.g. [-2,-1] in 2-D.
        grads = torch.gradient(y, dim=spatial_axes)       # tuple length == D
        div = sum(g[:, d] for d, g in enumerate(grads))   # pick aligned comps
        return div

