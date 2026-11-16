import numpy as np

import fastremap

import functools

import torch
from sklearn.utils.extmath import cartesian
import math

def get_module(x):
    if isinstance(x, (np.ndarray, tuple, int, float)) or np.isscalar(x):
        return np
    elif torch.is_tensor(x):
        return torch
    else:
        raise ValueError("Input must be a numpy array, a tuple, a torch tensor, an integer, or a float")

def safe_divide(num, den, cutoff=0):
    """ Division ignoring zeros and NaNs in the denominator.""" 
    module = get_module(num)
    valid_den = (den > cutoff) & module.isfinite(den) #isfinite catches both nan and inf

    if module == np:
        r = num.astype(np.float32, copy=False)
        r = np.divide(r, den, out=np.zeros_like(r), where=valid_den)
    elif module == torch:
        r = num.float()
        den = den.float()
        small_val = torch.finfo(den.dtype).tiny  # smallest positive representable number
        safe_den = torch.where(valid_den, den, small_val)
        r = torch.div(r, safe_den)
    else:
        raise TypeError("num must be a numpy array or a PyTorch tensor")

    return r




def normalize_field(mu,use_torch=False,cutoff=0):
    """ normalize all nonzero field vectors to magnitude 1
    
    Parameters
    ----------
    mu: ndarray, float
        Component array of lenth N by L1 by L2 by ... by LN. 
    
    Returns
    --------------
    normalized component array of identical size. 
    """
    if use_torch:
        mag = torch_norm(mu,dim=0)
        # out = torch.zeros_like(mu)
        # sel = mag>cutoff
        # out[:,sel] = torch.div(mu[:,sel],mag[sel])
        # return out
        # return torch.where(mag>cutoff,mu/mag,torch.zeros_like(mu))
        return torch.where(mag>cutoff,mu/mag,mu)
        
    else:
        mag = np.sqrt(np.nansum(mu**2,axis=0))
        return safe_divide(mu,mag,cutoff)
    
# @torch.jit.script
def torch_norm(a,dim=0,keepdim=False):
    """ Wrapper for torch.linalg.norm to handle ARM architecture. """
    # if ARM: 
    #     #torch.linalg.norm not implemented on MPS yet
    #     # this is the fastest I have tested but still slow in comparison 
    #     return a.square().sum(dim=dim,keepdim=keepdim).sqrt()
    # else:
    #     return torch.linalg.norm(a,dim=dim,keepdim=keepdim)
        
        
        # Compute squared norm with a minimal number of intermediate tensors.
    norm_sq = (a * a).sum(dim=dim, keepdim=keepdim)
    # Use the in-place sqrt when possible (if not tracking gradients).
    return norm_sq.sqrt_() if not norm_sq.requires_grad else norm_sq.clamp_min(1e-12).sqrt()   # [pluriseg] clamp if norm_sq == 0 (SqrtBackward 0.5/sqrt(x) NaN)
    
    # in the future when MPS supports it, just use try catch and print a warning to upgrade torch 

def normalize99(Y, lower=0.01, upper=99.99, contrast_limits=None, dim=None):
    """ normalize array/tensor so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile 
    Upper and lower percentile ranges configurable. 
    
    Parameters
    ----------
    Y: ndarray/tensor, float
        Input array/tensor. 
    upper: float
        upper percentile above which pixels are sent to 1.0
    
    lower: float
        lower percentile below which pixels are sent to 0.0
        
    contrast_limits: list, float (optional, override computation)
        list of two floats, lower and upper contrast limits
    
    Returns
    --------------
    normalized array/tensor with a minimum of 0 and maximum of 1
    
    """
    module = get_module(Y)
    
    if contrast_limits is None:
    
        quantiles = np.array([lower, upper]) / 100
        if module == torch:
            quantiles = torch.tensor(quantiles, dtype=Y.dtype, device=Y.device)
   
        if dim is not None:
            # Reshape Y into a 2D tensor for quantile computation
            Y_flattened = Y.reshape(Y.shape[dim], -1)

            lower_val, upper_val = module.quantile(Y_flattened, quantiles, axis=-1)        
            
            # Reshape back into original shape for broadcasting
            if dim == 0:
                lower_val = lower_val.reshape(Y.shape[dim], *([1] * (len(Y.shape) - 1)))
                upper_val = upper_val.reshape(Y.shape[dim], *([1] * (len(Y.shape) - 1)))
            else:
                lower_val = lower_val.reshape(*Y.shape[:dim], *([1] * (len(Y.shape) - dim - 1)))
                upper_val = upper_val.reshape(*Y.shape[:dim], *([1] * (len(Y.shape) - dim - 1)))
        else:
            # lower_val, upper_val = module.quantile(Y, quantiles)
            try:
                lower_val, upper_val = module.quantile(Y, quantiles)
            except RuntimeError:
                lower_val, upper_val = auto_chunked_quantile(Y, quantiles)

    else:
        if module == np:
            contrast_limits = np.array(contrast_limits)
        elif module == torch:
            contrast_limits = torch.tensor(contrast_limits)

        lower_val, upper_val = contrast_limits
        
    
    # Y = module.clip(Y, lower_val, upper_val) # is this needed? 
    # Y -= lower_val
    # Y /= (upper_val - lower_val)
    
    # return Y
    # return (Y-lower_val)/(upper_val-lower_val)
    # return module.clip((Y-lower_val)/(upper_val-lower_val),0,1)
    # return module.clip((Y-lower_val)/(upper_val-lower_val),0,1)
    # in this case, since lower_val is not the absolute minimum, but the lowerr quanitle, 
    # Y-lower_val can be less than zero. Likewise for the upward scalimg being slightly >1. 
    return module.clip(safe_divide(Y-lower_val,upper_val-lower_val),0,1)




def auto_chunked_quantile(tensor, q):
    # Determine the maximum number of elements that can be handled by PyTorch's quantile function
    max_elements = 16e6 - 1  

    # Determine the number of elements in the tensor
    num_elements = tensor.nelement()

    # Determine the chunk size
    chunk_size = math.ceil(num_elements / max_elements)
    
    # Split the tensor into chunks
    chunks = torch.chunk(tensor, chunk_size)

    # Compute the quantile for each chunk
    return torch.stack([torch.quantile(chunk, q) for chunk in chunks]).mean(dim=0)

def get_neighbors(coords, steps, dim, shape, edges=None, pad=0):
    """
    Get the neighbor coordinates for each pixel in `coords` for each offset in `steps`.
    Out-of-bounds neighbors get clipped or replaced with original coords (depending on `edges`).
    """
    if edges is None:
        edges = [np.array([-1+pad, s-pad]) for s in shape]

    npix = coords[0].shape[-1]
    nsteps = len(steps)  # e.g. 8 (2D) or 26 (3D)
    
    # neighbors.shape = (dim, nsteps, npix)
    neighbors = np.empty((dim, nsteps, npix), dtype=np.int64)
    
    # Precompute edge_masks for each dimension
    edge_masks = []
    for d in range(dim):
        mask = np.zeros(shape[d], dtype=bool)
        valid_edges = edges[d][(edges[d] >= 0) & (edges[d] < shape[d])]
        mask[valid_edges] = True
        edge_masks.append(mask)
    
    # For each dimension d, process each step offset one by one
    for d in range(dim):
        current_mask = edge_masks[d]
        size_d = shape[d]
        
        for n, step_d in enumerate(steps[:, d]):
            # X is just 1D, shape: (npix,)
            X = coords[d] + step_d
            
            # clip in-place (avoid creating a second large array)
            # You can do: np.clip(X, 0, size_d - 1, out=X), but that modifies coords[d]! 
            # so we copy first:
            Xc = X.copy()
            np.clip(Xc, 0, size_d - 1, out=Xc)
            
            # shift also clipped in place, if you need it:
            Xs = X + step_d
            np.clip(Xs, 0, size_d - 1, out=Xs)
            
            # Out-of-bounds condition: 
            # "oob if current_mask[Xc] == True and current_mask[Xs] == False"
            # We'll do it only where Xc is within [0, size_d -1].
            # NB: Xc is an array of indices, we can check current_mask at those indices:
            oob = np.logical_and(current_mask[Xc], ~current_mask[Xs])
            
            # Now pick either coords[d] or the clipped coordinate.
            # Instead of np.where(...), we can do in-place assignment:
            out = Xc  # start with clipped
            out[oob] = coords[d][oob]  # revert out-of-bounds neighbors

            neighbors[d, n] = out  # store in final array

    return neighbors

def get_neigh_inds(neighbors,coords,shape,background_reflect=False):
    """
    For L pixels and S steps, find the neighboring pixel indexes 
    0,1,...,L for each step. Background index is -1. Returns:
    
    
    Parameters
    ----------
    neighbors: ND array, int
        ndim x nsteps x npix array of neighbor coordinates
    
    coords: tuple, int
        coordinates of nonzero pixels, <ndim>x<npix>

    shape: tuple, int
        shape of the image array

    Returns
    -------
    indexes: 1D array
        list of pixel indexes 0,1,...L-1
        
    neigh_inds: 2D array
        SxL array corresponding to affinity graph
    
    ind_matrix: ND array
        indexes inserted into the ND image volume
    """
    neighbors = tuple(neighbors) # just in case I pass it as ndarray

    npix = neighbors[0].shape[-1]
    indexes = np.arange(npix)
    ind_matrix = -np.ones(shape,int)
    
    ind_matrix[tuple(coords)] = indexes
    neigh_inds = ind_matrix[neighbors]
    
    # If needed, we can do a similar thing I do at boundaries and make neighbor
    # references to background redirect back to the edge pixel. However, this should 
    # not be default, since I rely on accurate neighbor indices later to test for background
    # So, probably better to do this sort of thing while contructing the affinity graph itself 
    if background_reflect:
        oob = np.nonzero(neigh_inds==-1) # 2 x nbad , pos 0 is the 0-step inds and pos 1 is the npix inds 
        neigh_inds[oob] = indexes[oob[1]] # reflect back to itself 
        ind_matrix[neighbors] = neigh_inds # update ind matrix as well

        # should I also update neighbor coordinate array? No, that's more fixed. 
        # index points to the correct coordinate. 
    
    # not sure if -1 is general enough, probbaly should be since other adjacent masks will be unlinked
    # can test it by adding some padding to the concatenation...
    
    # also, the reflections should be happening at edges of the image, but it is not? 
    
    return indexes, neigh_inds, ind_matrix



@functools.lru_cache(maxsize=None) 
def get_steps(dim):
    """
    Get a symmetrical list of all 3**N points in a hypercube represented
    by a list of all possible sequences of -1, 0, and 1 in ND.

    Examples
    --------
    .. code-block:: text

        1D: [[-1], [0], [1]]
        2D: [[-1, -1],
             [-1,  0],
             [-1,  1],
             [ 0, -1],
             [ 0,  0],
             [ 0,  1],
             [ 1, -1],
             [ 1,  0],
             [ 1,  1]]
    
    The opposite pixel at index i is always found at index -(i+1). The number
    of possible face, edge, vertex, etc. connections grows exponentially with
    dimension: 3 steps in 1D, 9 steps in 3D, 3**N in ND. 
    """
    neigh = [[-1,0,1] for i in range(dim)]
    steps = cartesian(neigh) # all the possible step sequences in ND
    
    # a new function I learned about, np.ndindex, could be used here instead, 
    # np.stack([s for s in np.ndindex(*(3,) * ndim)])-1, 
    # but it runs in microseconds rather than nanoseconds... 
    return steps

# @functools.lru_cache(maxsize=None)
def steps_to_indices(steps):
    """
    Get indices of the hupercubes sharing m-faces on the central n-cube. These
    are sorted by the connectivity (by center, face, edge, vertex, ...). I.e.,
    the central point index is first, followed by cardinal directions, ordinals,
    and so on. 
    """
     # each kind of m-face can be categorized by the number of steps to get there
    sign = np.sum(np.abs(steps),axis=1)
    
    # we want to bin them into groups 
    # E.g., in 2D: [4] (central), [1,3,5,7] (cardinal), [0,2,6,8] (ordinal)
    uniq = fastremap.unique(sign)
    inds = [np.where(sign==i)[0] for i in uniq] 
    
    # weighting factor for each hypercube group (distance from central point)
    fact = np.sqrt(uniq) 
    return inds, fact, sign

# [steps[:idx],steps[idx+1:]] can give the other steps 
@functools.lru_cache(maxsize=None) 
def kernel_setup(dim):
    """
    Get relevant kernel information for the hypercube of interest. 
    Calls get_steps(), steps_to_indices(). 
    
    Parameters
    ----------

    dim: int
        dimension (usually 2 or 3, but can be any positive integer)
    
    Returns
    -------
    steps: ndarray, int 
        list of steps to each kernal point
        see get_steps()
        
    idx: int
        index of the central point within the step list
        this is always (3**dim)//2
        
    inds: ndarray, int
        list of kernel points sorted by type
        see  steps_to_indices()
    
    fact: float
        list of face/edge/vertex/... distances 
        see steps_to_indices()
        
    sign: 1D array, int
        signature distinguishing each kind of m-face via the number of steps
        see steps_to_indices()

    
    """
    steps = get_steps(dim)
    inds, fact, sign = steps_to_indices(steps)
    idx = inds[0][0] # the central point is always first 
    return steps,inds,idx,fact,sign


from collections import defaultdict
def get_supporting_inds(steps):
    """
    For each step 'v', find all pairs (i, j) such that steps[i] + steps[j] == steps[v],
    excluding the center index.

    Steps shape: (S, d), with a 'center_index' = S//2 by default.
    """
    steps = np.array(steps, copy=False)
    S, d = steps.shape
    center_index = S // 2
    
    # Create a mask that excludes the center
    mask = np.arange(S) != center_index
    # Steps without the center
    steps_nocenter = steps[mask]  # shape: (S-1, d)
    orig_indices = np.nonzero(mask)[0]  # original indices in [0..S-1], skipping center
    
    N = S - 1  # number of non-center steps
    
    # Pairwise sums: shape (N, N, d)
    pair_sums = steps_nocenter[:, None, :] + steps_nocenter[None, :, :]
    # Flatten to (N*N, d)
    pair_sums_2d = pair_sums.reshape(-1, d)

    # We'll keep track of which (i,j) generated each sum
    # i_list, j_list are each of length N*N
    i_list = np.repeat(orig_indices, N)
    j_list = np.tile(orig_indices, N)

    # Build a dictionary: sum_map[ tuple_of_coords ] -> list of (i,j)
    sum_map = defaultdict(list)
    for k in range(N * N):
        key = tuple(pair_sums_2d[k])
        sum_map[key].append((i_list[k], j_list[k]))

    # Now for each v != center, look up tuple(steps[v]) in sum_map
    pairs = {}
    for v in range(S):
        if v == center_index:
            continue
        key = tuple(steps[v])
        pairs[v] = sum_map.get(key, [])

    return pairs

