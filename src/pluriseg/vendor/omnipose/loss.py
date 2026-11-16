import torch

from .utils import torch_norm


class BatchMeanMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction='none')

    def forward(self, pred, target, weight=None):
        """
        Compute mean‑squared error averaged per sample, then across batch.
        Optionally apply element‑wise weighting.
        """
        per_elem = self.mse(pred, target)
        if weight is not None:
            per_elem = per_elem * weight
        per_sample = per_elem.reshape(per_elem.size(0), -1).mean(dim=1)
        return per_sample.mean()


class BatchMeanBSE(torch.nn.Module):
    """
    Same idea as BatchMeanMSE but using binary cross‑entropy.
    Computes BCE loss per element → average over each sample →
    average those means across the batch.
    """
    def __init__(self):
        super().__init__()
        # keep element‑wise losses so we can average the way we want
        self.bce = torch.nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        per_elem = self.bce(pred, target)              # shape (B, …)
        per_sample = per_elem.reshape(per_elem.size(0), -1).mean(dim=1)
        return per_sample.mean()
      

class WeightedMSELoss(torch.nn.Module):
    """
    Weighted MSE implemented via BatchMeanMSE for consistent reduction.
    """
    def __init__(self):
        super().__init__()
        self.base = BatchMeanMSE()

    def forward(self, pred, target, weight):
        return self.base(pred, target, weight)


            

class SineSquaredLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,y,w):
        eps = 1e-12
        magX = torch_norm(x,dim=1)
        magY = torch_norm(y,dim=1)
        denom = torch.multiply(magX,magY)
        dot = torch.sum(torch.stack([x[:,k]*y[:,k] for k in range(x.shape[1])],dim=1),dim=1)
        
        # need to handle zero denominator, so if either or both are zero, cos is 1 so loss is minimum (0)
        # so that handles transitions from boundary to background just fine 
        cos = torch.where(denom>eps,dot/(denom+eps),1) 
        return torch.mean((1-cos**2)*w)        
        # return torch.where(mask,(1-cos**2)*w,torch.nan).nanmean() # possible alternative 

        
class NormLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,y,Y,w):
        return torch.mean(torch.square(torch_norm(y,dim=1)-torch_norm(Y,dim=1))*w)/25
        # return torch.mean(torch.square(torch_norm(y,dim=1)-torch_norm(Y,dim=1))*w)/25
        
        # return torch.nn.functional.binary_cross_entropy_with_logits(torch_norm(y,dim=1)/5,torch_norm(Y,dim=1)/5)
        # return torch.nn.functional.l1_loss(torch_norm(y,dim=1)/5,torch_norm(Y,dim=1)/5)

# more efficient combination of the two loss functions, fewer calls to torch_norm 
class SSL_Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.MSE = BatchMeanMSE()
        self.WMSE = WeightedMSELoss()
        
    def forward(self,x,y,dist,w,bd): # y is GT, x is predicted 
        eps = 1e-12
        magX = torch_norm(x,dim=1)
        magY = torch_norm(y,dim=1)
        denom = torch.multiply(magX,magY)
        # dot = torch.sum(torch.stack([x[:,k]*y[:,k] for k in range(x.shape[1])],dim=1),dim=1)
        dot = (x * y).sum(dim=1)
        # cossq = torch.where(denom>eps,dot/(denom+eps),1)**2 #golden 
        # cossq = torch.where(denom>0,dot/(denom+eps),1)**2 
       
       
       # this was in use for a while, jan 2025 
        mask = dist>0
        cossq = torch.where(mask,dot/(denom+eps),1)**2 #experiment to limit to where cell pixels are predicted only 
        
        
        
        # potential substitute for norm loss? 
#         normdiff = torch.where(magX>eps,1-(magY/(magX)),magX)
#         wsum = torch.sum(w)
#         return torch.sum((1-cos**2)*w)/wsum,torch.sum((normdiff**2)*w)/wsum

        # maybe should scale norm loss by cos**2? Only care if mag is right if direction is right 
        
        # return torch.mean((1-cossq)*w),torch.mean(torch.square(magX-magY)*w)/25 # golden version
        # SSL = torch.mean((1-cossq)*w) # golden 
        
        
        # NL = torch.mean(torch.abs(magX-magY)*w)/5 # pretty good, an improvement? 
        # NL = torch.mean(torch.square(magX-magY))/25 # most basic version
        # NL = torch.mean(torch.square(magX-magY)*bd/5) # just boundary - works pretty well, one of the better ones
        
        # this version is doing extremely well so far, but maybe a bit too much weight overall
        # NL = torch.mean(torch.square(mask*magY-magX)*bd) #/5
        # SSL = torch.mean((1-cossq)*bd)*10 # upweight here for experiment
        
        # reduce these terms by a bit to avoid some artifacts, decent 
        # NL = torch.mean(torch.square(mask*magY-magX)*bd)/2 
#         SSL = torch.mean((1-cossq)*bd)*6 # upweight here for experiment
        
        # golden
        # reduce further to get performance back maybe? This worked very, very well 
        # NL = (torch.mean(torch.square(mask*magY-magX)*bd)/2+torch.mean(torch.square(magX-magY))/25)/2
        # SSL = torch.mean((1-cossq)*bd)*4
        
        # this was in use for a long while, january 2025
        # s = torch.square(magY-magX)
        # NL = (torch.mean(s*bd)/2+torch.mean(s)/25)/2
        # sel = torch.where(bd)
        # SSL = torch.mean((1-cossq[sel]))
        
        # if we predict the right direction, we should predict the right magnitude
        # thus cos**2 should be 1, leading to a minumum if  magX/5 is close to 1
        # if we predict the wrong direction, cos**2 should be 0 this will push magX/5 to be 0
        # NL = torch.mean(torch.square(magX/5-cossq)) 
        
        # weighting the normloss above high makes things 1 everywhere, not good
        # NL = torch.mean(torch.square(magX/5-(magX>0).float())) + 
        NL = self.MSE(magX, magY)                           # unweighted magnitude error
        SSL = self.WMSE(cossq, torch.ones_like(cossq), w)   # weighted orientation error
        # sel = torch.where(bd)
        # SSL = torch.mean((1-cossq[sel]))
        
        
        # see if SSL should be weighted instead of purely masked by bd 
        # NL = (torch.mean(torch.square(mask*magY-magX)*w)/2+torch.mean(torch.square(magX-magY))/25)/2
        # SSL = torch.mean((1-cossq)*w)*4
        
        
        # I think that I can simplify the NL a lot by just making the appropriate weight matrix
        # might as well apply it to SSL instead of just bd, but then the overall factor needs to go back up
        # turns out this was not a good idea...
        # weight = (mask+1)/25+bd/2
        # NL = torch.mean(torch.square(mask*magY-magX)*weight)
        # # SSL = torch.mean((1-cossq)*weight)*8
        # SSL = torch.mean((1-cossq)*bd)*4 # not sure if applying it to SSL was good        
        
        
        # loss on the gradient of the norm - seems to work ok for making the field the right magnitude  
        # dim = x.shape[1]
        # dims = [k for k in range(-dim,0)]
        # dx = torch.stack(torch.gradient(magX,dim=dims))
        # dy = torch.stack(torch.gradient(magY,dim=dims))
        # NL += torch.mean(torch.square(dx-dy))/5
        
        # soft dice
        # eps = 1e-5
        # A = torch.sum(magX/5)
        # B = torch.sum(magY/5)
        # C = torch.sum(magX*magY/25)
        # NL = 1-(2*C+eps)/(A+B+eps)
        
        # attempt at polynomial loss 
        # d = magX-magY
        # NL =  torch.mean(torch.square(d*(d-5)*(d+5)))/(5**6)
        # NL =  torch.mean(torch.square((magX-magY)*(magX-2*magY)*magX))
        
        # maybe norm loss should only act to push the predictions avobe cutoff to GT, below cutoff to 0
        # NL = torch.mean(torch.where(magY>1,torch.square(magX-magY)*w/25,magY)) # experimental version 
        
        return SSL, NL
        
        # return torch.mean((1-cossq)*w),torch.mean(torch.square(magX-magY)*cossq)/25
        
        # return torch.mean(w*((1-cos**2)+(1/25)*torch.square(magX-magY))),torch.zeros(1,device=w.device)
        
# maybe should blend between MSE and SSL
# SSL counts at the border most, MSE counts most at the center 
class SSL_Norm_MSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x,y,w,dist):
        eps = 1e-12
        magX = torch_norm(x,dim=1)
        magY = torch_norm(y,dim=1)
        denom = torch.multiply(magX,magY)
        dot = torch.sum(torch.stack([x[:,k]*y[:,k] for k in range(x.shape[1])],dim=1),dim=1)
        cos = torch.where(denom>eps,dot/(denom+eps),1)
        err = torch.square((x-y)/5.).sum(dim=1)
        
        w3 = torch.clip(dist,0.5,2)/2
        w2 = 1.-w3
        cos_weighted = torch.sum((1-cos**2)*w2)/torch.sum(w2)
        mse_weighted = torch.sum(err*w3)/torch.sum(w3)
        
        cos_weighted = torch.mean((1-cos**2)*err)
        mse_weighted = 0
        
        # or maybe MSE / cos?
        cos_weighted = torch.mean(err / (cos**2+1)) # terrible, causes border merging 
        
        return mse_weighted, cos_weighted, torch.mean(torch.square(magX-magY)*w)/25
        # return torch.mean(w*((1-cos**2)+(1/25)*(magX-magY)**2))
        
        

# class DerivativeLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self,y,Y,w,mask):
#         # y is awlays nbatch x dim x shape, shape is Ly x Lx, Lt x Ly x Lx, or Lz x Ly x Lx. 
#         # so y[0] grabs one example
#         # axes = [k for k in range(len(y[0]))]     
#         # print('shape',y.shape,y[0].shape)
#         dim = y.shape[1]
#         dims = [k for k in range(-dim,0)]
#         # print('dims',dim,dims)
#         dy = torch.stack(torch.gradient(y,dim=dims))
#         dY = torch.stack(torch.gradient(Y,dim=dims))
#         sel = torch.where(mask) # read that masked selection could be causing this 
#         return torch.mean(torch.sum(torch.square((dy-dY)/5.),axis=0)[sel]*w[sel])    



class DerivativeLoss(torch.nn.Module):
    """
    Gradient‑domain loss rewritten to use the BatchMean/Weighted MSE
    infrastructure for consistent per‑sample reduction.

    Computes the mean‑squared error between spatial gradients of the
    prediction `y` and ground‑truth `Y`.  Results are averaged per‑sample,
    then across the batch via `WeightedMSELoss`, with optional pixel‑wise
    weights `w` and a binary mask `mask` defining valid regions.

    Mathematically equivalent to the original implementation:

        mean( ((∇y − ∇Y)/5)² * w )[mask]
    """
    def __init__(self):
        super().__init__()
        self.WMSE = WeightedMSELoss()

    def forward(self, y, Y, w, mask):
        # Number of spatial dimensions inferred the same way as before
        dim = y.shape[1]
        spatial_axes = [k for k in range(-dim, 0)]

        # Gradients along each spatial axis → stack then bring batch dim first
        dy = torch.stack(torch.gradient(y, dim=spatial_axes)).transpose(0, 1)
        dY = torch.stack(torch.gradient(Y, dim=spatial_axes)).transpose(0, 1)

        # Combine weights with mask; broadcast to gradient tensor shape
        weight = w * mask                          # (B,1,…)
        while weight.ndim < dy.ndim:
            weight = weight.unsqueeze(1)           # prepend dims
        weight = weight.expand_as(dy)              # (B,dim,C,…)

        # Weighted MSE, scaled by 1/25 to match (⋅/5)² factor
        return self.WMSE(dy, dY, weight) / 25
    

# it will probably be better just to do the gradient for the whole image....
# i.e use a constant affinity graph 
# class EikonalLoss(torch.nn.Module):
#     def __init__(self,device,dim):
#         self.device = device
#         steps,inds,idx,fact,sign = utils.kernel_setup(dim)
#         self.dim = torch.tensor(dim,device=device)
#         self.idx = torch.tensor(idx)
#         self.fact = torch.tensor(fact)
#         self.steps = torch.tensor(steps,device=device)        
#         self.inds = tuple([torch.tensor(i) for i in inds])
            
        
#         super().__init__()

#     def forward(self,dist_pred,flow_gt,mask):
#     neighbors = utils.get_neighbors(coords,steps,d,shape,edges) # shape (d,3**d,npix)   
    
    
#         isneigh = torch.tensor(affinity_graph,device=device,dtype=torch.bool) 
#         neigh_inds = torch.tensor(neigh_inds,device=device)
#         central_inds = torch.tensor(central_inds,device=device,dtype=torch.long)
        
#         # get the gradient of the predicted distance field
#         T = dist_pred[mask]


class CorrelationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,y):
        vx = x - torch.mean(x, dim=0)
        vy = y - torch.mean(y, dim=0)
        num = torch.sum(vx * vy, dim=0)
        denom = torch.sum(vx**2, dim=0) * torch.sum(vy**2, dim=0)
        cost = torch.where(denom>0,num/torch.sqrt(denom),-1)
        return -torch.mean(cost) 
        
        
# dist_loss = self.TruncatedMSELoss(dt,dist)  # less glowy edges, /10 maybe not dominant enough.. eventually broke down 
class TruncatedMSELoss(torch.nn.Module):
    def __init__(self, t=5.0):
        super().__init__()
        self.t = t

    def forward(self, pred, target):
        """
        pred, target: shape (...), float tensors
        """
        SE = torch.square(pred - target)
        # Where error is small, use MSE
        # Where error is bigger than threshold, use a constant penalty
        loss = torch.where(SE < self.t, SE, self.t)
        return loss.mean()
        
        

class MeanAdjustedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MeanAdjustedMSELoss, self).__init__()

    def forward(self, pred, target):
        """
        Compute the mean-adjusted mean squared error.

        Args:
            pred (torch.Tensor): Predicted distance field.
            target (torch.Tensor): Ground truth distance field.

        Returns:
            torch.Tensor: Mean-adjusted MSE loss.
        """
        # Calculate the mean error (bias)
        mean_error = torch.mean(pred - target)

        # Adjust predictions to remove the mean error
        adjusted_pred = pred - mean_error

        # Compute the mean squared error on the adjusted predictions
        loss = torch.mean((adjusted_pred - target) ** 2)

        return loss
        

import torch


class GradNormLoss(torch.nn.Module):
    def __init__(self, num_losses, device, alpha=0.12):
        super().__init__()
        self.alpha = alpha
        self.loss_weights = torch.nn.Parameter(torch.ones(num_losses, device=device, dtype=torch.float32))

    def forward(self, losses, shared_params):
        """
        Compute GradNorm-weighted loss efficiently.
        """

        # Skip empty loss list
        if len(losses) == 0:
            raise RuntimeError("All loss terms are zero or don't require gradients.")

        # Compute weighted loss
        weighted_losses = self.loss_weights[:len(losses)] * torch.stack(losses)
        total_loss = weighted_losses.sum()

        # Compute gradients for all losses **in one backward pass**
        grads = torch.autograd.grad(total_loss, shared_params, retain_graph=True, create_graph=False)

        # Compute gradient norms per loss term (mean over parameters)
        grad_norms = []
        for loss, grad in zip(losses, grads):
            if grad is not None:
                grad_norms.append(torch.norm(grad))

        grad_norms = torch.stack(grad_norms)#.to(self.device)

        # Compute mean gradient magnitude
        mean_grad = grad_norms.mean().detach()

        # Compute scaling targets
        target_scales = (grad_norms / mean_grad) ** self.alpha
        target_scales = target_scales.detach()

        # Compute loss for updating weights
        weight_loss = torch.sum(torch.abs(self.loss_weights[:len(losses)] * grad_norms - target_scales * mean_grad))

        # Efficiently compute weight updates
        weight_grads = torch.autograd.grad(weight_loss, self.loss_weights[:len(losses)], allow_unused=True)[0]
        self.loss_weights.grad = weight_grads  # Directly set gradients

        return total_loss
