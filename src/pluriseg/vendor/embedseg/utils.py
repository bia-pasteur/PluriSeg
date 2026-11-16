
import numpy as np
import torch
from skimage.feature import peak_local_max # pylint: disable=E0611





class Cluster:
    """
    A class used to cluster pixel embeddings in 2D

    Attributes
    ----------
    xym :  float (2, W, D)
            pixel coordinates of tile /grid

    one_hot : bool
            Should be set to True, if the GT label masks are present in a
            one-hot encoded fashion

    grid_x: int
            Length (width) of grid / tile

    grid_y: int
            Height of grid / tile

    pixel_x: float
            if grid_x = 1000 and pixel_x = 1.0, then the pixel spacing
            along the x direction is pixel_x/(grid_x-1) = 1/999
    pixel_y: float
            if grid_y = 1000 and pixel_y = 1.0, then the pixel spacing
            along the y direction is pixel_y/(grid_y-1) = 1/999


    Methods
    -------
    __init__: Initializes an object of class `Cluster_3d`

    cluster_with_gt: use the predicted spatial embeddings
                from all pixels belonging to the GT label mask
                to identify the predicted cluster
                (used during training and validation)

    cluster:    use the  predicted spatial embeddings
                from all pixels in the test image.
                Employs `fg_thresh` and `seed_thresh`
    cluster_local_maxima: use the  predicted spatial embeddings
                from all pixels in the test image.
                Employs only `fg_thresh`
    """

    def __init__(self, grid_y, grid_x, pixel_y, pixel_x, device, one_hot=False):
        """
        Parameters
        ----------
        xym :  float (2, W, D)
                 pixel coordinates of tile /grid

         one_hot : bool
                 Should be set to True, if the GT label masks are present
                 in a one-hot encoded fashion

         grid_x: int
                 Length (width) of grid / tile

         grid_y: int
                 Height of grid / tile

         pixel_x: float
                 if grid_x = 1000 and pixel_x = 1.0, then the pixel spacing
                 along the x direction is pixel_x/(grid_x-1) = 1/999
         pixel_y: float
                 if grid_y = 1000 and pixel_y = 1.0, then the pixel spacing
                 along the y direction is pixel_y/(grid_y-1) = 1/999

        """
        xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
        ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
        xym = torch.cat((xm, ym), 0)

        self.device = device
        self.xym = xym.to(self.device)
        self.one_hot = one_hot
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

    def cluster_with_gt(
        self,
        prediction,
        gt_instance,
        n_sigma=2,
    ):
        """
        Parameters
        ----------
        prediction :  PyTorch Tensor
                Model Prediction (5, H, W)
        gt_instance : PyTorch Tensor
                Ground Truth Instance Segmentation Label Map

        n_sigma: int, default = 2
                Number of dimensions in Raw Image
        Returns
        ----------
        instance: PyTorch Tensor (H, W)
                instance segmentation
        """

        height, width = prediction.size(1), prediction.size(2)

        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2 : 2 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(height, width).short().to(self.device)
        if self.one_hot:
            unique_instances = torch.arange(gt_instance.size(0))
        else:
            unique_instances = gt_instance.unique()
            unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            if self.one_hot:
                mask = gt_instance[id].eq(1).view(1, height, width)
            else:
                mask = gt_instance.eq(id).view(1, height, width)

            center = (
                spatial_emb[mask.expand_as(spatial_emb)]
                .view(2, -1)
                .mean(1)
                .view(2, 1, 1)
            )  # 2 x 1 x 1

            s = (
                sigma[mask.expand_as(sigma)]
                .view(n_sigma, -1)
                .mean(1)
                .view(n_sigma, 1, 1)
            )

            s = torch.exp(s * 10)  # n_sigma x 1 x 1 #
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))
            proposal = dist > 0.5
            if self.one_hot:
                instance_map[proposal] = id.item() + 1
            else:
                instance_map[proposal] = id.item()

        return instance_map

    def cluster_local_maxima(
        self,
        prediction,
        n_sigma=2,
        fg_thresh=0.5,
        min_mask_sum=0,
        min_unclustered_sum=0,
        min_object_size=36,
    ):
        """
        Parameters
        ----------
        prediction :  PyTorch Tensor
                Model Prediction (5, H, W)
        n_sigma: int, default = 2
                Number of dimensions in Raw Image
        fg_thresh: float, default=0.5
                Foreground Threshold defines which pixels are considered
                to form the Foreground
                and which would need to be clustered into unique objects
        min_mask_sum: int
                Only start creating instances, if there are at least
                `min_mask_sum` pixels in foreground!
        min_unclustered_sum: int
                Stop when the number of seed candidates are less
                than `min_unclustered_sum`
        min_object_size: int
            Predicted Objects below this threshold are ignored

        Returns
        ----------
        instance: PyTorch Tensor (H, W)
                instance segmentation
        """

        from scipy.ndimage import gaussian_filter

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2 : 2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma : 2 + n_sigma + 1])  # 1 x h x w
        instance_map = torch.zeros(height, width).short()
        # instances = []  # list
        count = 1
        mask_fg = seed_map > fg_thresh
        seed_map_cpu = seed_map.cpu().detach().numpy()
        seed_map_cpu_smooth = gaussian_filter(seed_map_cpu[0], sigma=1)
        coords = peak_local_max(seed_map_cpu_smooth)
        zeros = np.zeros((coords.shape[0], 1), dtype=np.uint8)
        coords = np.hstack((zeros, coords))

        mask_local_max_cpu = np.zeros(seed_map_cpu.shape, dtype=np.bool)
        mask_local_max_cpu[tuple(coords.T)] = True
        mask_local_max = torch.from_numpy(mask_local_max_cpu).bool().to(self.device)

        mask_seed = mask_fg * mask_local_max
        if mask_fg.sum() > min_mask_sum:
            spatial_emb_fg_masked = spatial_emb[mask_fg.expand_as(spatial_emb)].view(
                n_sigma, -1
            )  # fg candidate pixels
            spatial_emb_seed_masked = spatial_emb[
                mask_seed.expand_as(spatial_emb)
            ].view(
                n_sigma, -1
            )  # seed candidate pixels

            sigma_seed_masked = sigma[mask_seed.expand_as(sigma)].view(
                n_sigma, -1
            )  # sigma for seed candidate pixels
            seed_map_seed_masked = seed_map[mask_seed].view(
                1, -1
            )  # seediness for seed candidate pixels

            unprocessed = (
                torch.ones(mask_seed.sum()).short().to(self.device)
            )  # unclustered seed candidate pixels
            unclustered = (
                torch.ones(mask_fg.sum()).short().to(self.device)
            )  # unclustered fg candidate pixels
            instance_map_masked = torch.zeros(mask_fg.sum()).short().to(self.device)
            while unprocessed.sum() > min_unclustered_sum:
                seed = (seed_map_seed_masked * unprocessed.float()).argmax().item()
                center = spatial_emb_seed_masked[:, seed : seed + 1]
                unprocessed[seed] = 0
                s = torch.exp(sigma_seed_masked[:, seed : seed + 1] * 10)
                dist = torch.exp(
                    -1 * torch.sum(torch.pow(spatial_emb_fg_masked - center, 2) * s, 0)
                )
                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if (
                        unclustered[proposal].sum().float() / proposal.sum().float()
                        > 0.5
                    ):
                        instance_map_masked[proposal.squeeze()] = count
                        count += 1
                        unclustered[proposal] = 0
            instance_map[mask_fg.squeeze().cpu()] = instance_map_masked.cpu()
        return instance_map

    def cluster(
        self,
        prediction,
        n_sigma=2,
        seed_thresh=0.9,
        fg_thresh=0.5,
        min_mask_sum=0,
        min_unclustered_sum=0,
        min_object_size=36,
    ):
        """
        Parameters
        ----------
        prediction :  PyTorch Tensor
                Model Prediction (5, H, W)
        n_sigma: int, default = 2
                Number of dimensions in Raw Image
        seed_thresh : float, default=0.9
                Seediness Threshold defines which pixels are considered to
                identify object centres
        fg_thresh: float, default=0.5
                Foreground Threshold defines which pixels are considered to
                form the "foreground"
                and hence would need to be clustered into unique objects
        min_mask_sum: int
                Only start creating instances, if there are at least
                `min_mask_sum` pixels in foreground!
        min_unclustered_sum: int
                Stop when the number of seed candidates are less than
                `min_unclustered_sum`
        min_object_size: int
            Predicted Objects below this threshold are ignored

        Returns
        ----------
        instance: PyTorch Tensor (H, W)
                instance segmentation
        """

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w

        sigma = prediction[2 : 2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma : 2 + n_sigma + 1])
        # 1 x h x w

        instance_map = torch.zeros(height, width).short()
        # instances = []  # list

        count = 1
        mask = seed_map > fg_thresh

        if mask.sum() > min_mask_sum:
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(
                n_sigma, -1
            )
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).short().to(self.device)
            instance_map_masked = torch.zeros(mask.sum()).short().to(self.device)

            while unclustered.sum() > min_unclustered_sum:
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < seed_thresh:
                    break
                center = spatial_emb_masked[:, seed : seed + 1]
                unclustered[seed] = 0

                s = torch.exp(sigma_masked[:, seed : seed + 1] * 10)
                dist = torch.exp(
                    -1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0)
                )

                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if (
                        unclustered[proposal].sum().float() / proposal.sum().float()
                        > 0.5
                    ):
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).short()
                        instance_mask[mask.squeeze().cpu()] = proposal.short().cpu()
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map

