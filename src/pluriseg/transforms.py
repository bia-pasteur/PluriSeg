"""Geometric transforms shared by images and encoded representation (target) maps.

OrthoTransform covers the D4 group (flips + 90-degree rotations): exact on
encoded target maps (no resampling). AffineTransform covers rotation +
isotropic scale + translation: vector channels (flow/offset) transform by its
linear part, scalar channels are recomputed per Representation.transform_affine.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
from scipy.ndimage import affine_transform


def normalize_percentile(
    image: np.ndarray, low: float = 1.0, high: float = 99.0
) -> np.ndarray:
    """Clamp to the (low, high) percentiles and rescale so low -> 0.0, high -> 1.0.

    Works on (H, W) or (H, W, C) arrays (a single gain/offset for all channels).
    A degenerate image (hi == lo, e.g. constant) maps to all zeros.
    """
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, (low, high))
    if hi > lo:
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(image)
    return np.clip(image, 0.0, 1.0)


@dataclass
class OrthoTransform:
    """Composition of axis flips and 90-degree rotations (2D)."""

    flip_x: bool = False  # flip along the x axis (last spatial axis, W)
    flip_y: bool = False  # flip along the y axis (first spatial axis, H)
    rot90: int = 0  # number of 90-degree counter-clockwise rotations

    def __post_init__(self):
        self.rot90 = self.rot90 % 4

    def is_identity(self) -> bool:
        return not self.flip_x and not self.flip_y and self.rot90 == 0

    # -- images ---------------------------------------------------------------

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """Apply to an image array of shape (H, W) or (H, W, C).

        Returns a fresh contiguous copy when any transform applies (flip /
        rot90 views carry negative or transposed strides that torch.from_numpy
        rejects downstream); identity input passes through unchanged.
        """
        if self.flip_y:
            image = image[::-1, ...]
        if self.flip_x:
            image = image[:, ::-1, ...]
        if self.rot90:
            # rotate the spatial axes (the two leading axes)
            image = np.rot90(image, k=self.rot90, axes=(0, 1))
        if self.is_identity():
            return image
        return np.ascontiguousarray(image)

    # -- points ---------------------------------------------------------------

    def apply_to_points(self, points: np.ndarray, H: int, W: int) -> np.ndarray:
        """Apply to (N, 2) point coordinates in (x, y) convention.

        Index-space convention: pixel centers at integer (col=x, row=y).
        Forward map matching apply_to_image (flips first, then rot90):
        flip_y sends (y, x) -> (H-1-y, x), flip_x -> (y, W-1-x); one 90-degree
        CCW array rotation sends (y, x) -> (W-1-x, y) and swaps the extent.
        Returns the transformed *index* coordinates (x', y') in the transformed
        array's frame; H and W are the source image extent.
        """
        pts = np.asarray(points, dtype=np.float64).copy()
        x, y = pts[..., 0], pts[..., 1]
        if self.flip_y:
            y = (H - 1) - y
        if self.flip_x:
            x = (W - 1) - x
        # array rot90(k=1) on (H, W): (y, x) -> (W-1-x, y), extent -> (W, H)
        for _ in range(self.rot90):
            y, x = (W - 1) - x, y
            H, W = W, H
        pts[..., 0] = x
        pts[..., 1] = y
        return pts

    # -- composition ----------------------------------------------------------

    def compose(self, other: "OrthoTransform") -> "OrthoTransform":
        """Return the transform equivalent to applying self, then other."""
        raise NotImplementedError(
            "OrthoTransform composition is not needed yet; implement when required"
        )

    def inverse(self) -> "OrthoTransform":
        """Inverse transform (self-inverse group element)."""
        raise NotImplementedError(
            "OrthoTransform.inverse is not needed yet; implement when required"
        )


@dataclass
class AffineTransform:
    """2D affine transform: arbitrary rotation + isotropic scale + translation.

    Convention: maps source (y, x) pixel indices to output (y', x') via
        out_coords = self.matrix @ [y, x, 1]
    with matrix (2, 3) in (y, x) order. This matches scipy.ndimage.affine_transform
    (which inverts it internally for resampling) and OrthoTransform's array-index
    orientation: rot90 CCW on the array is positive theta here.

    Semantics chosen for augmentation: output pixel (y', x') samples source
    coords through the *inverse* matrix. Corners of the source not covered by
    the (rotated) output extent get the fill value passed to apply_to_image.
    """

    matrix: np.ndarray  # (2, 3) in (y, x) order

    @classmethod
    def from_params(
        cls,
        rot_deg: float = 0.0,
        scale: float = 1.0,
        center: tuple[float, float] | None = None,
    ) -> "AffineTransform":
        """Build from rotation (degrees, CCW on the array) and isotropic scale,
        optionally about `center` (y, x) so the transform is relative to a
        pivot instead of the origin."""
        c, s = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))
        # array (y, x): CCW visual rotation (rot90 k=1) is (y, x) -> (-x, y)
        R = np.array([[c, -s], [s, c]]) * scale
        if center is not None:
            cy, cx = center
            # A @ [cy, cx, 1] = [cy, cx]: shift -> rotate -> shift back
            t = np.array([cy, cx]) - R @ np.array([cy, cx])
            return cls(np.column_stack([R, t]))
        return cls(np.column_stack([R, np.zeros(2)]))

    def is_identity(self) -> bool:
        return bool(np.allclose(self.matrix, np.eye(2, 3)))

    @cached_property
    def scale(self) -> float:
        """Isotropic scale factor: sqrt(|det|) of the linear part (abs keeps
        reflected transforms, det < 0, positive). Read-only."""
        return float(np.sqrt(abs(np.linalg.det(self.matrix[:, :2]))))

    @cached_property
    def rot_deg(self) -> float:
        """Rotation in degrees (CCW on the array), inverse of from_params."""
        c, s = self.matrix[0, 0] / self.scale, self.matrix[1, 0] / self.scale
        return float(np.rad2deg(np.arctan2(s, c)))

    @cached_property
    def center(self) -> tuple[float, float]:
        """Fixed point (y, x) of the transform if one exists (the rotation/
        scale pivot), else None when the transform is a pure translation."""
        A = self.matrix
        M = A[:, :2] - np.eye(2)
        if np.linalg.matrix_rank(M) < 2:
            return None
        cy, cx = np.linalg.solve(M, -A[:, 2])
        return (float(cy), float(cx))

    # -- images ---------------------------------------------------------------

    def apply_to_image(
        self, image: np.ndarray, order: int = 1, fill: float = 0.0
    ) -> np.ndarray:
        """Warp (H, W) or (H, W, C); each channel resampled with scipy.

        order: interpolation degree (1 = bilinear, 0 = nearest). Use 0 for
        label / integer maps, 1+ for images and float channels. fill is the
        out-of-bounds value.
        """
        # affine_transform computes input_coords = matrix @ output_coords
        # (output = input through the given matrix), so pass the inverse.
        # grid-constant (not constant) keeps coordinates within one eps of the
        # border sampling the border value instead of the fill.
        A_inv = np.linalg.inv(np.vstack([self.matrix, [0, 0, 1]]))[:2]
        kwargs = dict(
            matrix=A_inv, order=order, mode="grid-constant", cval=fill
        )
        if image.ndim == 2:
            return affine_transform(image, **kwargs)
        return np.stack(
            [affine_transform(image[..., c], **kwargs)
             for c in range(image.shape[-1])],
            axis=-1,
        )

    def apply_to_images(
        self,
        images: list[np.ndarray],
        bilinear: list[bool] | None = None,
    ) -> list[np.ndarray]:
        """Warp several arrays, looping apply_to_image over them (slow path;
        AffineTransformV2 batches them through one grid_sample call).

        images: arrays of identical (H, W) extent, any mix of 2D and (H, W, C).
        bilinear: per-array interpolation (True = bilinear, False = nearest);
        default: bilinear for float arrays, nearest for integer arrays.
        """
        if bilinear is None:
            bilinear = [arr.dtype.kind == "f" for arr in images]
        return [
            self.apply_to_image(arr, order=1 if b else 0)
            for arr, b in zip(images, bilinear)
        ]

    # -- points ---------------------------------------------------------------

    def apply_to_points(self, points: np.ndarray) -> np.ndarray:
        """Apply to (N, 2) points in (x, y) convention (forward map)."""
        pts = np.asarray(points, dtype=np.float64)
        x, y = pts[..., 0], pts[..., 1]
        (a, b, ty), (d, e, tx) = self.matrix
        # A @ [y, x, 1] gives (y', x'); convert back to (x', y')
        out = np.stack([d * y + e * x + tx, a * y + b * x + ty], axis=-1)
        return out

    # -- composition ----------------------------------------------------------

    def compose(self, other: "AffineTransform") -> "AffineTransform":
        """Return the transform equivalent to applying self, then other."""
        A1 = np.vstack([self.matrix, [0, 0, 1]])
        A2 = np.vstack([other.matrix, [0, 0, 1]])
        return AffineTransform((A2 @ A1)[:2])

    def inverse(self) -> "AffineTransform":
        """Inverse transform."""
        A = np.vstack([self.matrix, [0, 0, 1]])
        return AffineTransform(np.linalg.inv(A)[:2])


@dataclass
class AffineTransformV2:
    """2D affine transform (rotation + isotropic scale + translation), torch
    backend: all channels warp through one `F.grid_sample` call instead of
    per-channel scipy calls (~20x faster for chimera's ~15 map channels).

    Same convention as AffineTransform.from_params: output pixel (y', x')
    samples source coords through the inverse of the (y, x)-order matrix
    (scipy affine_transform semantics), CCW positive rotation. Pixel-center
    (integer-index) coordinates with align_corners=True, so warped label
    (nearest) and float maps (bilinear) share one grid and stay consistent
    (stardist's radii re-encoded from t_label must align with warped
    edt_prob). Verified: interior samples match scipy order-1 to float32
    precision. Out-of-source samples get `fill` (scipy grid-constant blends
    toward the fill across one pixel at the border; here they get the fill
    directly — an at most 1-px ring inside the discarded margin ring of the
    crop-first data pipeline).
    """

    matrix: np.ndarray  # (2, 3) in (y, x) order

    @classmethod
    def from_params(
        cls,
        rot_deg: float = 0.0,
        scale: float = 1.0,
        center: tuple[float, float] | None = None,
    ) -> "AffineTransformV2":
        """Same construction as AffineTransform.from_params."""
        c, s = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))
        R = np.array([[c, -s], [s, c]]) * scale
        if center is not None:
            cy, cx = center
            t = np.array([cy, cx]) - R @ np.array([cy, cx])
            return cls(np.column_stack([R, t]))
        return cls(np.column_stack([R, np.zeros(2)]))

    def is_identity(self) -> bool:
        return bool(np.allclose(self.matrix, np.eye(2, 3)))

    @cached_property
    def scale(self) -> float:
        return float(np.sqrt(abs(np.linalg.det(self.matrix[:, :2]))))

    @cached_property
    def _grid_norm(self) -> dict:
        """Per-extent normalized (x, y) grids, built lazily per extent.
        One transform instance is used per item (all its map warps share the
        geometry); caching the normalized grid here removes the repeated
        float64 meshgrid+inverse from multi-rep transforms (chimera builds
        5 grids per item without this)."""
        return {}

    def _normalized_grid(self, out_h: int, out_w: int) -> torch.Tensor:
        """grid_sample-ready (1, out_h, out_w, 2) (x, y) grid in [-1, 1],
        align_corners=True (pixel centers at the border)."""
        key = (out_h, out_w)
        g = self._grid_norm.get(key)
        if g is None:
            grid = self._source_grid(out_h, out_w)
            gy = grid[..., 0] * (2.0 / (out_h - 1)) - 1.0
            gx = grid[..., 1] * (2.0 / (out_w - 1)) - 1.0
            g = torch.stack([gx, gy], dim=-1)[None]
            self._grid_norm[key] = g
        return g

    def _source_grid(self, out_h: int, out_w: int) -> torch.Tensor:
        """(out_h, out_w, 2) source sampling coordinates (y, x), float32, for
        input_coords = A_inv @ [y', x', 1] (scipy affine_transform semantics:
        output samples the source through the inverse matrix)."""
        yy, xx = np.meshgrid(np.arange(out_h, dtype=np.float64),
                             np.arange(out_w, dtype=np.float64),
                             indexing="ij")
        A_inv = np.linalg.inv(np.vstack([self.matrix, [0, 0, 1]]))[:2]
        sy = A_inv[0, 0] * yy + A_inv[0, 1] * xx + A_inv[0, 2]
        sx = A_inv[1, 0] * yy + A_inv[1, 1] * xx + A_inv[1, 2]
        return torch.from_numpy(np.stack([sy, sx], axis=-1)).float()

    def apply_to_image(
        self, image: np.ndarray, order: int = 1, fill: float = 0.0
    ) -> np.ndarray:
        """Warp (H, W) or (H, W, C); one grid_sample for all channels.

        order: 1 = bilinear (images, float channels), 0 = nearest (labels).
        fill is the out-of-bounds value (0 expected; the center-crop margin
        design keeps fill out of the surviving patch).
        """
        H, W = image.shape[:2]
        out_h, out_w = H, W  # same-extent warp (crop-first design)
        if image.ndim == 2:
            image = image[..., None]
        squeeze = True
        grid = self._source_grid(out_h, out_w)
        g = self._normalized_grid(out_h, out_w)  # grid_sample-ready (x, y)
        x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None]
        if x.dtype == torch.float64:
            # float64 arrays (e.g. flow fields) downcast: float32 precision is
            # ample for warp resampling
            x = x.float()
        elif not x.is_floating_point():
            # grid_sample only implements floating dtypes; nearest sampling
            # rounds back to the integer values exactly, so cast is lossless
            # for the label/integer channels this path serves
            x = x.float()
        mode = "nearest" if order == 0 else "bilinear"
        out = torch.nn.functional.grid_sample(
            x, g, mode=mode, padding_mode="zeros", align_corners=True)
        out = out[0].permute(1, 2, 0).numpy()
        if fill != 0.0:
            # padding_mode="zeros" only; nonzero fill via a validity mask
            valid = (grid[..., 0] >= 0) & (grid[..., 0] <= H - 1) & \
                    (grid[..., 1] >= 0) & (grid[..., 1] <= W - 1)
            out[~valid.numpy()] = fill
        if squeeze and out.shape[-1] == 1:
            out = out[..., 0]
        return out

    def apply_to_images(
        self,
        images: list[np.ndarray],
        bilinear: list[bool] | None = None,
    ) -> list[np.ndarray]:
        """Warp several arrays through one shared grid_sample call.

        The real pipeline warps image + label + all encoded maps with the same
        transform: building the sampling grid once and running a single
        (1, C, H, W) grid_sample cuts per-item cost ~5x vs repeated
        apply_to_image calls (grid build dominates at these sizes).

        images: arrays of identical (H, W) extent, any mix of 2D and (H, W, C).
        bilinear: per-array interpolation (True = bilinear, False = nearest);
        default: bilinear for float arrays, nearest for integer arrays.
        Returns arrays of the same shapes as the inputs.
        """
        ref = images[0]
        H, W = ref.shape[:2]
        g = self._normalized_grid(H, W)

        if bilinear is None:
            bilinear = [arr.dtype.kind == "f" for arr in images]
        tensors = []
        for arr in images:
            if arr.ndim == 2:
                arr = arr[..., None]
            t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))[None]
            if t.dtype == torch.float64:
                t = t.float()
            elif not t.is_floating_point():
                # grid_sample is floating-only; nearest rounds back to the
                # integer values exactly (lossless for the label channels)
                t = t.float()
            tensors.append(t)
        x = torch.cat(tensors, dim=1)
        # channel offsets of each array within the concatenated tensor
        sizes = [t.shape[1] for t in tensors]
        offsets = np.cumsum([0] + sizes)
        # one bilinear + one nearest pass over the concatenated channels;
        # keep each array's channel count to restore its original shape
        out = []
        for mode, mask in (("bilinear", bilinear), ("nearest", [not b for b in bilinear])):
            idx = [i for i, b in enumerate(mask) if b]
            if not idx:
                continue
            chans = [c for i in idx for c in range(offsets[i], offsets[i + 1])]
            r = torch.nn.functional.grid_sample(
                x[:, chans], g, mode=mode, padding_mode="zeros", align_corners=True)
            r = r[0].permute(1, 2, 0).numpy()
            col = 0
            for i in idx:
                c = images[i].shape[-1] if images[i].ndim == 3 else 1
                ch = r[..., col:col + c]
                col += c
                if images[i].ndim == 2:
                    ch = ch[..., 0]
                out.append((i, ch))
        return [arr for _, arr in sorted(out)]

    def apply_to_points(self, points: np.ndarray) -> np.ndarray:
        """Apply to (N, 2) points in (x, y) convention (forward map)."""
        pts = np.asarray(points, dtype=np.float64)
        x, y = pts[..., 0], pts[..., 1]
        (a, b, ty), (d, e, tx) = self.matrix
        out = np.stack([d * y + e * x + tx, a * y + b * x + ty], axis=-1)
        return out
