"""Affine-transform contract tests, parameterized over the registry.

transform_affine output is compared against the ground-truth re-encode of the
warped label (encode_targets applied to t_label) — the values the encoder
itself would produce on the transformed geometry.

Two levels of checking:
- Instance recovery: decoded masks of both sides are matched greedily by IoU.
  A small fraction of GT instances may legitimately vanish or fragment at
  instance borders under resampling; wholesale instance loss is a failure.
- Derived-value sanity: per channel, on pixels active in the GT re-encode,
  mean and extremes of the transformed values (optionally mapped to their
  loss-relevant scale, e.g. exp(10*sigma) for EmbedSeg) must stay close to
  the reference. This catches warped-value artifacts such as bilinear-warp
  sigma inflation.

The identity transform must reproduce the encoded maps exactly.
"""

import imageio.v3 as iio
import numpy as np
import pytest

from pluriseg.representations import REGISTRY
from pluriseg.transforms import AffineTransformV2

LABEL_FILE = "data_sample/blobs_masks.tif"

# fixed, deterministic transforms: (rot_deg, scale)
AFFINES = [(0.0, 1.0), (30.0, 0.8), (135.0, 1.25)]

IOU_THR = 0.5         # IoU above which decoded instances are matched
MAX_UNMATCHED = 0.25  # max fraction of GT instances without an IoU match
MEAN_TOL = 0.30       # mean deviation relative to mean|value| of the GT
MEAN_TOL_SIGMA = 0.15  # tighter for sigma (re-encoding should be exact)
EXT_TOL = 3.0         # tolerated deviation of extremes, relative to GT

# instance-id channels: ids are labels, not quantities (the warp keeps
# original ids with gaps where instances vanished, a fresh re-encode
# relabels contiguously) — their geometry is covered by the IoU matching
ID_CHANNELS = {"instance", "cellpose_instance", "embedseg_instance",
               "omnipose_instance", "stardist_instance"}


@pytest.fixture(scope="module")
def label():
    """Real instance label map: skimage blobs (63 instances)."""
    return iio.imread(LABEL_FILE).astype(np.uint32)


@pytest.fixture(params=list(REGISTRY))
def rep(request):
    r = REGISTRY[request.param]()
    # stubs raise NotImplementedError; probe with a minimal call
    lab = np.zeros((8, 8), np.uint32)
    lab[2:6, 2:6] = 1
    t = AffineTransformV2.from_params(0.0, 1.0, center=(4, 4))
    try:
        r.transform_affine(r.encode_targets(lab), t, lab)
    except NotImplementedError:
        pytest.skip("transform_affine not implemented")
    return r


def test_transform_affine_matches_reencode(rep, label):
    maps = rep.encode_targets(label)
    H, W = label.shape
    for rot, scale in AFFINES:
        t = AffineTransformV2.from_params(
            rot_deg=rot, scale=scale, center=(H // 2, W // 2))
        t_label = t.apply_to_image(label, order=0).astype(label.dtype)
        out = rep.transform_affine(maps, t, t_label)
        gt = rep.encode_targets(t_label)

        assert set(out.keys()) == set(maps.keys())
        for k in out:
            assert out[k].shape == maps[k].shape
            if maps[k].dtype.kind == "f":
                assert out[k].dtype.kind == "f", k
            # identity warp must reproduce the encoded maps, within float32
            # epsilon: grid normalization rounds sampling coords by ~1e-5 px,
            # so border pixels blend ~1e-4 of a neighbouring pixel's value
            if rot == 0.0 and scale == 1.0:
                o = np.asarray(out[k], float)
                g_ = np.asarray(gt[k], float)
                atol = 1e-4 * max(np.abs(g_).max(), 1.0)
                assert np.allclose(o, g_, rtol=1e-3, atol=atol,
                                   equal_nan=False), k

        dec_out = rep.decode_preds(rep.encoded_to_perfect_pred(out))
        dec_gt = rep.decode_preds(rep.encoded_to_perfect_pred(gt))
        frac, n_gt = _matched_fraction(dec_gt, dec_out)
        assert n_gt == 0 or frac >= 1 - MAX_UNMATCHED, (
            f"rot={rot} scale={scale}: only {frac:.0%} of {n_gt} GT instances "
            f"matched (transform_affine vs re-encode decode)")

        for k in out:
            if k in ID_CHANNELS:
                continue
            fails = _check_channel(np.asarray(out[k]), np.asarray(gt[k]), k)
            assert not fails, f"rot={rot} scale={scale}: " + "; ".join(fails)


def _matched_fraction(gt_map, pred_map, thr=IOU_THR):
    """Greedy IoU matching; returns (matched fraction of GT instances, n_gt)."""
    gids = np.unique(gt_map[gt_map > 0])
    pids = np.unique(pred_map[pred_map > 0])
    if len(gids) == 0:
        return 1.0, 0
    if len(pids) == 0:
        return 0.0, len(gids)
    gi = {g: i for i, g in enumerate(gids)}
    pi = {p: j for j, p in enumerate(pids)}
    gl = np.fromiter((gi.get(v, -1) for v in gt_map.ravel()), int, gt_map.size)
    pl = np.fromiter((pi.get(v, -1) for v in pred_map.ravel()), int, pred_map.size)
    both = (gl >= 0) & (pl >= 0)
    cont = np.zeros((len(gids), len(pids)))
    np.add.at(cont, (gl[both], pl[both]), 1)
    ga = np.bincount(gl[gl >= 0], minlength=len(gids))
    pa = np.bincount(pl[pl >= 0], minlength=len(pids))
    union = ga[:, None] + pa[None, :] - cont
    iou = cont / np.maximum(union, 1)
    used, matched = set(), 0
    for i in np.argsort(-iou.max(axis=1)):
        j = int(iou[i].argmax())
        if iou[i, j] >= thr and j not in used:
            matched += 1
            used.add(j)
    return matched / len(gids), len(gids)


def _derived(name, v):
    """Loss-relevant scale of a channel's values (e.g. EmbedSeg sigma)."""
    if name.split("_")[-1] == "sigma":
        return np.exp(10.0 * v)
    return v


def _check_channel(o, g, name):
    """Compare transformed vs GT-re-encoded values on GT-active pixels."""
    fails = []
    act = (np.abs(g).sum(axis=-1) > 0) if g.ndim == 3 else (g != 0)
    if not act.any():
        return fails
    dv = _derived(name, o[act].astype(np.float64))
    dg = _derived(name, g[act].astype(np.float64))
    scale = max(np.abs(dg).mean(), 1e-12)
    tol = MEAN_TOL_SIGMA if name.split("_")[-1] == "sigma" else MEAN_TOL
    if abs(dv.mean() - dg.mean()) > tol * scale:
        fails.append(f"{name}: mean {dv.mean():.4g} vs GT {dg.mean():.4g}")
    if dv.max() > EXT_TOL * dg.max() + 1e-12:
        fails.append(f"{name}: max {dv.max():.4g} vs GT {dg.max():.4g}")
    return fails
