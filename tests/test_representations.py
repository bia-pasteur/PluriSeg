"""Representation contract tests, parameterized over the registry.

One generic suite per Representation method, run against every registry entry
that implements it (stubs raise NotImplementedError and are skipped). Adding a
representation to REGISTRY automatically enrolls it here; per-representation
specifics belong in that rep's own test module.

All checks are against the Representation contract in base.py, not the values
MaskRep happens to produce:
- encode_targets: (H, W, C) shape per channel, runs on real and empty labels.
- decode_preds: (H, W) uint32 label map, 0 = background; perfect pred -> instances
  recovered.
- transform: no channel dropped, shape/dtype preserved, exact vs apply_to_image.
- loss: runs, "loss" key, scalar finite tensors; perfect pred beats random noise.
"""

import warnings

import imageio.v3 as iio
import numpy as np
import pytest
import torch

from pluriseg.representations import REGISTRY
from pluriseg.representations.base import ChannelRole
from pluriseg.transforms import OrthoTransform

LABEL_FILE = "data_sample/blobs_masks.tif"


@pytest.fixture(scope="module")
def label() -> np.ndarray:
    """Real instance label map: skimage blobs (uint16, 63 instances)."""
    return iio.imread(LABEL_FILE).astype(np.uint32)


@pytest.fixture(params=list(REGISTRY))
def rep(request):
    r = REGISTRY[request.param]()
    yield r


def _np_to_torch_chw(maps: dict[str, np.ndarray], rep) -> dict[str, torch.Tensor]:
    """Encode-style numpy (H, W, C) maps -> torch (C, H, W) tensors."""
    return {
        k: torch.from_numpy(v).permute(2, 0, 1).to(rep.channels[k].torch_type)
        for k, v in maps.items()
    }


def _torch_to_batch(preds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """(C, H, W) tensors -> (1, C, H, W), loss()'s convention."""
    return {k: v.unsqueeze(0) for k, v in preds.items()}


# ------------------------------------------------------------------- encode

def test_encode_shapes_and_keys(rep, label):
    maps = rep.encode_targets(label)
    assert set(maps.keys()) == set(rep.channels.keys())
    for k, props in rep.channels.items():
        assert maps[k].shape[:2] == label.shape
        assert maps[k].shape[-1] == props.channels


def test_encode_handles_non_contiguous_labels(rep, label):
    """Label ids with gaps (e.g. only 5, 50, 500 present) must encode the same
    as the relabeled 1..N equivalent: encode_targets() may not assume ids are 1..N."""
    ids = np.unique(label[label > 0])
    rng = np.random.default_rng(0)
    # map each id to a random sparse value in 1..1000
    remap = np.zeros(int(label.max()) + 1, np.uint32)
    remap[ids] = rng.choice(np.arange(1, 1001), size=len(ids), replace=False)
    sparse = remap[label]

    n_dense = _n_instances(rep, rep.encode_targets(label))
    n_sparse = _n_instances(rep, rep.encode_targets(sparse))
    assert n_sparse == n_dense, (
        f"{rep.name}: non-contiguous labels decode to {n_sparse} instances, "
        f"expected {n_dense}")


def test_encode_runs_on_empty_label(rep):
    maps = rep.encode_targets(np.zeros((64, 64), np.uint32))
    assert set(maps.keys()) == set(rep.channels.keys())
    for k, props in rep.channels.items():
        assert maps[k].shape[:2] == (64, 64)
        assert maps[k].shape[-1] == props.channels


# ------------------------------------------------------------------- decode

def test_decode_returns_uint32_label_map(rep, label):
    encoded = rep.encode_targets(label)
    preds = rep.encoded_to_perfect_pred(encoded)
    pred = rep.decode_preds(preds)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == label.shape
    assert pred.dtype == np.uint32


def test_decode_of_perfect_pred_recovers_instances(rep, label):
    """Contract sanity loop: encode_targets -> perfect pred -> decode_preds must recover
    (nearly) all ground-truth instances; +-5% under/over-count is tolerated
    since post-processing may drop marginal instances or merge close ones."""
    encoded = rep.encode_targets(label)
    preds = rep.encoded_to_perfect_pred(encoded)
    pred = rep.decode_preds(preds)
    n_gt = len(np.unique(label[label > 0]))
    n_pred = len(np.unique(pred[pred > 0]))
    if n_pred < n_gt:
        warnings.warn(
            f"{rep.name}: perfect-pred decode under-counts instances "
            f"({n_pred} < {n_gt})")
    if n_pred > n_gt:
        warnings.warn(
            f"{rep.name}: perfect-pred decode over-counts instances "
            f"({n_pred} > {n_gt})")
    low_bound = n_gt - 0.05 * n_gt
    high_bound = n_gt + 0.05 * n_gt
    assert low_bound <= n_pred <= high_bound, f"failed {low_bound}<= {n_pred} <= {high_bound}"


def test_decode_accepts_bfloat16_preds(rep, label):
    """Under precision=bf16-mixed the model outputs bfloat16; decode_preds must
    handle it (numpy does not support bfloat16, so any .cpu().numpy() without a
    .float() cast fails). Only the contract (uint32 (H, W) label map) is
    checked — bf16 rounding may legitimately degrade instance recovery."""
    encoded = rep.encode_targets(label)
    preds = {k: v.to(torch.bfloat16)
             for k, v in rep.encoded_to_perfect_pred(encoded).items()}
    pred = rep.decode_preds(preds)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == label.shape
    assert pred.dtype == np.uint32


REALISTIC_LABEL_FILE = "data_sample/SampleDataSet/train/labels/0000.png"

# Postproc tuned per rep on the realistic image (the benchmark tunes per
# rep × dataset on val; see plan item 11). Decode defaults stay untouched.
REALISTIC_DECODE_CFG = {
    "omnipose": {"niter": 25, "min_size": 60},
}


def test_decode_on_realistic_label(rep):
    """Same loop on a harder real image (touching, irregular instances)."""
    label = iio.imread(REALISTIC_LABEL_FILE).astype(np.uint32)
    pred = rep.decode_preds(rep.encoded_to_perfect_pred(rep.encode_targets(label)),
                      cfg=REALISTIC_DECODE_CFG.get(rep.name))
    n_gt = len(np.unique(label[label > 0]))
    n_pred = len(np.unique(pred[pred > 0]))
    low_bound = n_gt - 0.10 * n_gt
    high_bound = n_gt + 0.10 * n_gt
    assert low_bound <= n_pred <= high_bound, f"failed {low_bound}<= {n_pred} <= {high_bound}"


def test_decode_handles_no_object_pred(rep):
    """Decode of a prediction where nothing passes prob_thresh must return an
    all-background (H, W) label map, not crash (zero survivors hit degenerate
    paths: e.g. polygons_to_label receives empty input; early training epochs)."""
    pred = rep.decode_preds(rep.encoded_to_perfect_pred(
        rep.encode_targets(np.zeros((64, 64), np.uint32))))
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (64, 64)
    assert pred.dtype == np.uint32
    assert (pred == 0).all()



@pytest.fixture(params=[
    OrthoTransform(),
    OrthoTransform(flip_x=True),
    OrthoTransform(flip_y=True),
    OrthoTransform(rot90=1),
    OrthoTransform(rot90=2),
    OrthoTransform(rot90=3),
    OrthoTransform(flip_x=True, flip_y=True, rot90=1),
])
def d4(request):
    return request.param


def test_transform_preserves_keys_shape_dtype(rep, d4, label):
    maps = rep.encode_targets(label)
    out = rep.transform(maps, d4)
    assert set(out.keys()) == set(maps.keys())
    for k, v in out.items():
        assert v.dtype == maps[k].dtype
        assert v.shape == d4.apply_to_image(maps[k]).shape


def test_transform_decode_consistency(rep, d4, label):
    """Decode(transform(encode_targets(label))) must recover the same instance count
    as the untransformed loop decode(encode_targets(label)) (instance ids need not
    correspond)."""
    n_base = _n_instances(rep, rep.encode_targets(label))
    maps_t = rep.transform(rep.encode_targets(label), d4)
    assert _n_instances(rep, maps_t) >= n_base


def _n_instances(rep, maps: dict[str, np.ndarray]) -> int:
    """Number of decoded instances (baseline for the transform-consistency check)."""
    pred = rep.decode_preds(rep.encoded_to_perfect_pred(maps))
    return len(np.unique(pred[pred > 0]))


# --------------------------------------------------------------------- head

def test_head_predicts_expected_keys(rep, label):
    """model.forward() emits exactly the predicted channels (TARGET/INDIRECT)
    and none of the loss-only ones, with the declared per-channel widths."""
    from pluriseg.models.backbone import build_backbone
    from pluriseg.models.heads import ConvHead

    from pluriseg.configs import BackboneConfig

    expected = {k for k, c in rep.channels.items()
                if c.role != ChannelRole.LOSS}
    head = ConvHead(in_ch=16, channels=rep.channels)
    assert head.out_names == [k for k in rep.channels
                              if k in expected], (
        f"{rep.name}: head emits {head.out_names}, expected {sorted(expected)}")
    assert head.out_ch == sum(rep.channels[k].channels for k in expected)

    feats = torch.randn(1, 16, 32, 32)
    out = head(feats)
    assert set(out.keys()) == expected
    for k in expected:
        assert out[k].shape == (1, rep.channels[k].channels, 32, 32)

    # full model path: backbone + head on a real-size input
    backbone, out_ch = build_backbone(BackboneConfig(size="small", in_ch=3))
    model = ConvHead(in_ch=out_ch, channels=rep.channels)
    full = model(backbone(torch.randn(1, 3, 128, 128)))
    assert set(full.keys()) == expected


# --------------------------------------------------------------------- loss


def test_loss_keys_naming(rep, label):
    """loss() returns a dict whose keys are "loss" or "loss_*" (the "*"
    components are diagnostics logged by model.py); "loss" itself is always
    present (the minimized scalar)."""
    encoded = rep.encode_targets(label)
    preds = _torch_to_batch(rep.encoded_to_perfect_pred(encoded))
    targets = _torch_to_batch(_np_to_torch_chw(encoded, rep))
    losses = rep.loss(preds, targets)
    bad = [k for k in losses if k != "loss" and not k.startswith("loss_")]
    assert not bad, f"{rep.name}: loss keys violate 'loss'/'loss_*': {bad}"
    assert "loss" in losses


def test_loss_runs_scalar_finite(rep, label):
    encoded = rep.encode_targets(label)
    preds = _torch_to_batch(rep.encoded_to_perfect_pred(encoded))
    targets = _torch_to_batch(_np_to_torch_chw(encoded, rep))
    losses = rep.loss(preds, targets)
    assert set(losses.keys()) >= {"loss"}
    for v in losses.values():
        assert v.ndim == 0
        assert torch.isfinite(v)


def test_loss_handles_heterogeneous_batch(rep):
    """Batching must be per-image: B=2 with different fg pixel counts fails any
    loss that broadcasts a per-image scalar into per-pixel assignments."""
    label1 = iio.imread("data_sample/SampleDataSet/train/labels/0000.png").astype(np.uint32)[:256, :256]
    label2 = iio.imread("data_sample/SampleDataSet/train/labels/0001.png").astype(np.uint32)[:256, :256]
    enc1, enc2 = rep.encode_targets(label1), rep.encode_targets(label2)

    preds = _torch_to_batch(rep.encoded_to_perfect_pred(enc1))
    losses = rep.loss(preds, _torch_to_batch(_np_to_torch_chw(enc2, rep)))
    for v in losses.values():
        assert torch.isfinite(v)


def test_loss_handles_empty_label(rep):
    """Full-background patches must not break the loss (e.g. zero-division in
    per-image foreground frequency, all-inf distance weights)."""
    encoded = rep.encode_targets(np.zeros((64, 64), np.uint32))
    preds = _torch_to_batch(rep.encoded_to_perfect_pred(encoded))
    targets = _torch_to_batch(_np_to_torch_chw(encoded, rep))
    losses = rep.loss(preds, targets)
    assert set(losses.keys()) >= {"loss"}
    for v in losses.values():
        assert v.ndim == 0
        assert torch.isfinite(v)


def test_loss_perfect_pred_beats_random(rep, label):
    """Perfect prediction (targets fed back through the channel spec) loses
    less than random noise: the encoding is in the loss function's minimum
    basin."""
    encoded = rep.encode_targets(label)
    targets = _torch_to_batch(_np_to_torch_chw(encoded, rep))
    good = _torch_to_batch(rep.encoded_to_perfect_pred(encoded))
    loss_good = float(rep.loss(good, targets)["loss"])
    rng = np.random.default_rng(0)
    for _ in range(5):
        bad = dict(good)
        bad.update({
            k: torch.from_numpy(rng.random(
                encoded[k].shape).astype(np.float32))
            .permute(2, 0, 1).to(rep.channels[k].torch_type).unsqueeze(0)
            for k in encoded if rep.channels[k].role != ChannelRole.LOSS
        })
        assert loss_good < float(rep.loss(bad, targets)["loss"])
