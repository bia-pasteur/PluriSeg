"""Dataloader smoke tests against data_sample/SampleDataSet (fold-0 is pregenerated)."""

import numpy as np
import pytest
import torch

from pluriseg.configs import RepresentationConfig
from pluriseg.representations import build_representation
from pluriseg.representations.base import ChannelRole

# SampleDataSet: 11 pairs, CIS scheme -> fold-0: 6 train / 1 val / 4 test
# (test = the official test dir, val = a seeded fraction of train).
N_TRAIN, N_VAL, N_TEST = 6, 1, 4


def test_setup_split_sizes(datamodule):
    datamodule.setup("fit")
    assert len(datamodule.train_ds) == N_TRAIN
    assert len(datamodule.val_ds) == N_VAL
    datamodule.setup("test")
    assert len(datamodule.test_ds) == N_TEST


def test_train_batch_contract(datamodule):
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    img, lbl = batch["image"], batch["label"]
    assert img.shape == (2, 3, 256, 256)
    assert img.dtype == torch.float32
    assert lbl.shape == (2, 256, 256)
    assert lbl.dtype == torch.int64

    rep = build_representation(RepresentationConfig())
    for name, t in batch["targets"].items():
        n_ch = rep.channels[name].channels
        if rep.channels[name].role != ChannelRole.LOSS:
            assert t.shape == (2, n_ch, 256, 256)
            assert t.dtype == rep.channels[name].torch_type
    # spatial shape identical across image / targets / label
    for t in batch["targets"].values():
        assert t.shape[-2:] == img.shape[-2:]


def test_val_batch_contract(datamodule):
    datamodule.setup("fit")
    dl = datamodule.val_dataloader()
    batch = next(iter(dl))
    # val_ds has 1 item, so the batch is capped at 1 despite val_batch_size=3
    assert batch["image"].shape == (1, 3, 256, 256)
    assert batch["label"].shape == (1, 256, 256)


def test_test_loader_full_image_unpadded(datamodule):
    """Test items carry the original extent; the model pads around the
    backbone and decodes on the original extent (predict_labels)."""
    datamodule.setup("test")
    dl = datamodule.test_dataloader()
    batch = next(iter(dl))
    assert batch["image"].shape[0] == 1
    h, w = batch["image"].shape[-2:]
    # full image: 383x512 stays 383x512
    assert (h, w) == (383, 512)
    assert batch["label"].shape == (1, h, w)
    for t in batch["targets"].values():
        assert t.shape[-2:] == (h, w)


def test_val_is_deterministic(datamodule):
    datamodule.setup("fit")
    a = datamodule.val_ds[0]["image"]
    b = datamodule.val_ds[0]["image"]
    assert torch.equal(a, b)


def test_train_random_crop_varies(datamodule):
    datamodule.setup("fit")
    imgs = [datamodule.train_ds[0]["image"] for _ in range(8)]
    assert any(not torch.equal(imgs[0], x) for x in imgs[1:])


def test_deterministic_crop_pads_small_image(datamodule, tmp_path):
    from pluriseg.data import PatchDataset

    ds = PatchDataset(
        image_label_pairs=[], cache_dir=str(tmp_path / "cache"),
        dataset_name="test", rep=build_representation(RepresentationConfig()),
        cfg=datamodule.dataset_cfg, augs=datamodule.augs,
        patch=256, train=False, mode="val",
    )
    img = np.zeros((100, 90, 3), np.float32)
    out = ds._deterministic_crop(img)
    assert out[0].shape == (256, 256, 3)
    assert out[0][100:, :, :].sum() == 0  # bottom pad
    assert out[0][:, 90:, :].sum() == 0  # right pad


def test_train_random_crop_pads_small_image(datamodule, tmp_path):
    from pluriseg.data import PatchDataset

    ds = PatchDataset(
        image_label_pairs=[], cache_dir=str(tmp_path / "cache"),
        dataset_name="test", rep=build_representation(RepresentationConfig()),
        cfg=datamodule.dataset_cfg, augs=datamodule.augs,
        patch=256, train=True, mode="train",
    )
    ds.rng = np.random.default_rng(0)

    # small image: all outputs are exactly patch-sized so batches collate
    for shape in [(100, 90, 3), (256, 300, 3), (256, 256, 3)]:
        img = np.random.rand(*shape).astype(np.float32)
        for _ in range(4):
            out = ds._random_crop(img)
            assert all(a.shape[:2] == (256, 256) for a in out)

    # pad region must be GT-background in the label: zeros = total pad area
    img = np.zeros((100, 90, 3), np.float32)
    lbl = np.ones((100, 90), np.uint32)
    out_lbl = ds._random_crop(lbl)[0]
    n_pad = (256 - 100) * 256 + 100 * (256 - 90)
    assert (out_lbl == 0).sum() == n_pad


def test_train_loader_batches_uniform(datamodule):
    datamodule.setup("fit")
    dl = datamodule.train_dataloader()
    for batch in dl:
        h, w = batch["image"].shape[-2:]
        assert (h, w) == (256, 256)


def test_numpy_image_to_model_input_matches_test_getitem(datamodule):
    """Inference preprocessing must reproduce test-mode __getitem__ exactly."""
    import imageio.v3 as iio

    from pluriseg.data import numpy_image_to_model_input

    datamodule.setup("test")
    ds = datamodule.test_ds
    for idx in range(len(ds)):
        image_file, _, _ = ds.image_label_pairs[idx]
        raw = iio.imread(image_file)
        assert torch.equal(
            numpy_image_to_model_input(raw), ds[idx]["image"]
        )


def test_cache_written_and_reused(datamodule, tmp_path):
    datamodule.dataset_cfg.cache_dir = str(tmp_path / "cache")
    datamodule.setup("fit")
    cache_files = list((tmp_path / "cache").rglob("*.npz"))
    # warm_cache prebuilt everything: one entry per train+val image
    n_expected = len(datamodule.train_ds) + len(datamodule.val_ds)
    assert len(cache_files) == n_expected
    # second pass loads from cache (same content, no new files)
    _ = datamodule.val_ds[0]
    assert len(list((tmp_path / "cache").rglob("*.npz"))) == len(cache_files)


def test_warm_cache_opt_out(datamodule, tmp_path):
    datamodule.dataset_cfg.cache_dir = str(tmp_path / "cache")
    datamodule.dataset_cfg.warm_cache = False
    datamodule.setup("fit")
    # nothing prebuilt; cache fills lazily on first touch only
    assert not list((tmp_path / "cache").rglob("*.npz"))
    _ = datamodule.val_ds[0]
    assert len(list((tmp_path / "cache").rglob("*.npz"))) == 1


def test_warm_cache_multiworker(datamodule, tmp_path):
    """Worker-process fill must produce the same entries as the in-process
    path (atomic writes make concurrent fills safe)."""
    datamodule.dataset_cfg.cache_dir = str(tmp_path / "cache")
    datamodule.num_workers = 2
    datamodule.setup("fit")
    n_expected = len(datamodule.train_ds) + len(datamodule.val_ds)
    assert len(list((tmp_path / "cache").rglob("*.npz"))) == n_expected
