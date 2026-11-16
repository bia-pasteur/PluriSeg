"""End-to-end smoke test: 2-epoch fit on SampleDataSet, checkpointing + CSV logging."""

import os
import warnings
from pathlib import Path

import pytest
import torch

from pluriseg.configs import (
    AugmentationConfig,
    BackboneConfig,
    DatasetConfig,
    OptimizerConfig,
    RepresentationConfig,
    TrainConfig,
)
from pluriseg.train import run_training

RUN_DIR = "scratch/test-smoke-train"


def test_train_smoke(tmp_path):
    run_dir = str(tmp_path / "smoke")
    run_training(
        rep=RepresentationConfig(name="mask", params={}),
        backbone=BackboneConfig(size="small", in_ch=3),
        data=DatasetConfig(root="data_sample/SampleDataSet", split_name="SampleDataSet"),
        augs=AugmentationConfig(flip_x=True, flip_y=True, rot90=True),
        optimizer=OptimizerConfig(lr=3e-4, wd=1e-5, epochs=2),
        train=TrainConfig(
            seed_everything=0,
            patch=256,
            batch_size=2,
            val_batch_size=2,
            num_workers=0,
            run_dir=run_dir,
            model_name="test",
            save_top_k=1,
            ckpt_every_n_epochs=1,
            log_every_n_steps=1,
        ),
    )

    ckpt_dir = os.path.join(run_dir, "test")
    names = os.listdir(ckpt_dir)
    # one run subdir (auto timestamp_name) holding checkpoints + metrics
    assert len(names) == 1, names
    run_subdir = os.path.join(ckpt_dir, names[0])
    ckpts = os.listdir(os.path.join(run_subdir, "checkpoints"))
    # val_ap50 is logged every epoch, so the monitor sees 2 candidate points.
    assert any(n.startswith("best-") for n in ckpts), ckpts
    assert "last.ckpt" in ckpts, ckpts

    assert os.path.isfile(os.path.join(run_subdir, "metrics.csv"))

    # checkpoint round-trips: hyperparameter config alone rebuilds the model
    best = next(n for n in ckpts if n.startswith("best-"))
    ckpt = torch.load(
        os.path.join(run_subdir, "checkpoints", best),
        weights_only=False,
        map_location="cpu",
    )
    assert ckpt["hyper_parameters"]["rep_cfg"].name == "mask"

    # whole-image test loop ran: per-image rows + mean row + saved preds
    import pandas as pd

    from pluriseg.splits import load_split

    test_dir = os.path.join(run_subdir, "eval")
    metrics = pd.read_csv(os.path.join(test_dir, "metrics.csv"))
    _, split = load_split(DatasetConfig(
        root="data_sample/SampleDataSet", split_name="SampleDataSet"))
    n_test = len(split["test"])
    # per-image rows + the mean row
    assert len(metrics) == n_test + 1, metrics
    assert metrics.iloc[-1]["image_source"] == "mean"
    assert set(metrics.columns) >= {
        "image_source", "n_gt", "n_pred", "AP@0.25", "AP@0.50", "AP@0.75",
        "AP@[.5:.95]"}
    preds = [n for n in os.listdir(test_dir) if n.endswith("_pred.tif")]
    assert len(preds) == n_test, preds


def test_eval_cli_smoke(tmp_path):
    """Standalone eval: fit briefly, then eval the best checkpoint via eval()."""
    from pluriseg.cli import eval_ as eval_cli

    run_dir = str(tmp_path / "eval-smoke")
    run_training(
        rep=RepresentationConfig(name="mask", params={}),
        backbone=BackboneConfig(size="small", in_ch=3),
        data=DatasetConfig(root="data_sample/SampleDataSet", split_name="SampleDataSet"),
        augs=AugmentationConfig(flip_x=True, flip_y=True, rot90=True),
        optimizer=OptimizerConfig(lr=3e-4, wd=1e-5, epochs=1),
        train=TrainConfig(
            seed_everything=0,
            patch=256,
            batch_size=2,
            val_batch_size=2,
            num_workers=0,
            run_dir=run_dir,
            model_name="test-eval",
            save_top_k=1,
            ckpt_every_n_epochs=1,
            log_every_n_steps=1,
            test_after_training=False,
        ),
    )

    run_subdir = next(
        os.path.join(run_dir, "test-eval", d)
        for d in os.listdir(os.path.join(run_dir, "test-eval")))
    best = next(
        n for n in os.listdir(os.path.join(run_subdir, "checkpoints"))
        if n.startswith("best-"))
    ckpt_path = os.path.join(run_subdir, "checkpoints", best)

    eval_cli(
        ckpt_path=ckpt_path,
        out_dir=str(tmp_path / "eval-out"),
    )

    import pandas as pd

    from pluriseg.splits import load_split

    out_dir = tmp_path / "eval-out"
    metrics = pd.read_csv(out_dir / "metrics.csv")
    _, split = load_split(DatasetConfig(
        root="data_sample/SampleDataSet", split_name="SampleDataSet"))
    n_test = len(split["test"])
    assert len(metrics) == n_test + 1, metrics
    assert metrics.iloc[-1]["image_source"] == "mean"
    preds = [n for n in os.listdir(out_dir) if n.endswith("_pred.tif")]
    assert len(preds) == n_test, preds


def test_eval_missing_checkpoint(tmp_path):
    from pluriseg.cli import eval_ as eval_cli

    with pytest.raises(FileNotFoundError):
        eval_cli(ckpt_path=str(tmp_path / "does-not-exist.ckpt"))


def test_config_roundtrip(tmp_path):
    """The saved config.yaml must fully reconstruct every config section:
    run_training's dump feeds eval_'s load; nothing may be lossy."""
    import yaml

    from fluometrik.ops import OPS
    from pluriseg.configs import (AugmentationConfig, BackboneConfig,
                                DatasetConfig, OptimizerConfig, TrainConfig,
                                load_photometric)
    from pluriseg.eval.evaluate import eval_ as eval_cli
    from pluriseg.train import run_training

    run_dir = str(tmp_path / "cfg-roundtrip")
    photometric = [
        {"op": "Background", "p": 0.2, "channel_mode": "independent",
         "amplitude": [0.05, 0.2], "scale": [0.05, 0.5]},
        {"op": "Gamma", "p": 0.5, "channel_mode": "independent",
         "gamma": [0.5, 2.0]},
        {"op": "ShotNoise", "p": 0.5, "peak": [10.0, 100.0]},
    ]
    run_training(
        rep=RepresentationConfig(name="mask", params={}),
        backbone=BackboneConfig(size="small", in_ch=3),
        data=DatasetConfig(root="data_sample/SampleDataSet", split_name="SampleDataSet"),
        augs=AugmentationConfig(flip_x=True, flip_y=True, rot90=True,
                                affine=None, photometric=photometric),
        optimizer=OptimizerConfig(lr=3e-4, wd=1e-5, epochs=1),
        train=TrainConfig(
            seed_everything=0,
            patch=256,
            batch_size=2,
            val_batch_size=2,
            num_workers=0,
            run_dir=run_dir,
            model_name="cfg",
            save_top_k=1,
            ckpt_every_n_epochs=1,
            log_every_n_steps=1,
            test_after_training=False,
            debug=True,
            monitor="val_ap75",
        ),
    )

    saved = yaml.safe_load(open(
        next(Path(run_dir).rglob("config.yaml")), encoding="utf-8"))

    # every section rebuilds into its config class — photometric is pure
    # data, so the config round-trips with no special-casing
    train = TrainConfig(**saved['train'])
    augs = AugmentationConfig(**saved['augs'])
    dataset = DatasetConfig(**saved['dataset'])
    optimizer = OptimizerConfig(**saved['optimizer'])
    backbone = BackboneConfig(**saved['backbone'])
    # scalar sections round-trip exactly (model_name/run_name are filled in
    # at train time, so compare against the dump)
    assert saved['train'] == train.model_dump()
    assert train.patch == 256 and train.batch_size == 2
    assert train.seed_everything == 0 and train.debug is True
    assert train.monitor == "val_ap75"
    assert optimizer.epochs == 1 and optimizer.warmup_epochs == 5
    assert augs.flip_x and augs.flip_y and augs.rot90
    assert augs.affine is None

    # photometric specs round-trip verbatim as data; instantiation recovers
    # concrete classes and fields
    assert augs.photometric == saved['augs']['photometric']
    loaded = load_photometric(augs.photometric)
    assert len(loaded) == len(photometric)
    for spec, rebuilt in zip(photometric, loaded):
        original = load_photometric([spec])[0]
        assert type(rebuilt) is type(original)
        assert original.__dict__ == rebuilt.__dict__, spec

    # unknown op names are dropped, not fatal
    assert load_photometric([{"op": "NotAnOp", "p": 0.2}]) == []
    # legacy class_path entries still load (old run dirs)
    legacy = load_photometric(
        [{"class_path": "fluometrik.Blur", "init_args": {"p": 0.4}}])
    assert [type(o).__name__ for o in legacy] == ["Blur"]


def test_eval_returns_labels(tmp_path):
    import numpy as np

    from pluriseg.model import UnifedSegmentationModel

    model = UnifedSegmentationModel(
        rep_cfg=RepresentationConfig(name="mask", params={}),
        backbone_cfg=BackboneConfig(size="small", in_ch=3),
        optimizer_cfg=OptimizerConfig(lr=3e-4, wd=1e-5, epochs=2),
    )
    model.train()
    img = torch.rand(3, 64, 64)
    with pytest.warns(UserWarning):
        labels = model.predict_labels(img)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (64, 64)
    assert labels.dtype == np.uint32
    # train mode is preserved (caller owns eval/train state)
    assert model.training

    # no warning when properly in eval mode
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        labels = model.predict_labels(img)
