"""Per-head eval for chimera runs: predict_labels_all_heads, per-head CSVs,
metrics_complete chimera requirement, agglomerate/to_latex head rows."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from pluriseg.configs import (
    AugmentationConfig,
    BackboneConfig,
    DatasetConfig,
    OptimizerConfig,
    RepresentationConfig,
)

DATA_ROOT = "data_sample/SampleDataSet"
CHIMERA_SUBS = ["mask", "cellpose", "omnipose", "stardist", "embedseg"]
REQUIRED_COLS = [
    "AP@0.50", "SQ", "RQ", "PQ",
]


def _chimera_model() -> "UnifedSegmentationModel":
    from pluriseg.model import UnifedSegmentationModel

    return UnifedSegmentationModel(
        rep_cfg=RepresentationConfig(name="chimera", params={}),
        backbone_cfg=BackboneConfig(size="small", in_ch=3),
        optimizer_cfg=OptimizerConfig(lr=3e-4, wd=1e-5, epochs=2),
    )


def _metrics_csv(image_sources: list[str], values: dict | None = None
                 ) -> pd.DataFrame:
    """A minimal metrics CSV frame: image_source + required metric columns
    (values default to 1.0, overridable per column)."""
    values = values or {}
    return pd.DataFrame([{
        "image_source": src,
        "n_gt": 3,
        "n_pred": 3,
        **{c: values.get(c, 1.0) for c in
           ["AP@0.25", "AP@0.50", "AP@0.75", "AP@[.5:.95]",
            "matched_iou@0", "matched_iou@0.5", "SQ", "RQ", "PQ"]},
    } for src in image_sources])


def test_predict_labels_all_heads_keys():
    from pluriseg.model import UnifedSegmentationModel

    model = _chimera_model().eval()
    decodes = model.predict_labels_all_heads(torch.rand(3, 64, 64))
    assert list(decodes) == [""] + CHIMERA_SUBS
    for labels in decodes.values():
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (64, 64)
        assert labels.dtype == np.uint32

    # a plain rep has no sub-representations: only the default decode
    mask_model = UnifedSegmentationModel(
        rep_cfg=RepresentationConfig(name="mask", params={}),
        backbone_cfg=BackboneConfig(size="small", in_ch=3),
        optimizer_cfg=OptimizerConfig(lr=3e-4, wd=1e-5, epochs=2),
    ).eval()
    decodes = mask_model.predict_labels_all_heads(torch.rand(3, 64, 64))
    assert list(decodes) == [""]


def test_evaluate_head_rows(tmp_path):
    from pluriseg.data import SegmentationData
    from pluriseg.eval.evaluate import evaluate

    model = _chimera_model().eval()
    dm = SegmentationData(
        rep_cfg=RepresentationConfig(name="chimera", params={}),
        dataset_cfg=DatasetConfig(root=DATA_ROOT, split_name="SampleDataSet"),
        augs=AugmentationConfig(),
        patch=256,
        batch_size=2,
        val_batch_size=3,
        num_workers=0,
    )
    dm.setup("test")
    n_test = len(dm.test_ds)

    evaluate(model, dm, out_dir=tmp_path / "eval", enable_progress_bar=False)

    base = pd.read_csv(tmp_path / "eval" / "metrics.csv")
    # base CSV schema unchanged: one row per image + mean, no head column
    assert len(base) == n_test + 1
    assert base["image_source"].iloc[-1] == "mean"
    assert "head" not in base.columns
    for col in REQUIRED_COLS:
        assert col in base.columns

    # one per-head CSV per sub, same schema, one mean row per file
    for sname in CHIMERA_SUBS:
        hcsv = tmp_path / "eval" / f"metrics_head_{sname}.csv"
        assert hcsv.exists(), sname
        hdf = pd.read_csv(hcsv)
        assert len(hdf) == n_test + 1, sname
        assert hdf["image_source"].iloc[-1] == "mean", sname
        for col in REQUIRED_COLS:
            assert col in hdf.columns, (sname, col)


def _write_run(run_dir: Path, rep: dict, rows: dict[str, pd.DataFrame],
               cfg: dict | None = None):
    """Fabricate a run dir: config.yaml + eval/<name>.csv files."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(exist_ok=True)
    payload = {
        "representation": rep,
        "dataset": {"root": "", "split_name": "DSB2018", "fold": 0},
    }
    if cfg:
        payload.update(cfg)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(payload),
                                         encoding="utf-8")
    for name, df in rows.items():
        df.to_csv(run_dir / "eval" / name, index=False)


def _pq_csv(images: dict[str, float]) -> pd.DataFrame:
    """Per-image PQ-only frame (gate-accuracy join input)."""
    return pd.DataFrame([{"image_source": src, "PQ": pq}
                         for src, pq in images.items()])


def _gated_csv(images: list[tuple[str, float, str]]) -> pd.DataFrame:
    """Per-image (source, PQ, gate_pick) frame as evaluate() writes it."""
    return pd.DataFrame([{"image_source": src, "PQ": pq, "gate_pick": pick}
                         for src, pq, pick in images])


def test_metrics_complete_head_column(tmp_path):
    from pluriseg.eval.evaluate import metrics_complete

    # chimera run without per-head CSVs: incomplete (forces re-eval)
    run = tmp_path / "chimera_old"
    _write_run(
        run, {"name": "chimera", "params": {"sub_params":
                                            {"mask": {}, "cellpose": {}}}},
        {"metrics.csv": _metrics_csv(["a", "mean"])})
    assert not metrics_complete(run)

    # complete per-head CSVs: complete
    _write_run(
        tmp_path / "chimera_new",
        {"name": "chimera", "params": {"sub_params":
                                       {"mask": {}, "cellpose": {}}}},
        {"metrics.csv": _metrics_csv(["a", "mean"]),
         "metrics_head_mask.csv": _metrics_csv(["a", "mean"]),
         "metrics_head_cellpose.csv": _metrics_csv(["a", "mean"])})
    assert metrics_complete(tmp_path / "chimera_new")

    # per-head CSV present but missing required columns: incomplete
    _write_run(
        tmp_path / "chimera_partial",
        {"name": "chimera", "params": {"sub_params":
                                       {"mask": {}, "cellpose": {}}}},
        {"metrics.csv": _metrics_csv(["a", "mean"]),
         "metrics_head_mask.csv": _metrics_csv(["a", "mean"]),
         "metrics_head_cellpose.csv": pd.DataFrame([{"image_source": "a"}])})
    assert not metrics_complete(tmp_path / "chimera_partial")

    # non-chimera run: base CSV only, unaffected
    _write_run(tmp_path / "cellpose", {"name": "cellpose", "params": {}},
               {"metrics.csv": _metrics_csv(["a", "mean"])})
    assert metrics_complete(tmp_path / "cellpose")


def test_agglomerate_head_qualification(tmp_path):
    from pluriseg.tools.agglomerate_metrics import agglomerate, drop_zero_repeats

    zeros = {c: 0.0 for c in
             ["AP@0.50", "SQ", "RQ", "PQ"] + REQUIRED_COLS}
    _write_run(
        tmp_path / "DSB2018-chimera" / "run0",
        {"name": "chimera", "params": {"sub_params":
                                       {"embedseg": {}, "stardist": {}}}},
        {"metrics.csv": _metrics_csv(["a", "b"]),
         "metrics_head_embedseg.csv": _metrics_csv(["a", "b"], zeros),
         "metrics_head_stardist.csv": _metrics_csv(["a", "b"])})
    _write_run(
        tmp_path / "DSB2018-mask" / "run0",
        {"name": "mask", "params": {}},
        {"metrics.csv": _metrics_csv(["a", "b"])})

    agg = agglomerate("*/*", runs=tmp_path)
    reps = set(agg["rep"])
    assert {"mask", "chimera", "chimera/embedseg", "chimera/stardist"} <= reps
    assert "head" not in agg.columns
    # both repeat axes are emitted whatever the mode
    assert {"fold", "seed"} <= set(agg.columns)

    # collapsed chimera/embedseg repeat is dropped; chimera/stardist survives
    kept = set(drop_zero_repeats(agg)["rep"])
    assert "chimera/embedseg" not in kept
    assert "chimera/stardist" in kept


def test_gate_accuracy_margin_and_picks(tmp_path):
    from pluriseg.tools.agglomerate_metrics import (
        GATE_ACC_REP, gate_selection_accuracy)

    def gated_run(fold: int, images: list[tuple[str, float, str]],
                  head_pq: dict[str, dict[str, float]]):
        _write_run(
            tmp_path / f"DSB2018-chimera" / f"run{fold}",
            {"name": "chimera", "params": {"sub_params":
                                           {"mask": {}, "stardist": {}}}},
            {"metrics.csv": _gated_csv(images),
             **{f"metrics_head_{h}.csv": _pq_csv(v)
                for h, v in head_pq.items()}},
            cfg={"gate": {"enabled": True},
                 "dataset": {"root": "", "split_name": "DSB2018",
                             "fold": fold},
                 "train": {"seed_everything": fold}})

    # f0/s0: a ties the best head exactly; b is within HEAD_PERF_MARGIN of
    # the best (both count as correct); picks split mask/stardist
    gated_run(0, [("a", 0.9, "mask"), ("b", 0.98, "stardist")],
              {"mask": {"a": 0.9, "b": 1.0}, "stardist": {"a": 0.3, "b": 0.5}})
    # f1/s1: c misses the best head by more than the margin; pick stardist
    gated_run(1, [("c", 0.5, "stardist")],
              {"mask": {"c": 1.0}, "stardist": {"c": 0.4}})

    # both repeat axes must produce the same two-row picture (the runs
    # carry matching fold and seed ids)
    for repeats, rcol in (("seeds", "seed"), ("folds", "fold")):
        acc = gate_selection_accuracy(runs=tmp_path, repeats=repeats)
        assert set(acc["rep"]) == {GATE_ACC_REP}
        assert acc.sort_values(rcol)["PQ"].tolist() == [1.0, 0.0]
        # repeat means: a, b correct; c misses; picks split mask/stardist
        assert acc["PQ"].mean() == pytest.approx(0.5)
        assert acc["pick_mask"].mean() == pytest.approx(0.25)
        assert acc["pick_stardist"].mean() == pytest.approx(0.75)

    # agglomerate surfaces the same pick fractions on the chimera rep rows
    from pluriseg.tools.agglomerate_metrics import agglomerate
    agg = agglomerate("*/*", runs=tmp_path)
    chim = agg[agg["rep"] == "chimera"]
    assert chim["pick_mask"].mean() == pytest.approx(0.25)
    assert chim["pick_stardist"].mean() == pytest.approx(0.75)


def test_seeds_mode_aggregation(tmp_path):
    """repeats='seeds' (default): per-seed rows from the run config's
    train.seed_everything; summarize aggregates mean/std over seeds;
    folds mode still collapses by fold."""
    from pluriseg.tools.agglomerate_metrics import agglomerate, summarize

    pq0 = {"PQ": 0.6, "AP@0.50": 0.6}
    pq1 = {"PQ": 0.8, "AP@0.50": 0.8}
    for seed, vals in ((0, pq0), (1, pq1)):
        _write_run(
            tmp_path / "DSB2018-embedseg" / f"s{seed}",
            {"name": "embedseg", "params": {}},
            {"metrics.csv": _metrics_csv(["a", "b"], vals)},
            cfg={"train": {"seed_everything": seed}})

    agg = agglomerate("*/*", runs=tmp_path)
    emb = agg[agg["rep"] == "embedseg"].sort_values("seed")
    assert emb["seed"].tolist() == [0, 1]
    assert emb["fold"].tolist() == [0, 0]
    assert emb["PQ"].tolist() == [0.6, 0.8]

    s = summarize(agg)
    row = s[s["rep"] == "embedseg"].iloc[0]
    assert row["n_repeats"] == 2
    assert row["PQ_mean"] == pytest.approx(0.7)
    assert row["PQ_std"] == pytest.approx(np.std([0.6, 0.8], ddof=1))

    # folds mode: both runs share fold 0 -> one row averaging them
    agg_f = agglomerate("*/*", runs=tmp_path, repeats="folds")
    assert agg_f[agg_f["rep"] == "embedseg"]["fold"].tolist() == [0]
    assert agg_f[agg_f["rep"] == "embedseg"]["PQ"].iloc[0] == \
        pytest.approx(0.7)


def test_gate_post_refit(tmp_path):
    """Fit pays no gate cost (no loss_gate; the router stays random);
    train_selection_head refits the router on the val bank under frozen
    weights."""
    from pluriseg.configs import GateConfig
    from pluriseg.data import SegmentationData
    from pluriseg.model import UnifedSegmentationModel
    from pluriseg.head_selector.train import train_selection_head

    gate = GateConfig(enabled=True, post_epochs=20, post_seeds=1)
    model = UnifedSegmentationModel(
        rep_cfg=RepresentationConfig(name="chimera", params={}),
        backbone_cfg=BackboneConfig(size="small", in_ch=3),
        optimizer_cfg=OptimizerConfig(lr=3e-4, wd=1e-5, epochs=2),
        gate_cfg=gate,
    )
    model.eval()

    # self.log needs an attached trainer; record the logged keys instead
    logged: dict = {}
    model.log = lambda *a, **k: None
    model.log_dict = lambda d, *a, **k: logged.update(d)

    dm = SegmentationData(
        rep_cfg=RepresentationConfig(name="chimera", params={}),
        dataset_cfg=DatasetConfig(root=DATA_ROOT, split_name="SampleDataSet"),
        augs=AugmentationConfig(),
        patch=256,
        batch_size=2,
        val_batch_size=3,
        num_workers=0,
    )
    dm.setup("fit")
    model.training_step(next(iter(dm.train_dataloader())), 0)
    assert "loss_gate" not in logged

    # train_selection_head: whole val images scored under the frozen model,
    # router refit on per-head PQ, loaded back into the model
    stats = train_selection_head(model, dm, gate,
                                 OptimizerConfig(lr=3e-4, wd=1e-5), seed=0)
    assert stats is not None
    assert 0.0 <= stats["routed_pq"] <= 1.0
    assert 0.0 <= stats["best_pq"] <= 1.0
    decodes = model.predict_labels_all_heads(torch.rand(3, 64, 64))
    assert list(decodes) == [""] + CHIMERA_SUBS


