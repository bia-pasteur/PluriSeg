"""Shared model/datamodule construction for the fit and eval entrypoints."""

from pluriseg.configs import (AugmentationConfig, BackboneConfig, DatasetConfig,
                            GateConfig, OptimizerConfig, RepresentationConfig,
                            TrainConfig)
from pluriseg.data import SegmentationData
from pluriseg.model import UnifedSegmentationModel


def build_model_and_dm(
    rep: RepresentationConfig,
    backbone: BackboneConfig,
    optimizer: OptimizerConfig,
    data: DatasetConfig,
    augs: AugmentationConfig,
    train: TrainConfig,
    timing_log_dir: str = "",
    gate_cfg: GateConfig | None = None,
):
    """Build the Lightning model and datamodule from the shared config blocks."""
    # EmbedsegRep normalizes offsets by 1/train_patch; chimera wraps it, so the
    # patch must reach its sub_params too.
    if rep.name == "embedseg":
        rep.params["train_patch"] = train.patch
    if rep.name == "chimera":
        rep.params.setdefault("sub_params", {}).setdefault(
            "embedseg", {})["train_patch"] = train.patch

    if data.in_ch and data.in_ch != backbone.in_ch:
        raise ValueError(
            f"data.in_ch={data.in_ch} (channel normalization) must match "
            f"backbone.in_ch={backbone.in_ch}, or be 0 (off)")

    model = UnifedSegmentationModel(
        rep_cfg=rep,
        backbone_cfg=backbone,
        optimizer_cfg=optimizer,
        gate_cfg=gate_cfg,
    )
    model.debug = train.debug

    dm = SegmentationData(
        dataset_cfg=data,
        augs=augs,
        rep_cfg=rep,
        patch=train.patch,
        batch_size=train.batch_size,
        val_batch_size=train.val_batch_size,
        num_workers=train.num_workers,
        timing_train_log_dir=timing_log_dir,
    )
    return model, dm
