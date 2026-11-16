import os

import pytest

# Lightning reads this for reproducible worker RNG; set before any import.
os.environ.setdefault("PL_GLOBAL_SEED", "0")

from pluriseg.configs import (  # noqa: E402
    AugmentationConfig,
    DatasetConfig,
    RepresentationConfig,
)

DATA_ROOT = "data_sample/SampleDataSet"


@pytest.fixture()
def datamodule():
    from pluriseg.data import SegmentationData

    dm = SegmentationData(
        rep_cfg=RepresentationConfig(),
        dataset_cfg=DatasetConfig(root=DATA_ROOT, split_name="SampleDataSet"),
        augs=AugmentationConfig(),
        patch=256,
        batch_size=2,
        val_batch_size=3,
        num_workers=0,
    )
    return dm
