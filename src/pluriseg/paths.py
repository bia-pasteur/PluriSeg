"""Well-known directory paths used across the repo.

All paths are relative to the current working directory. 
Import these instead of hardcoding "images", "runs", "pluriseg-cache", etc.
"""

from pathlib import Path

#: project root (parent of the src/pluriseg package)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: training runs (checkpoints, configs, metrics) — TrainConfig.run_dir
RUNS = PROJECT_ROOT / "runs"

#: all figures/visualizations, subdivided per purpose
RESULTS = PROJECT_ROOT / "results"
FIGURES_RESULTS = RESULTS / "predictions"  # prediction visualizations
RESULTS_METRICS = RESULTS / "metrics"  # metric curves

#: shared caches
CACHE = PROJECT_ROOT / "pluriseg-cache"
CACHE_SHAPES = CACHE / "shapes"

#: dataset split files
SPLITS = PROJECT_ROOT / "data-splits"

DEFAULT_ROOT_MAP = {
    "*": "data/Datasets",
    "cis_cellpose": "data/cellpose",
    "cis_dsb2018": "data/dsb2018",
    "cis_livecell": "data/livecell",
    "cis_omnipose": "data/omnipose",
    "cis_tissuenet": "data/tissuenet",
}