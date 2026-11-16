"""Unified CLI: python -m pluriseg.cli fit|eval ...

jsonargparse wires each command's function args; a --config yaml maps 1:1 onto
the signature (see configs/train/base.yaml).
"""


import jsonargparse

from pluriseg.eval.evaluate import eval_
from pluriseg.splits import make_train_test_split
from pluriseg.tools.check_batches import check_batches
from pluriseg.train import run_training
from pluriseg.tools.shape_metrics import compute, plot
from pluriseg.tools.agglomerate_metrics import agglomerate_metrics

COMPONENTS = {
    "fit": run_training,
    "eval": eval_,
    "make_train_test_split": make_train_test_split,
    "check_batches": check_batches,
    "shape_metrics": {
        "compute": compute,
        "plot": plot
    },
    "agglomerate_metrics": agglomerate_metrics
}


def main():
    jsonargparse.auto_cli(
        COMPONENTS,
        as_positional=False
    )


if __name__ == "__main__":
    main()
