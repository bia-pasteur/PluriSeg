from pluriseg.configs import RepresentationConfig
from pluriseg.representations.base import Representation
from pluriseg.representations.cellpose import CellposeRep
from pluriseg.representations.chimera import ChimeraRep
from pluriseg.representations.embed import EmbedsegRep
from pluriseg.representations.mask import MaskRep
from pluriseg.representations.omnipose import OmniposeRep
from pluriseg.representations.stardist import StarDistRep
REGISTRY: dict[str, type[Representation]] = {
    MaskRep.name: MaskRep,
    CellposeRep.name: CellposeRep,
    StarDistRep.name: StarDistRep,
    OmniposeRep.name: OmniposeRep,
    EmbedsegRep.name: EmbedsegRep,
    ChimeraRep.name: ChimeraRep,
}


def build_representation(cfg: RepresentationConfig) -> Representation:
    """Build the single Representation instance from its config block.

    cfg: {"name": "stardist", ...remaining keys are constructor kwargs}
    """

    cls = REGISTRY[cfg.name]
    return cls(**cfg.params)
