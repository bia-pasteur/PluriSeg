import pandas as pd
from pydantic import BaseModel, Field

from pluriseg.shapes.features import FEATURE_COLUMNS


class ShapeRule(BaseModel):
    """One rule-based category: per-feature (lo, hi) bounds, None = open."""

    name: str
    # feature column -> (lo, hi) inclusive bounds; None = unbounded.
    features: dict[str, tuple[float | None, float | None]] = Field(
        default_factory=dict
    )


class RuleShapeCategorizer:
    """Assign categories by first matching rule (order of `rules`)."""

    def __init__(self, rules: list[ShapeRule], default: str = "other"):
        for rule in rules:
            unknown = set(rule.features) - set(FEATURE_COLUMNS)
            if unknown:
                raise ValueError(
                    f"unknown features in rule: {sorted(unknown)}")
        self.rules = rules
        self.default = default

    def categorize(self, feats: pd.DataFrame) -> pd.Series:
        """Category per row; first matching rule wins, else `default`."""
        out = pd.Series(self.default, index=feats.index, dtype=object)
        unassigned = pd.Series(True, index=feats.index)
        for rule in self.rules:
            mask = pd.Series(True, index=feats.index)
            for col, (lo, hi) in rule.features.items():
                values = feats.loc[:, col]
                if lo is not None:
                    mask &= values >= lo
                if hi is not None:
                    mask &= values <= hi
            take = unassigned & mask
            out.loc[take] = rule.name
            unassigned.loc[take] = False
        return out