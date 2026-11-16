"""Generic conv head: backbone feature map F -> one map per representation channel group.

The head topology is driven entirely by `rep.channels`, so the head can never
desync from the representation it serves. Per-representation custom stacks
(e.g. StarDist's radii refinement) can subclass later.
"""

import torch
import torch.nn as nn

from pluriseg.representations.base import ChannelProps, ChannelRole


class ConvHead(nn.Module):
    """3x3 conv stack on F producing sum(rep.channels.values()) channels,
    split into a named dict keyed by rep.channels order."""

    def __init__(self, in_ch: int, channels: dict[str, ChannelProps], hidden: int = 64):
        super().__init__()
        self.pred_channels = dict({k:v for k,v in channels.items() if v.role != ChannelRole.LOSS})
        self.pred_channels_sizes = [c.channels for c in self.pred_channels.values()]
        self.out_names = list(self.pred_channels.keys())
        self.out_ch = sum(self.pred_channels_sizes)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Conv2d(hidden, self.out_ch, 3, padding=1),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(features)
        splits = torch.split(out, self.pred_channels_sizes, dim=1)
        return {name: t for name, t in zip(self.out_names, splits)}


class GroupedConvHead(ConvHead):
    """ConvHead with one conv stack per channel group, all reading the same
    backbone features (the shared trunk).

    groups: {group_name: [channel names]} — names index into `channels`, so
    per-channel props are not repeated. LOSS channels named in a group are
    ignored (consistent with ConvHead's filtering). groups=None (default)
    falls back to ConvHead's single-stack behaviour unchanged. forward()
    returns the same flat dict in both modes, so loss/decode code is
    identical either way.

    Every predicted (non-LOSS) channel must be assigned to exactly one
    group; unassigned channels raise, since silently dropping them would
    break loss() downstream.
    """
    def __init__(self, in_ch: int, channels: dict[str, ChannelProps],
                 groups: dict[str, list[str]] | None = None,
                 hidden: int = 64):
        if groups is None:
            super().__init__(in_ch, channels, hidden=hidden)
            return
        nn.Module.__init__(self)  # skip ConvHead.__init__: this mode owns per-group stacks
        missing = [c for names in groups.values() for c in names if c not in channels]
        if missing:
            raise ValueError(f"group channels not in `channels`: {missing}")
        assigned = {c for names in groups.values() for c in names}
        unassigned = [c for c in channels
                      if channels[c].role != ChannelRole.LOSS and c not in assigned]
        if unassigned:
            raise ValueError(
                f"predicted channels not assigned to any group: {unassigned}")
        self.groups = nn.ModuleDict({
            gname: ConvHead(in_ch,
                            {c: channels[c] for c in names},
                            hidden=hidden)
            for gname, names in groups.items()
        })

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if not hasattr(self, "groups"):
            return super().forward(features)
        return {name: t for head in self.groups.values()
                for name, t in head(features).items()}
