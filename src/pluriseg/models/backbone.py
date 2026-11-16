"""Shared backbone: nnU-Net-style Residual Encoder UNet exposing a raw feature map.

Wraps `dynamic_network_architectures.architectures.unet.ResidualEncoderUNet`.
The decoder's 1x1 `seg_layers` (normally producing segmentation logits) are
replaced by Identity so `forward(x) -> F` returns the raw full-resolution
feature map that every representation head consumes.
"""

import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import (
    get_matching_convtransp,
    get_matching_instancenorm,
)
from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
    StackedConvBlocks,
)

from pluriseg.configs import BackboneConfig

# Size presets, following nnU-Net v2's ResEnc presets (same topology across
# sizes: nnU-Net's M/L differ only in VRAM-target patch/batch sizing, which
# lives in the training config here, not in the network).
_PRESETS: dict[str, dict] = {
    # widest-stem budget presets: keep the 64-ch full-res stem the heads
    # consume, shrink the deep stages (cellpose-3 wins with ~6.6M params)
    "tiny2": {
        "n_stages": 6,
        "features_per_stage": [64, 128, 192, 256, 256, 256],
        "n_blocks_per_stage": [1, 2, 2, 2, 2, 2],
    },
    "small": {
        "n_stages": 6,
        "features_per_stage": [32, 64, 128, 256, 512, 512],
        "n_blocks_per_stage": [1, 2, 2, 2, 2, 2],
    },
    "small2": {
        "n_stages": 6,
        "features_per_stage": [64, 128, 256, 512, 512, 512],
        "n_blocks_per_stage": [1, 2, 2, 2, 2, 2],
    },
    "small-64": {
        "n_stages": 6,
        "features_per_stage": [32, 128, 256, 512, 512, 512],
        "n_blocks_per_stage": [1, 2, 2, 2, 2, 2],
        "decoder_out_features": 64,
    },
    "small-128": {
        # 32-ch stem (cheap full-res encoder), 128-ch decoder output: the
        # final decoder stage is rebuilt at 128 wide (the width the
        # representation heads and the gate router consume) so the
        # multi-task heads get real 128-ch capacity without a fat encoder
        # stem.
        "n_stages": 6,
        "features_per_stage": [32, 128, 256, 512, 512, 512],
        "n_blocks_per_stage": [1, 2, 2, 2, 2, 2],
        "decoder_out_features": 128,
    },

    "medium": {
        "n_stages": 6,
        "features_per_stage": [32, 64, 128, 256, 512, 512],
        "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
    },
    # medium topology with a widened decoder stem (64 full-res channels for
    # the representation heads); small/medium stay pristine for old ckpts
    "medium2": {
        "n_stages": 6,
        "features_per_stage": [64, 128, 256, 512, 512, 512],
        "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
    },
    # large topology (more encoder blocks) with the same widened stem
    "large2": {
        "n_stages": 6,
        "features_per_stage": [64, 128, 256, 512, 512, 512],
        "n_blocks_per_stage": [1, 3, 6, 8, 8, 8],
    },
    "large": {
        "n_stages": 6,
        "features_per_stage": [32, 64, 128, 256, 512, 512],
        "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
    },
}


def build_backbone(cfg: BackboneConfig) -> nn.Module:
    """Build the shared backbone returning F (B, decoder_out_features, H, W).

    F is the raw full-resolution decoder feature map: by default its width
    is features_per_stage[0]. A preset may
    set `decoder_out_features` to widen that final decoder stage (its
    transposed conv and residual blocks rebuilt at the new width — the
    vendor file is untouched); that width is what every representation
    head and the gate router consume. No init or other overrides — the
    library defaults stand.

    cfg keys:
        in_ch: input channels (default 1)
        size: one of _PRESETS keys (default "small")
    """
    size = cfg.size
    in_ch = cfg.in_ch

    arch = _PRESETS[size]
    stem_ch = arch["features_per_stage"][0]
    out_ch = arch.get("decoder_out_features", stem_ch)
    n_conv_decoder = 2  # matches n_conv_per_stage_decoder below
    net = ResidualEncoderUNet(
        input_channels=in_ch,
        n_stages=arch["n_stages"],
        features_per_stage=arch["features_per_stage"],
        conv_op=nn.Conv2d,
        kernel_sizes=[3] * arch["n_stages"],
        strides=[1] + [2] * (arch["n_stages"] - 1),
        n_blocks_per_stage=arch["n_blocks_per_stage"],
        num_classes=1,  # placeholder; seg_layers are replaced by Identity below
        n_conv_per_stage_decoder=[2] * (arch["n_stages"] - 1),
        conv_bias=True,
        norm_op=get_matching_instancenorm(nn.Conv2d),
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True, "negative_slope": 0.01},
        deep_supervision=False,
    )
    # Expose the raw decoder features: replace every 1x1 seg head with
    # Identity. When the preset widens the decoder output, rebuild the
    # final (full-res) decoder stage at out_ch: the transposed conv
    # upsample to out_ch, then blocks take concat(out_ch, stem skip) and
    # output out_ch (mirrors the library's own stage construction).
    for i in range(len(net.decoder.seg_layers)):
        net.decoder.seg_layers[i] = nn.Identity()
    if out_ch != stem_ch:
        dec, enc = net.decoder, net.encoder
        t = dec.transpconvs[-1]
        dec.transpconvs[-1] = get_matching_convtransp(conv_op=enc.conv_op)(
            t.in_channels, out_ch, t.stride[0], t.stride[0],
            bias=enc.conv_bias)
        # concat(out_ch upsampled, stem_ch skip) -> out_ch
        dec.stages[-1] = StackedConvBlocks(
            n_conv_decoder, enc.conv_op, out_ch + stem_ch, out_ch,
            enc.kernel_sizes[0], 1, enc.conv_bias, enc.norm_op,
            enc.norm_op_kwargs, enc.dropout_op, enc.dropout_op_kwargs,
            enc.nonlin, enc.nonlin_kwargs)

    return net, out_ch
