"""
tracking/export_onnx_mha.py

Exports the xattn MHA correlation variant of LightTrack for DFC parsing / op support check.
No checkpoint is loaded — weights are random. This is purely a graph structure test.

Differences from export_onnx.py:
  - feature_fusor.pw_corr is replaced with PWCA(corr_type='xattn') after model build
  - No checkpoint loading (not needed for DFC parse)
  - Verification disabled (no reference weights to compare against)

Correlation shape flow (xattn) — follows exact Hailo MHA example reshape pattern:
  x: (b, 96, 16, 16) → flatten(2) → (b, 96, 256) → permute(0,2,1) → (b, 256, 96)  [Q — search]
  z: (b, 96,  8,  8) → flatten(2) → (b, 96,  64) → permute(0,2,1) → (b,  64, 96)  [K/V — template]
  Linear Q/K/V projections → MHA → (b, 256, embed_dim)
  permute(0,2,1) → (b, embed_dim, 256) → reshape(b, embed_dim, 16, 16)  ← same H,W as Q input ✓
  Conv2d(embed_dim, 64, 1) → (b, 64, 16, 16)

Graph topology
--------------
  template [1, 3, 128, 128] --> backbone --> BN_z --+
                                                     +--> xattn MHA --> adj --> towers --> cls [1,1,16,16]
  search   [1, 3, 256, 256] --> backbone --> BN_x --+                                 `--> reg [1,4,16,16]
"""

import copy
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _init_paths  # noqa

from lib.models.models import LightTrackM_Speed
from lib.models.connect import PWCA

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT        = "../snapshot/LightTrackM/lighttrack_mha_check.onnx"

PATH_NAME     = "back_04502514044521042540+cls_211000022+reg_100000111_ops_32"
SEARCH_SIZE   = 256
TEMPLATE_SIZE = 128
STRIDE        = 16
ADJ_CHANNEL   = 128

# Must match backbone output channels at the selected DP stage
EMBED_DIM     = 96
# hz*wz = 8*8 = 64 template positions (output channels of correlation)
NUM_KERNEL    = 64
# hx*wx = 16*16 = 256 search positions (adapter projection target)
SEARCH_NUM    = 256

# =============================================================================

TEMPLATE_SHAPE = (1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE)
SEARCH_SHAPE   = (1, 3, SEARCH_SIZE,   SEARCH_SIZE)


class LightTrackSiamese(nn.Module):
    """End-to-end siamese wrapper for ONNX export (MHA variant)."""

    def __init__(self, model: LightTrackM_Speed):
        super().__init__()
        self.backbone_z    = copy.deepcopy(model.features)
        self.backbone_x    = model.features
        self.neck          = model.neck
        self.feature_fusor = model.feature_fusor
        self.head          = model.head

    def forward(self, template: torch.Tensor, search: torch.Tensor):
        zf = self.backbone_z(template)   # [1, 96,  8,  8]
        xf = self.backbone_x(search)     # [1, 96, 16, 16]
        zf, xf = self.neck(zf, xf)
        feat_dict = self.feature_fusor(zf, xf)
        oup = self.head(feat_dict)
        return oup["cls"], oup["reg"]


def build_model() -> LightTrackSiamese:
    print("[1/3] Building model architecture (random weights — graph check only)")
    base = LightTrackM_Speed(
        path_name=PATH_NAME,
        search_size=SEARCH_SIZE,
        template_size=TEMPLATE_SIZE,
        stride=STRIDE,
        adj_channel=ADJ_CHANNEL,
    )

    print("[2/3] Swapping pw_corr -> PWCA(corr_type='xattn')")
    base.feature_fusor.pw_corr = PWCA(
        num_channel=NUM_KERNEL,
        CA=True,
        corr_type='xattn',
        embed_dim=EMBED_DIM,
        search_num=SEARCH_NUM,
    )
    base.eval()

    return LightTrackSiamese(base)


def export(model: LightTrackSiamese):
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT)), exist_ok=True)

    dummy_template = torch.randn(TEMPLATE_SHAPE)
    dummy_search   = torch.randn(SEARCH_SHAPE)

    print(f"[3/3] Tracing and exporting -> {OUTPUT}")
    torch.onnx.export(
        model,
        (dummy_template, dummy_search),
        OUTPUT,
        opset_version=17,
        input_names=["template", "search"],
        output_names=["cls", "reg"],
        dynamic_axes=None,
        do_constant_folding=True,
        verbose=False,
        dynamo=False,
    )
    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"      Saved ({size_mb:.1f} MB) -> {OUTPUT}")


def print_summary():
    print()
    print("=" * 60)
    print("Export complete — xattn MHA DFC parsing check")
    print("=" * 60)
    print(f"  Output file : {OUTPUT}")
    print(f"  Correlation : xattn (flatten(2)+permute → MHA → permute+reshape — exact Hailo example pattern)")
    print(f"  embed_dim   : {EMBED_DIM}  (backbone channels)")
    print(f"  num_kernel  : {NUM_KERNEL}  (correlation output channels, hz*wz template positions)")
    print(f"  search_num  : {SEARCH_NUM}  (hx*wx search positions)")
    print()
    print("Reshape ops — following exact Hailo MHA example:")
    print("  flatten(2)       : spatial dims only (b,C,H,W)→(b,C,H*W) — kept inside MHA fusion context")
    print("  permute(0,2,1)   : transpose (b,C,N)→(b,N,C)              — supported standalone")
    print("  reshape(b,C,H,W) : back to spatial, same H,W as Q input   — inside MHA fusion context")
    print("  Conv2d(96,64,1)  : channel projection to num_channel       — standard conv")
    print()
    print("Next steps:")
    print("  1. Open in Netron — verify MHA appears as a fused attention subgraph")
    print("  2. Run DFC parse:  hailo_sdk_client translate_onnx_model <this .onnx>")
    print("  3. If it fails: DFC only recognises self-attention (Q=K=V source), not cross-attention")


if __name__ == "__main__":
    model = build_model()
    export(model)
    print_summary()
