"""
tracking/export_onnx.py

Place this file in the tracking/ directory (same level as FLOPs_Params.py).
Run directly from your IDE — no terminal arguments needed, just edit the
CONFIG block below and press Run.

Graph topology
--------------
  template [1, 3, 128, 128] --> backbone --> BN_z --+
                                                     +--> pixel_corr --> adj --> towers --> cls [1,1,16,16]
  search   [1, 3, 256, 256] --> backbone --> BN_x --+                                 `--> reg [1,4,16,16]

Both branches share backbone weights. The tracer expands this into two
parallel paths in the ONNX graph with all shapes baked in.
"""

import copy
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# ── always resolves correctly when run from tracking/ or repo root ────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _init_paths  # noqa: registers lib/ on sys.path

from lib.models.models import LightTrackM_Speed
from lib.utils.utils import load_pretrain

# =============================================================================
# CONFIG — edit these, then press Run
# =============================================================================

SNAPSHOT    = "../snapshot/LightTrackM/LightTrackM.pth"   # path to checkpoint
OUTPUT      = "../snapshot/LightTrackM/lighttrack_siamese.onnx"  # where to save

# Architecture — matches the published LightTrackM checkpoint
PATH_NAME   = "back_04502514044521042540+cls_211000022+reg_100000111_ops_32"
SEARCH_SIZE   = 256
TEMPLATE_SIZE = 128
STRIDE        = 16
ADJ_CHANNEL   = 128

# Set True to numerically diff PyTorch vs ONNX Runtime after export
# (requires:  pip install onnxruntime)
VERIFY = True

# =============================================================================


TEMPLATE_SHAPE = (1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE)   # [1, 3, 128, 128]
SEARCH_SHAPE   = (1, 3, SEARCH_SIZE,   SEARCH_SIZE)     # [1, 3, 256, 256]


# ── model wrapper ─────────────────────────────────────────────────────────────

class LightTrackSiamese(nn.Module):
    """
    Full end-to-end siamese wrapper for ONNX export.

    Inputs
        template : [1, 3, 128, 128]  normalised RGB template patch
        search   : [1, 3, 256, 256]  normalised RGB search patch

    Outputs
        cls      : [1, 1,  16,  16]  score heatmap
        reg      : [1, 4,  16,  16]  ltrb offset map
    """

    def __init__(self, model: LightTrackM_Speed):
        super().__init__()
        # Two separate backbone instances with identical weights.
        #
        # Calling the same nn.Module twice in a traced forward causes PyTorch's
        # ONNX exporter to emit  weight / weight_1  pairs where weight_1 is not
        # registered as an initializer — ONNX Runtime and Hailo both reject this.
        # Deepcopying gives each branch its own uniquely-named weight tensors so
        # the graph is fully valid, and produces two clean parallel subgraphs
        # which is exactly what Hailo's compiler wants to see.
        self.backbone_z    = copy.deepcopy(model.features)   # template branch
        self.backbone_x    = model.features                  # search branch
        self.neck          = model.neck            # BN_adj  (BN_z + BN_x)
        self.feature_fusor = model.feature_fusor   # pixel_corr_mat + adj 1x1 conv
        self.head          = model.head            # cls/reg towers + pred heads

    def forward(self, template: torch.Tensor, search: torch.Tensor):
        # Two independent backbone passes — identical weights, different spatial sizes
        zf = self.backbone_z(template)   # [1, 96,  8,  8]
        xf = self.backbone_x(search)     # [1, 96, 16, 16]

        # Per-branch BN before correlation
        zf, xf = self.neck(zf, xf)

        # Pixel-wise correlation + channel unification
        feat_dict = self.feature_fusor(zf, xf)

        # Prediction towers
        oup = self.head(feat_dict)

        # Return tuple — ONNX can't serialise dicts
        return oup["cls"], oup["reg"]


# ── steps ─────────────────────────────────────────────────────────────────────

def build_model() -> LightTrackSiamese:
    print("[1/4] Building model architecture")
    base = LightTrackM_Speed(
        path_name=PATH_NAME,
        search_size=SEARCH_SIZE,
        template_size=TEMPLATE_SIZE,
        stride=STRIDE,
        adj_channel=ADJ_CHANNEL,
    )

    print(f"[2/4] Loading checkpoint: {SNAPSHOT}")
    if not os.path.isfile(SNAPSHOT):
        raise FileNotFoundError(
            f"Checkpoint not found: '{SNAPSHOT}'\n"
            f"Download LightTrackM.pth and update SNAPSHOT in this file."
        )
    load_pretrain(base, SNAPSHOT, print_unuse=False)
    base.eval()

    return LightTrackSiamese(base)


def export(model: LightTrackSiamese):
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT)), exist_ok=True)

    dummy_template = torch.zeros(TEMPLATE_SHAPE)
    dummy_search   = torch.zeros(SEARCH_SHAPE)

    print(f"[3/4] Tracing and exporting -> {OUTPUT}")
    torch.onnx.export(
        model,
        (dummy_template, dummy_search),
        OUTPUT,
        opset_version=18,
        input_names=["template", "search"],
        output_names=["cls", "reg"],
        dynamic_axes=None,         # fully static shapes — required for Hailo
        do_constant_folding=True,  # bakes BN + pool kernel sizes into graph
        verbose=False,
    )
    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"      Saved ({size_mb:.1f} MB) -> {OUTPUT}")


def verify(model: LightTrackSiamese):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[4/4] Skipping verification — install onnxruntime to enable")
        return

    print("[4/4] Verifying: PyTorch outputs vs ONNX Runtime outputs")

    rng       = np.random.default_rng(42)
    np_tmpl   = rng.standard_normal(TEMPLATE_SHAPE).astype(np.float32)
    np_search = rng.standard_normal(SEARCH_SHAPE).astype(np.float32)

    # PyTorch reference
    with torch.no_grad():
        pt_cls, pt_reg = model(torch.from_numpy(np_tmpl), torch.from_numpy(np_search))

    # ONNX Runtime
    sess = ort.InferenceSession(OUTPUT, providers=["CPUExecutionProvider"])
    ort_cls, ort_reg = sess.run(None, {"template": np_tmpl, "search": np_search})

    diff_cls = float(np.abs(pt_cls.numpy() - ort_cls).max())
    diff_reg = float(np.abs(pt_reg.numpy() - ort_reg).max())

    print(f"      cls  max|delta| = {diff_cls:.2e}")
    print(f"      reg  max|delta| = {diff_reg:.2e}")

    if diff_cls < 1e-4 and diff_reg < 1e-4:
        print("      OK  outputs match")
    else:
        print("      WARN  diff above threshold — open in Netron and check for unsupported ops")


def print_summary():
    print()
    print("=" * 50)
    print("Export complete")
    print("=" * 50)
    print(f"  Output file   : {OUTPUT}")
    print(f"  template in   : {list(TEMPLATE_SHAPE)}")
    print(f"  search in     : {list(SEARCH_SHAPE)}")
    print(f"  cls out       : [1, 1, 16, 16]")
    print(f"  reg out       : [1, 4, 16, 16]")
    print()
    print("Hailo next steps:")
    print("  1. Open in Netron — confirm two parallel backbone paths and no dynamic shapes")
    print("  2. hailo_sdk_client: parse -> optimize -> quantize (use real patches for calib data)")
    print("  3. Compile to HEF")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model()
    export(model)

    if VERIFY:
        verify(model)
    else:
        print("[4/4] Verification disabled (set VERIFY = True to enable)")

    print_summary()