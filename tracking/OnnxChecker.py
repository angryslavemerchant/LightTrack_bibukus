"""
check_ops.py  –  List all ops in a LightTrack ONNX graph and flag anything
                 that may not be supported by the Hailo-8 compiler.

Usage: just set ONNX_PATH below and hit Run.
"""

import os
from collections import Counter

import onnx

# ── EDIT THIS ─────────────────────────────────────────────────────────────────
ONNX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "snapshot/LightTrackM/lighttrack_siamese.onnx")
# ──────────────────────────────────────────────────────────────────────────────

# Hailo-8 supported ops (from Hailo Model Zoo / Dataflow Compiler docs)
# Not exhaustive — covers the common EfficientNet/MobileNet family ops
HAILO_SUPPORTED = {
    "Conv", "ConvTranspose", "Gemm", "MatMul",
    "Relu", "Sigmoid", "Tanh", "HardSigmoid", "HardSwish",
    "Mul", "Add", "Sub", "Div",
    "BatchNormalization",
    "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool",
    "Reshape", "Flatten", "Transpose", "Squeeze", "Unsqueeze", "Concat", "Slice",
    "Pad", "Resize",
    "Softmax", "LogSoftmax",
    "Clip",                 # used for ReLU6
    "Shape", "Gather",      # often foldable as constants
    "Cast", "Identity",
    "Exp", "Log", "Sqrt",
    "ReduceMean", "ReduceMax", "ReduceSum",
}

# Ops that appear in the graph but get constant-folded away at compile time
# (not a real concern, but flag them as informational)
TYPICALLY_FOLDED = {"Shape", "Gather", "Cast", "Unsqueeze", "Concat", "Identity"}


def check(onnx_path: str):
    print(f"Loading {onnx_path}\n")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    op_counts = Counter(node.op_type for node in model.graph.node)
    all_ops   = sorted(op_counts.keys())

    unsupported   = [op for op in all_ops if op not in HAILO_SUPPORTED]
    needs_check   = [op for op in all_ops if op in TYPICALLY_FOLDED]
    clean         = [op for op in all_ops if op in HAILO_SUPPORTED and op not in TYPICALLY_FOLDED]

    # ── full op list ──────────────────────────────────────────────────────────
    print(f"{'OP TYPE':<30} {'COUNT':>6}   STATUS")
    print("─" * 55)
    for op in all_ops:
        if op in unsupported:
            status = "❌  NOT in Hailo supported list"
        elif op in needs_check:
            status = "ℹ️   typically constant-folded"
        else:
            status = "✅"
        print(f"  {op:<28} {op_counts[op]:>6}   {status}")

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print(f"  Total unique ops : {len(all_ops)}")
    print(f"  Clean            : {len(clean)}")
    print(f"  Typically folded : {len(needs_check)}")
    print(f"  Needs review     : {len(unsupported)}")
    print("=" * 55)

    if unsupported:
        print("\nOps to investigate:")
        for op in unsupported:
            print(f"  {op}  (x{op_counts[op]})")
        print("\nCheck https://hailo.ai/developer-zone/ for your SDK version's")
        print("supported op list — it expands with each release.")
    else:
        print("\nAll ops are in the Hailo-8 supported list.")
        print("Run through hailo_sdk_client to confirm — the compiler is the")
        print("final authority, especially for specific attribute combinations.")


if __name__ == "__main__":
    if not os.path.isfile(ONNX_PATH):
        print(f"ERROR: ONNX file not found at:\n  {ONNX_PATH}")
        print("Update ONNX_PATH at the top of this file.")
    else:
        check(ONNX_PATH)