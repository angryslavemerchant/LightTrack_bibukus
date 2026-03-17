"""
Simple ONNX Graph Simplifier
Usage: python simplify_onnx.py <input_model.onnx> [output_model.onnx]
Requires: pip install onnx onnxsim
"""

import sys
import onnx
from onnxsim import simplify

def simplify_model(input_path: str, output_path: str) -> None:
    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)

    print("Simplifying...")
    simplified_model, check = simplify(model)

    if not check:
        print("Warning: Simplified model could not be validated.")

    onnx.save(simplified_model, output_path)
    print(f"Saved simplified model to: {output_path}")

    # Print a quick size comparison
    orig_size = sum(t.ByteSize() for t in model.graph.node)
    new_size  = sum(t.ByteSize() for t in simplified_model.graph.node)
    orig_nodes = len(model.graph.node)
    new_nodes  = len(simplified_model.graph.node)
    print(f"Nodes: {orig_nodes} → {new_nodes}  ({orig_nodes - new_nodes} removed)")

if __name__ == "__main__":


    input_path  = "../snapshot/LightTrackM/lighttrack_xattn.onnx"
    output_path = "../snapshot/lightTrackM/lighttrack_cerberus.onnx"

    simplify_model(input_path, output_path)