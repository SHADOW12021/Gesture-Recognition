from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gesture_pipeline.checkpoints import load_checkpoint
from gesture_pipeline.models import create_model
from train_static_classifier import export_to_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained gesture classifier checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint created by train_static_classifier.py")
    parser.add_argument("--output", default=None, help="Optional ONNX output path. Defaults next to the checkpoint.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location=args.device)
    model = create_model(checkpoint["architecture"], len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output) if args.output else checkpoint_path.with_suffix(".onnx")
    metadata_path = output_path.with_suffix(".json")

    export_to_onnx(model, output_path, checkpoint["image_size"], args.device)
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "onnx_path": str(output_path.resolve()),
                "architecture": checkpoint["architecture"],
                "image_size": checkpoint["image_size"],
                "class_names": checkpoint["class_names"],
                "metrics": checkpoint.get("metrics", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved ONNX classifier to {output_path.resolve()}")
    print(f"Saved classifier metadata to {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
