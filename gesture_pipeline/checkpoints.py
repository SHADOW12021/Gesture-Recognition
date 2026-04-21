from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    checkpoint_path: str | Path,
    model,
    architecture: str,
    class_names: list[str],
    image_size: int,
    metrics: dict,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": architecture,
            "class_names": class_names,
            "image_size": image_size,
            "metrics": metrics,
        },
        checkpoint_path,
    )


def load_checkpoint(checkpoint_path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(Path(checkpoint_path), map_location=map_location, weights_only=True)
