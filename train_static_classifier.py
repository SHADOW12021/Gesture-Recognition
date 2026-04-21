from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from datetime import datetime

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from PIL import Image

from gesture_pipeline.checkpoints import save_checkpoint
from gesture_pipeline.constants import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from gesture_pipeline.data import HuggingFaceImageDataset, load_hagrid_dataset
from gesture_pipeline.models import create_model

# ============================================================
# DATASET LOCATION NOTE
# ------------------------------------------------------------
# If your HaGRID subset is stored on an external drive, point
# --dataset to that folder when you run the script.
#
# Example Windows command:
# python train_static_classifier.py ^
#   --dataset "E:\\datasets\\hagrid-subset" ^
#   --architecture mobilenet_v3_small ^
#   --run-name cluster_run
#
# The folder you pass should look like this:
# E:\datasets\hagrid-subset\
#   annotations.csv
#   data\
#     train\
#     val\
#     test\
#
# This script reads:
# 1. annotations.csv
# 2. the image files under data\train, data\val, and data\test
#
# So after you redownload the dataset, make sure you pass the
# TOP-LEVEL hagrid-subset directory, not just the data folder.
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a static gesture classifier on HaGRID subset.")
    parser.add_argument("--dataset", default="hagrid-subset", help="Local dataset directory or HF repo id.")
    parser.add_argument("--architecture", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "squeezenet1_1"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output-dir", default="checkpoints", help="Directory where timestamped checkpoints are saved.")
    parser.add_argument("--run-name", default=None, help="Optional prefix for the checkpoint filename.")
    parser.add_argument("--skip-onnx-export", action="store_true", help="Disable automatic ONNX export for this run.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class RandomPixelate:
    def __init__(self, scale_range: tuple[float, float] = (0.2, 0.45), probability: float = 0.25):
        self.scale_range = scale_range
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if torch.rand(1).item() > self.probability:
            return image

        width, height = image.size
        scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
        downsampled_size = (max(8, int(width * scale)), max(8, int(height * scale)))
        image = image.resize(downsampled_size, Image.Resampling.BILINEAR)
        return image.resize((width, height), Image.Resampling.NEAREST)


def build_checkpoint_path(output_dir: str | Path, architecture: str, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = run_name or architecture
    return Path(output_dir) / f"{prefix}_{timestamp}.pt"


def build_log_path(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).with_suffix(".json")


def build_onnx_path(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).with_suffix(".onnx")


def export_to_onnx(model, onnx_path: str | Path, image_size: int, device: str) -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ONNX export requires the 'onnx' package. Install it with: pip install onnx"
        ) from exc

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
    )


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 24, image_size + 24), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.82, 1.0),
                ratio=(0.92, 1.08),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=12,
                        translate=(0.06, 0.06),
                        scale=(0.92, 1.08),
                        shear=(-6, 6),
                        interpolation=InterpolationMode.BILINEAR,
                    )
                ],
                p=0.7,
            ),
            transforms.RandomApply(
                [transforms.RandomPerspective(distortion_scale=0.15, p=1.0, interpolation=InterpolationMode.BILINEAR)],
                p=0.25,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.6))], p=0.25),
            RandomPixelate(scale_range=(0.2, 0.45), probability=0.25),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.1, scale=(0.01, 0.05), ratio=(0.5, 1.8), value="random"),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def run_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += images.size(0)

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += images.size(0)

    return total_loss / total_examples, total_correct / total_examples


def main() -> None:
    args = parse_args()
    export_onnx = not args.skip_onnx_export
    checkpoint_path = build_checkpoint_path(args.output_dir, args.architecture, args.run_name)
    log_path = build_log_path(checkpoint_path)
    onnx_path = build_onnx_path(checkpoint_path)
    dataset = load_hagrid_dataset(args.dataset)
    train_transform, eval_transform = build_transforms(args.image_size)

    if hasattr(dataset, "build_split"):
        train_set = dataset.build_split("train", train_transform)
        val_key = "validation" if "validation" in dataset else "val"
        val_set = dataset.build_split("val" if val_key == "val" else "validation", eval_transform)
        class_names = dataset.class_names
    else:
        train_set = HuggingFaceImageDataset(dataset["train"], train_transform)
        val_key = "validation" if "validation" in dataset else "val"
        val_set = HuggingFaceImageDataset(dataset[val_key], eval_transform)
        class_names = dataset["train"].features["label"].names

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    model = create_model(args.architecture, num_classes=len(class_names)).to(args.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state = None
    best_metrics = {"val_accuracy": 0.0}
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics))

        if val_acc >= best_metrics["val_accuracy"]:
            best_metrics = epoch_metrics
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    model.load_state_dict(best_state)
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        architecture=args.architecture,
        class_names=class_names,
        image_size=args.image_size,
        metrics={"best": best_metrics, "history": history},
    )
    if export_onnx:
        export_to_onnx(model, onnx_path, args.image_size, args.device)
    log_path.write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "onnx_path": str(onnx_path.resolve()) if export_onnx else None,
                "architecture": args.architecture,
                "dataset": str(args.dataset),
                "image_size": args.image_size,
                "class_names": class_names,
                "best": best_metrics,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved best checkpoint to {checkpoint_path.resolve()}")
    if export_onnx:
        print(f"Saved ONNX classifier to {onnx_path.resolve()}")
    print(f"Saved training log to {log_path.resolve()}")


if __name__ == "__main__":
    main()
