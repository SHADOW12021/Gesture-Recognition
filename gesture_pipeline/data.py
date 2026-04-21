from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


def load_hagrid_dataset(dataset_source: str | Path):
    source_path = Path(dataset_source)
    if source_path.exists():
        data_dir = source_path / "data"
        if data_dir.exists():
            return _LocalImageFolderSplits(data_dir)
        if (source_path / "annotations.csv").exists():
            return _LocalImageFolderSplits(source_path)
        class_dirs = [directory for directory in source_path.iterdir() if directory.is_dir()]
        if class_dirs:
            return _FlatClassFolderSplits(source_path)
        return _LocalImageFolderSplits(source_path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is only needed when loading from Hugging Face Hub. "
            "For local training, pass a local dataset path like 'hagrid-subset'."
        ) from exc

    return load_dataset(str(dataset_source))


class HuggingFaceImageDataset(Dataset):
    def __init__(self, dataset_split, transform: Callable[[Image.Image], object]):
        self.dataset_split = dataset_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset_split)

    def __getitem__(self, index: int):
        sample = self.dataset_split[index]
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        return self.transform(image), sample["label"]


class _LocalImageFolderSplits:
    def __init__(self, root: Path):
        self.root = root
        annotations_path = root.parent / "annotations.csv"
        if not annotations_path.exists():
            raise FileNotFoundError(f"Could not find annotations file: {annotations_path}")

        rows = []
        class_names = set()
        with annotations_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_path = root.parent / row["image_path"]
                if image_path.exists():
                    rows.append(
                        {
                            "image_path": image_path,
                            "label": row["label"],
                            "split": row["split"],
                        }
                    )
                    class_names.add(row["label"])

        if not rows:
            raise FileNotFoundError(f"No valid annotated image files were found from {annotations_path}")

        self.rows = rows
        self.class_names = sorted(class_names)
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}

    def build_split(self, split: str, transform: Callable[[Image.Image], object]):
        normalized_split = "val" if split == "validation" else split
        split_rows = [row for row in self.rows if row["split"] == normalized_split]
        if not split_rows:
            raise FileNotFoundError(f"No annotated samples were found for split '{split}'.")
        return _AnnotatedImageDataset(split_rows, self.class_to_idx, transform)

    def __contains__(self, key: str) -> bool:
        normalized_key = "val" if key == "validation" else key
        return any(row["split"] == normalized_key for row in self.rows)


class _AnnotatedImageDataset(Dataset):
    def __init__(self, rows: list[dict], class_to_idx: dict[str, int], transform: Callable[[Image.Image], object]):
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), self.class_to_idx[row["label"]]


class _FlatClassFolderSplits:
    def __init__(self, root: Path, train_fraction: float = 0.8, val_fraction: float = 0.1, seed: int = 42):
        self.root = root
        self.class_names = sorted([directory.name for directory in root.iterdir() if directory.is_dir()])
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}
        rng = random.Random(seed)

        rows = []
        for class_name in self.class_names:
            class_dir = root / class_name
            files = sorted(
                [
                    path
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                ]
            )
            rng.shuffle(files)
            total = len(files)
            if total == 0:
                continue

            train_end = int(total * train_fraction)
            val_end = int(total * (train_fraction + val_fraction))

            split_map = [
                ("train", files[:train_end]),
                ("val", files[train_end:val_end]),
                ("test", files[val_end:]),
            ]
            for split_name, split_files in split_map:
                for image_path in split_files:
                    rows.append({"image_path": image_path, "label": class_name, "split": split_name})

        if not rows:
            raise FileNotFoundError(f"No image files were found under class folders in {root}")

        self.rows = rows

    def build_split(self, split: str, transform: Callable[[Image.Image], object]):
        normalized_split = "val" if split == "validation" else split
        split_rows = [row for row in self.rows if row["split"] == normalized_split]
        if not split_rows:
            raise FileNotFoundError(f"No samples were found for split '{split}' in {self.root}")
        return _AnnotatedImageDataset(split_rows, self.class_to_idx, transform)

    def __contains__(self, key: str) -> bool:
        normalized_key = "val" if key == "validation" else key
        return any(row["split"] == normalized_key for row in self.rows)
