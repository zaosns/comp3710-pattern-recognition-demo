"""Shared data loading helpers for the preprocessed OASIS PNG dataset."""

from __future__ import annotations

import io
import random
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_SOURCE = ROOT / "keras_png_slices_data (1).zip"

IMAGE_FOLDERS = {
    "train": "keras_png_slices_train",
    "validate": "keras_png_slices_validate",
    "test": "keras_png_slices_test",
}
MASK_FOLDERS = {
    "train": "keras_png_slices_seg_train",
    "validate": "keras_png_slices_seg_validate",
    "test": "keras_png_slices_seg_test",
}
FILENAME_PATTERN = re.compile(r"(?:case|seg)_(\d+)_slice_(\d+)\.nii\.png$")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_key(name: str) -> tuple[int, int]:
    match = FILENAME_PATTERN.search(name)
    if match is None:
        raise ValueError(f"Unexpected OASIS filename: {name}")
    return int(match.group(1)), int(match.group(2))


def find_pngs(source: Path, folder: str) -> tuple[list[str], bool]:
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = [
                name
                for name in archive.namelist()
                if f"/{folder}/" in name and name.endswith(".png")
            ]
        return sorted(names), True
    if source.is_dir():
        names = [
            str(path)
            for path in source.rglob("*.png")
            if path.parent.name == folder
        ]
        return sorted(names), False
    raise FileNotFoundError(f"OASIS data not found: {source}")


class MRISliceDataset(Dataset):
    """Return one-channel MRI slices and their slice indices."""

    def __init__(self, source: Path, split: str, image_size: int = 128):
        if split not in IMAGE_FOLDERS:
            raise ValueError(f"Unknown split: {split}")
        self.source = Path(source)
        self.image_size = image_size
        self.samples, self.from_zip = find_pngs(
            self.source, IMAGE_FOLDERS[split]
        )
        self.archive: zipfile.ZipFile | None = None
        if not self.samples:
            raise FileNotFoundError(f"No OASIS images found for split '{split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["archive"] = None
        return state

    def open_image(self, name: str):
        if self.from_zip:
            if self.archive is None:
                self.archive = zipfile.ZipFile(self.source)
            return io.BytesIO(self.archive.read(name))
        return open(name, "rb")

    def __getitem__(self, index: int):
        name = self.samples[index]
        _, slice_index = sample_key(name)
        with self.open_image(name) as stream, Image.open(stream) as image:
            image = TF.resize(
                image.convert("L"),
                [self.image_size, self.image_size],
                antialias=True,
            )
            image = TF.to_tensor(image)
        return image, slice_index, Path(name).name


class SegmentationDataset(Dataset):
    """Return an MRI slice and its four-channel one-hot segmentation mask."""

    def __init__(self, source: Path, split: str, image_size: int = 128):
        if split not in IMAGE_FOLDERS:
            raise ValueError(f"Unknown split: {split}")
        self.source = Path(source)
        self.image_size = image_size
        image_names, self.from_zip = find_pngs(
            self.source, IMAGE_FOLDERS[split]
        )
        mask_names, _ = find_pngs(self.source, MASK_FOLDERS[split])
        images = {sample_key(name): name for name in image_names}
        masks = {sample_key(name): name for name in mask_names}
        if images.keys() != masks.keys():
            raise ValueError("OASIS images and segmentation masks do not match")
        self.samples = [(images[key], masks[key], key) for key in sorted(images)]
        self.archive: zipfile.ZipFile | None = None
        if not self.samples:
            raise FileNotFoundError(f"No OASIS masks found for split '{split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["archive"] = None
        return state

    def open_image(self, name: str):
        if self.from_zip:
            if self.archive is None:
                self.archive = zipfile.ZipFile(self.source)
            return io.BytesIO(self.archive.read(name))
        return open(name, "rb")

    def __getitem__(self, index: int):
        image_name, mask_name, key = self.samples[index]
        with self.open_image(image_name) as stream, Image.open(stream) as image:
            image = TF.resize(
                image.convert("L"),
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            image = TF.to_tensor(image)

        with self.open_image(mask_name) as stream, Image.open(stream) as mask:
            mask = TF.resize(
                mask.convert("L"),
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )
            mask = torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy())

        # Mask values 0, 85, 170 and 255 represent the four categories.
        labels = torch.div(mask.long(), 85, rounding_mode="floor").clamp(0, 3)
        one_hot = F.one_hot(labels, num_classes=4).permute(2, 0, 1).float()
        name = f"case_{key[0]:03d}_slice_{key[1]}"
        return image, one_hot, name
