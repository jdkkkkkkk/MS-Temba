from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List

from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_tracklet_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f'Tracklet root not found: {root}')
    tracklets = [path for path in root.iterdir() if path.is_dir()]
    return sorted(tracklets)



def list_frames(tracklet_dir: Path) -> List[Path]:
    frames = [
        path for path in tracklet_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(frames)



def load_images(frame_paths: Iterable[Path]) -> List[Image.Image]:
    images = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            images.append(image.convert('RGB'))
    return images



def save_images(images: List[Image.Image], output_tracklet_dir: Path, start_index: int = 1) -> None:
    ensure_dir(output_tracklet_dir)
    for index, image in enumerate(images, start=start_index):
        output_path = output_tracklet_dir / f'{index:05d}.jpg'
        image.save(output_path, quality=95)



def build_rng(seed: int | None) -> random.Random:
    return random.Random(seed)



def derive_tracklet_seed(global_seed: int | None, tracklet_name: str) -> int | None:
    if global_seed is None:
        return None
    return global_seed + sum(ord(char) for char in tracklet_name)
