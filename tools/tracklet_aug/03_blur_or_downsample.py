from __future__ import annotations

import argparse
from pathlib import Path

from PIL import ImageFilter

from shared import build_rng, derive_tracklet_seed, list_frames, list_tracklet_dirs, load_images, save_images


def degrade_images(images, mode, blur_radius, downsample_ratio):
    degraded = []
    for image in images:
        if mode in {'blur', 'mixed'}:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        if mode in {'downsample', 'mixed'}:
            width, height = image.size
            resized_w = max(1, int(width * downsample_ratio))
            resized_h = max(1, int(height * downsample_ratio))
            image = image.resize((resized_w, resized_h), resample=image.Resampling.BILINEAR)
            image = image.resize((width, height), resample=image.Resampling.BICUBIC)
        degraded.append(image)
    return degraded



def main() -> None:
    parser = argparse.ArgumentParser(description='Apply blur and/or resolution degradation to each tracklet.')
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--mode', choices=['blur', 'downsample', 'mixed'], default='mixed')
    parser.add_argument('--max-blur-radius', type=float, default=1.2)
    parser.add_argument('--min-downsample-ratio', type=float, default=0.65)
    parser.add_argument('--max-downsample-ratio', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.min_downsample_ratio <= args.max_downsample_ratio <= 1.0:
        raise ValueError('Downsample ratios must satisfy 0 < min <= max <= 1.')

    for tracklet_dir in list_tracklet_dirs(args.input_root):
        frame_paths = list_frames(tracklet_dir)
        if not frame_paths:
            continue
        rng = build_rng(derive_tracklet_seed(args.seed, tracklet_dir.name))
        blur_radius = rng.uniform(0.3, args.max_blur_radius)
        downsample_ratio = rng.uniform(args.min_downsample_ratio, args.max_downsample_ratio)
        images = load_images(frame_paths)
        save_images(
            degrade_images(images, args.mode, blur_radius=blur_radius, downsample_ratio=downsample_ratio),
            args.output_root / tracklet_dir.name,
        )


if __name__ == '__main__':
    main()
