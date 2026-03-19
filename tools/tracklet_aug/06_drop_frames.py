from __future__ import annotations

import argparse
from pathlib import Path

from shared import build_rng, derive_tracklet_seed, list_frames, list_tracklet_dirs, save_images


def main() -> None:
    parser = argparse.ArgumentParser(description='Randomly drop a small fraction of frames from each tracklet.')
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--drop-ratio', type=float, default=0.1)
    parser.add_argument('--min-keep', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not 0 <= args.drop_ratio < 1:
        raise ValueError('drop-ratio must be in [0, 1).')
    if args.min_keep <= 0:
        raise ValueError('min-keep must be positive.')

    for tracklet_dir in list_tracklet_dirs(args.input_root):
        frame_paths = list_frames(tracklet_dir)
        if len(frame_paths) <= args.min_keep:
            selected_paths = frame_paths
        else:
            rng = build_rng(derive_tracklet_seed(args.seed, tracklet_dir.name))
            drop_count = min(len(frame_paths) - args.min_keep, int(round(len(frame_paths) * args.drop_ratio)))
            drop_indices = set(rng.sample(range(len(frame_paths)), k=drop_count)) if drop_count > 0 else set()
            selected_paths = [path for idx, path in enumerate(frame_paths) if idx not in drop_indices]

        images = []
        for frame_path in selected_paths:
            from PIL import Image
            with Image.open(frame_path) as image:
                images.append(image.convert('RGB'))
        save_images(images, args.output_root / tracklet_dir.name)


if __name__ == '__main__':
    main()
