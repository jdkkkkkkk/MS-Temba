from __future__ import annotations

import argparse
from pathlib import Path

from shared import build_rng, derive_tracklet_seed, list_frames, list_tracklet_dirs, save_images


def sample_tracklet(frame_paths, min_length, max_length, rng):
    if not frame_paths:
        return []
    max_allowed = min(max_length, len(frame_paths))
    min_allowed = min(min_length, max_allowed)
    clip_length = rng.randint(min_allowed, max_allowed)
    start = rng.randint(0, len(frame_paths) - clip_length)
    return frame_paths[start : start + clip_length]



def main() -> None:
    parser = argparse.ArgumentParser(description='Randomly sample a subsequence from each tracklet folder.')
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--min-length', type=int, required=True)
    parser.add_argument('--max-length', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.min_length <= 0 or args.max_length <= 0:
        raise ValueError('min-length and max-length must be positive.')
    if args.min_length > args.max_length:
        raise ValueError('min-length cannot exceed max-length.')

    root_rng = build_rng(args.seed)
    for tracklet_dir in list_tracklet_dirs(args.input_root):
        frame_paths = list_frames(tracklet_dir)
        tracklet_rng = build_rng(derive_tracklet_seed(args.seed, tracklet_dir.name) or root_rng.randint(0, 10**9))
        sampled_frames = sample_tracklet(frame_paths, args.min_length, args.max_length, tracklet_rng)
        if not sampled_frames:
            continue
        images = []
        for frame_path in sampled_frames:
            from PIL import Image
            with Image.open(frame_path) as image:
                images.append(image.convert('RGB'))
        save_images(images, args.output_root / tracklet_dir.name)


if __name__ == '__main__':
    main()
