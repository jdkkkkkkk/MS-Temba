from __future__ import annotations

import argparse
from pathlib import Path

from PIL import ImageEnhance

from shared import build_rng, derive_tracklet_seed, list_frames, list_tracklet_dirs, load_images, save_images


def jitter_images(images, brightness, contrast, saturation):
    jittered = []
    for image in images:
        output = ImageEnhance.Brightness(image).enhance(brightness)
        output = ImageEnhance.Contrast(output).enhance(contrast)
        output = ImageEnhance.Color(output).enhance(saturation)
        jittered.append(output)
    return jittered



def main() -> None:
    parser = argparse.ArgumentParser(description='Apply the same color jitter to all frames in a tracklet.')
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--brightness', type=float, default=0.1, help='Range is sampled from [1-v, 1+v].')
    parser.add_argument('--contrast', type=float, default=0.1, help='Range is sampled from [1-v, 1+v].')
    parser.add_argument('--saturation', type=float, default=0.1, help='Range is sampled from [1-v, 1+v].')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    for tracklet_dir in list_tracklet_dirs(args.input_root):
        frame_paths = list_frames(tracklet_dir)
        if not frame_paths:
            continue
        rng = build_rng(derive_tracklet_seed(args.seed, tracklet_dir.name))
        brightness = rng.uniform(max(0.0, 1 - args.brightness), 1 + args.brightness)
        contrast = rng.uniform(max(0.0, 1 - args.contrast), 1 + args.contrast)
        saturation = rng.uniform(max(0.0, 1 - args.saturation), 1 + args.saturation)
        images = load_images(frame_paths)
        save_images(
            jitter_images(images, brightness=brightness, contrast=contrast, saturation=saturation),
            args.output_root / tracklet_dir.name,
        )


if __name__ == '__main__':
    main()
