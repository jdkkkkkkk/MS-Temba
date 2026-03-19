from __future__ import annotations

import argparse
from pathlib import Path

from PIL import ImageOps

from shared import build_rng, derive_tracklet_seed, list_frames, list_tracklet_dirs, load_images, save_images


def jitter_crop(image, shift_ratio_x, shift_ratio_y, scale_ratio):
    width, height = image.size
    pad_x = max(1, int(width * 0.1))
    pad_y = max(1, int(height * 0.1))
    padded = ImageOps.expand(image, border=(pad_x, pad_y, pad_x, pad_y), fill=0)

    crop_w = max(1, int(width * scale_ratio))
    crop_h = max(1, int(height * scale_ratio))
    center_x = pad_x + width / 2 + shift_ratio_x * width
    center_y = pad_y + height / 2 + shift_ratio_y * height

    left = int(round(center_x - crop_w / 2))
    top = int(round(center_y - crop_h / 2))
    right = left + crop_w
    bottom = top + crop_h

    left = max(0, min(left, padded.width - crop_w))
    top = max(0, min(top, padded.height - crop_h))
    right = left + crop_w
    bottom = top + crop_h

    cropped = padded.crop((left, top, right, bottom))
    return cropped.resize((width, height), resample=image.Resampling.BICUBIC)



def main() -> None:
    parser = argparse.ArgumentParser(
        description='Simulate bbox jitter on cropped tracklet frames by shifting/scaling the crop slightly.'
    )
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--max-shift-ratio', type=float, default=0.04)
    parser.add_argument('--max-scale-jitter', type=float, default=0.06, help='Crop size is sampled from [1-v, 1+v].')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    for tracklet_dir in list_tracklet_dirs(args.input_root):
        frame_paths = list_frames(tracklet_dir)
        if not frame_paths:
            continue
        rng = build_rng(derive_tracklet_seed(args.seed, tracklet_dir.name))
        shift_ratio_x = rng.uniform(-args.max_shift_ratio, args.max_shift_ratio)
        shift_ratio_y = rng.uniform(-args.max_shift_ratio, args.max_shift_ratio)
        scale_ratio = rng.uniform(1 - args.max_scale_jitter, 1 + args.max_scale_jitter)
        images = load_images(frame_paths)
        output_images = [jitter_crop(image, shift_ratio_x, shift_ratio_y, scale_ratio) for image in images]
        save_images(output_images, args.output_root / tracklet_dir.name)


if __name__ == '__main__':
    main()
