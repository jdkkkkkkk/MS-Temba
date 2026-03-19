from __future__ import annotations

import argparse
from pathlib import Path

from PIL import ImageDraw

from shared import build_rng, derive_tracklet_seed, list_frames, list_tracklet_dirs, load_images, save_images


def occlude_frame(image, rng, area_ratio):
    width, height = image.size
    rect_w = max(1, int(width * area_ratio))
    rect_h = max(1, int(height * area_ratio))
    max_left = max(0, width - rect_w)
    max_top = max(0, height - rect_h)
    left = rng.randint(0, max_left)
    top = rng.randint(0, max_top)
    output = image.copy()
    ImageDraw.Draw(output).rectangle((left, top, left + rect_w, top + rect_h), fill=(0, 0, 0))
    return output



def main() -> None:
    parser = argparse.ArgumentParser(description='Randomly occlude a subset of frames in each tracklet.')
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--frame-occlusion-prob', type=float, default=0.3)
    parser.add_argument('--occlusion-area-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not 0 <= args.frame_occlusion_prob <= 1:
        raise ValueError('frame-occlusion-prob must be in [0, 1].')
    if not 0 < args.occlusion_area_ratio < 1:
        raise ValueError('occlusion-area-ratio must be in (0, 1).')

    for tracklet_dir in list_tracklet_dirs(args.input_root):
        frame_paths = list_frames(tracklet_dir)
        if not frame_paths:
            continue
        rng = build_rng(derive_tracklet_seed(args.seed, tracklet_dir.name))
        images = load_images(frame_paths)
        output_images = []
        for image in images:
            if rng.random() < args.frame_occlusion_prob:
                output_images.append(occlude_frame(image, rng, args.occlusion_area_ratio))
            else:
                output_images.append(image)
        save_images(output_images, args.output_root / tracklet_dir.name)


if __name__ == '__main__':
    main()
