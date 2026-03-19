# Tracklet augmentation scripts

These scripts assume the input dataset is organized as one folder per tracklet, with frame names sorted like `00001.jpg`, `00002.jpg`, and so on.

## Recommended execution order

Run the scripts in this sequence so each stage writes into a new directory and feeds the next stage:

1. `01_sample_subsequence.py`
2. `02_uniform_color_jitter.py`
3. `03_blur_or_downsample.py`
4. `04_bbox_jitter.py`
5. `05_partial_occlusion.py`
6. `06_drop_frames.py`

Example:

```bash
python tools/tracklet_aug/01_sample_subsequence.py \
  --input-root data/raw_tracklets \
  --output-root data/aug_step01 \
  --min-length 8 \
  --max-length 16

python tools/tracklet_aug/02_uniform_color_jitter.py \
  --input-root data/aug_step01 \
  --output-root data/aug_step02
```

## Notes

- `04_bbox_jitter.py` simulates detection-box jitter **when the frame itself is already the cropped target patch**. It shifts and rescales the crop slightly, then resizes back to the original frame size.
- Each script keeps frame order and rewrites output frames starting from `00001.jpg`.
- Each tracklet uses one deterministic random seed derived from the global seed and tracklet name, so the same tracklet receives consistent augmentation when re-run.
- The scripts depend on Pillow: `pip install pillow`.
