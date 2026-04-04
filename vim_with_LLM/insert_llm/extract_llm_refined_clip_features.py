import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from llm_refiner import QwenLoRARefiner, refine_feature_tensor


def parse_args():
    p = argparse.ArgumentParser(description="Refine CLIP/I3D npy features with Qwen+LoRA")
    p.add_argument("--input_dir", type=str, required=True, help="Directory with original .npy features")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to save refined .npy features")
    p.add_argument("--llm_name_or_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--in_feat_dim", type=int, default=768, help="768 for CLIP, 1024 for I3D")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--alpha_init", type=float, default=0.1)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dtype", type=str, default="float32", choices=["float32", "float16"])
    p.add_argument("--notes_json", type=str, default="", help="Optional JSON file: {video_id: note_text}")
    return p.parse_args()


def list_npy_files(root: str):
    return sorted([p for p in Path(root).glob("*.npy")])


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    notes_lookup = {}
    if args.notes_json:
        with open(args.notes_json, "r", encoding="utf-8") as f:
            notes_lookup = json.load(f)

    model = QwenLoRARefiner(
        in_feat_dim=args.in_feat_dim,
        llm_name_or_path=args.llm_name_or_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        alpha_init=args.alpha_init,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    npy_files = list_npy_files(args.input_dir)
    if len(npy_files) == 0:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")

    for src_path in tqdm(npy_files, desc="Refining features"):
        feats = np.load(src_path)
        if feats.ndim != 2:
            raise ValueError(f"Expected [T, C] feature shape, got {feats.shape} for {src_path}")
        if feats.shape[-1] != args.in_feat_dim:
            raise ValueError(
                f"Feature dim mismatch for {src_path}: got {feats.shape[-1]}, expected {args.in_feat_dim}"
            )

        feats_tensor = torch.from_numpy(feats).float()
        video_id = src_path.stem
        note = notes_lookup.get(video_id, "")
        refined = refine_feature_tensor(
            model=model,
            feats=feats_tensor,
            device=device,
            chunk_size=args.chunk_size,
            note=note,
        ).numpy()

        if args.save_dtype == "float16":
            refined = refined.astype(np.float16)
        else:
            refined = refined.astype(np.float32)

        out_path = Path(args.output_dir) / src_path.name
        np.save(out_path, refined)


if __name__ == "__main__":
    main()
