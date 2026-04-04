import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Build notes_json from annotation + class map")
    p.add_argument("--ann_json", required=True, help="Annotation JSON file")
    p.add_argument("--class_map_json", required=True, help="Class map JSON: {id: text} or {classes:{id:text}}")
    p.add_argument("--output_json", required=True, help="Output notes JSON: {video_id: note_text}")
    p.add_argument("--subset", default="", help="Optional subset filter, e.g. training/testing")
    return p.parse_args()


def load_class_map(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "classes" in obj and isinstance(obj["classes"], dict):
        obj = obj["classes"]
    out = {}
    for k, v in obj.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            continue
    return out


def normalize_action(ann):
    if isinstance(ann, dict):
        cls_id = int(ann.get("label", ann.get("class_id", -1)))
        t_start = float(ann.get("start", ann.get("t_start", 0.0)))
        t_end = float(ann.get("end", ann.get("t_end", 0.0)))
        note = str(ann.get("note", ann.get("label_text", "")))
    else:
        cls_id = int(ann[0])
        t_start = float(ann[1])
        t_end = float(ann[2])
        note = str(ann[3]) if len(ann) > 3 else ""
    return cls_id, t_start, t_end, note


def parse_video_note(note_text: str):
    """Parse note schema: view ; action_description ; occlusion."""
    if note_text is None:
        return "", "", ""
    parts = [p.strip() for p in str(note_text).split(";")]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    if len(parts) == 1:
        return "", parts[0], ""
    return "", "", ""


def main():
    args = parse_args()
    class_map = load_class_map(args.class_map_json)

    with open(args.ann_json, "r", encoding="utf-8") as f:
        anns = json.load(f)

    notes = {}
    for vid, meta in anns.items():
        if args.subset and meta.get("subset", "") != args.subset:
            continue

        base_note = str(meta.get("note", "")).strip()
        view_text, action_desc_text, occlusion_text = parse_video_note(base_note)
        note_parts = [p.strip() for p in action_desc_text.split("|") if p.strip()]
        action_texts = []
        actions = meta.get("actions", [])
        for idx, ann in enumerate(actions):
            cls_id, t_start, t_end, ann_note = normalize_action(ann)
            if not ann_note and len(note_parts) > 1 and idx < len(note_parts):
                ann_note = note_parts[idx]
            elif not ann_note and len(note_parts) == 1 and len(actions) > 1:
                ann_note = note_parts[0]
            elif not ann_note and action_desc_text:
                ann_note = action_desc_text
            cls_text = class_map.get(cls_id, f"class_{cls_id}")
            seg = f"action={cls_text}; start={t_start:.2f}; end={t_end:.2f}"
            if view_text:
                seg = f"view={view_text}; {seg}"
            if ann_note:
                seg += f"; detail={ann_note}"
            if occlusion_text:
                seg += f"; occlusion={occlusion_text}"
            action_texts.append(seg)

        full = " | ".join(action_texts).strip() if action_texts else action_desc_text
        notes[vid] = full

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()