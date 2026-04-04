import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import numpy as np
import json
import os
import os.path
from tqdm import tqdm
import random
from utils import *

def _load_class_map(class_map_path):
    if not class_map_path:
        return {}
    with open(class_map_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # Support both {"1":"open door"} and {"classes":{"1":"open door"}}
    if isinstance(obj, dict) and "classes" in obj and isinstance(obj["classes"], dict):
        obj = obj["classes"]
    norm = {}
    for k, v in obj.items():
        try:
            norm[int(k)] = str(v)
        except Exception:
            continue
    return norm


def _action_to_struct(ann):
    """Normalize action annotation to (cls_id, t_start, t_end, action_note)."""
    action_note = ""
    if isinstance(ann, dict):
        cls_id = int(ann.get('label', ann.get('class_id', -1)))
        t_start = float(ann.get('start', ann.get('t_start', 0.0)))
        t_end = float(ann.get('end', ann.get('t_end', 0.0)))
        action_note = ann.get('note', ann.get('label_text', ''))
    else:
        cls_id = int(ann[0])
        t_start = float(ann[1])
        t_end = float(ann[2])
        if len(ann) > 3:
            action_note = ann[3]
    return cls_id, t_start, t_end, str(action_note) if action_note else ""


def _parse_video_note(note_text: str):
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


def make_dataset(split_file, split, root, num_classes=157, class_map_path=""):
    gamma = 0.5
    tau = 4
    ku = 1
    class_map = _load_class_map(class_map_path)
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)
    print('split!!!!', split)
    i = 0
    for vid in tqdm(data.keys()):
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid + '.npy')):
            continue

        if len(data[vid]['actions']) < 1:
            continue

        fts = np.load(os.path.join(root, vid + '.npy'))
        num_feat = fts.shape[0]
        label = np.zeros((num_feat, num_classes), np.float32)
        #
        hmap = np.zeros((num_feat, num_classes), np.float32)
        action_lengths = []
        center_loc = []
        num_action = 0

        fps = num_feat / data[vid]['duration']
        video_note = data[vid].get('note', '')
        if not isinstance(video_note, str):
            video_note = str(video_note)
        view_text, action_desc_text, occlusion_text = _parse_video_note(video_note)
        note_parts = [p.strip() for p in action_desc_text.split('|') if p.strip()]
        per_action_notes = []
        actions = data[vid]['actions']
        for idx, ann in enumerate(actions):
            cls_id, t_start, t_end, action_note = _action_to_struct(ann)
            # If note has pipeline parts and action has no explicit note, align by index.
            if not action_note and len(note_parts) > 1 and idx < len(note_parts):
                action_note = note_parts[idx]
            elif not action_note and len(note_parts) == 1 and len(actions) > 1:
                action_note = note_parts[0]
            elif not action_note and action_desc_text:
                action_note = action_desc_text
            cls_text = class_map.get(cls_id, f"class_{cls_id}")
            action_text = f"action={cls_text}; start={t_start:.2f}; end={t_end:.2f}"
            if view_text:
                action_text = f"view={view_text}; {action_text}"
            if action_note:
                action_text = f"{action_text}; detail={action_note}"
            if occlusion_text:
                action_text = f"{action_text}; occlusion={occlusion_text}"
            per_action_notes.append(action_text)
            # 
            if t_end < t_start:
                continue
            mid_point = (t_end + t_start) / 2
            for fr in range(0, num_feat, 1):
                if fr / fps > t_start and fr / fps < t_end:
                    label[fr, cls_id] = 1  # binary classification

                # G* Ground truth Heat-map
                # if fr / fps + 1 > mid_point and fr / fps < mid_point:
                if (fr+1) / fps > mid_point and fr / fps < mid_point:
                    center = fr + 1
                    class_ = cls_id
                    action_duration = int((t_end - t_start) * fps)
                    radius = int(action_duration / gamma)
                    generate_gaussian(hmap[:, class_], center, radius, tau, ku)
                    num_action = num_action + 1
                    center_loc.append([center, class_])
                    action_lengths.append([action_duration])

        merged_note = ""
        if per_action_notes:
            merged_note = " | ".join(per_action_notes)
        elif action_desc_text:
            merged_note = action_desc_text
        elif video_note:
            merged_note = video_note
        dataset.append((vid, label, data[vid]['duration'], [hmap, num_action, np.asarray(center_loc), np.asarray(action_lengths)], merged_note))
        i += 1

    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, batch_size, classes, num_clips, skip, class_map_path=""):
        
        self.data = make_dataset(split_file, split, root, classes, class_map_path=class_map_path)
        self.split=split
        self.split_file = split_file
        self.batch_size = batch_size
        self.root = root
        self.in_mem = {}
        self.num_clips = num_clips
        self.skip = skip

    def __getitem__(self, index):
        entry = self.data[index]
        feat = np.load(os.path.join(self.root, entry[0] + '.npy'))
        feat = feat.reshape((feat.shape[0], 1, 1, feat.shape[-1]))
        features = feat.astype(np.float32)

        labels = entry[1]

        hmap, num_action, center_loc, action_lengths = entry[3]
        note = entry[4] if len(entry) > 4 else ""
        # print('center_loc',center_loc.shape)
        # center_loc = np.transpose(center_loc, axes=[1, 0])
        num_clips = self.num_clips

        if self.split in ["training", "testing"]:
            if len(features) > num_clips and num_clips > 0:
                if self.split == "testing":
                    random_index = 0
                else:
                    random_index = random.choice(range(0, len(features) - num_clips))
                features = features[random_index: random_index + num_clips: 1]
                labels = labels[random_index: random_index + num_clips: 1]
                hmap = hmap[random_index: random_index + num_clips: 1]
        # center_loc = np.transpose(center_loc, axes=[1, 0])

        return features, labels, hmap, action_lengths, [entry[0], entry[2], num_action, note]

    def __len__(self):
        return len(self.data)


class collate_fn_unisize():

    def __init__(self,num_clips):
        self.num_clips = num_clips

    def charades_collate_fn_unisize(self, batch):
        max_len = int(self.num_clips)
        max_len1= int(self.num_clips)
        new_batch = []
        for b in batch:
            f = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
            m = np.zeros((max_len), np.float32)
            l = np.zeros((max_len, b[1].shape[1]), np.float32)
            h = np.zeros((max_len, b[2].shape[1]), np.float32)
            f[:b[0].shape[0]] = b[0]
            m[:b[0].shape[0]] = 1
            l[:b[0].shape[0], :] = b[1]
            h[:b[0].shape[0], :] = b[2]

            new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[4], torch.from_numpy(h)])

        return default_collate(new_batch)