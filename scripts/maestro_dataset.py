#!/usr/bin/env python3
"""
maestro_dataset.py

Defines a PyTorch Dataset that returns:
  - features: FloatTensor [T, 2] = [f0, conf]
  - targets : LongTensor [T]  = values in [0..88]

Place this file under ~/assignment2/scripts/.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class MaestroFrameDataset(Dataset):
    """
    PyTorch Dataset that loads (f0, confidence) features from CREPE .npz
    and frame‐level ground-truth labels from label .npz.  
    Each item:
      feats:  FloatTensor [T, 2]
      targets: LongTensor  [T] (0..88)
    If max_frames is set, each example is padded/truncated to exactly max_frames.
    """
    def __init__(self, crepe_dir, label_dir, max_frames=None):
        self.pairs = []
        for crepe_npz in glob.glob(os.path.join(crepe_dir, "**", "*.npz"), recursive=True):
            rel = os.path.relpath(crepe_npz, crepe_dir)
            label_npz = os.path.join(label_dir, os.path.splitext(rel)[0] + ".npz")
            if os.path.exists(label_npz):
                self.pairs.append((crepe_npz, label_npz))
        self.max_frames = max_frames

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        crepe_npz, label_npz = self.pairs[idx]
        data = np.load(crepe_npz)
        f0   = data["f0"]       # shape (T,)
        conf = data["conf"]     # shape (T,)
        feats = np.stack([f0, conf], axis=1).astype(np.float32)  # (T,2)

        lbl_data = np.load(label_npz)["labels"]  # (T,89)
        targets  = np.argmax(lbl_data, axis=1).astype(np.int64)  # (T,)

        if self.max_frames is not None:
            T = feats.shape[0]
            if T < self.max_frames:
                pad = self.max_frames - T
                feats   = np.pad(feats,   ((0,pad),(0,0)), mode="constant")
                targets = np.pad(targets, (0,pad),    mode="constant", constant_values=88)
            else:
                feats   = feats[:self.max_frames]
                targets = targets[:self.max_frames]

        return torch.from_numpy(feats), torch.from_numpy(targets) 