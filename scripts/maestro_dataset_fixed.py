import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class MaestroFrameDataset(Dataset):
    def __init__(self, crepe_dir, label_dir, max_frames=None):
        self.pairs = []
        for crepe_npz in glob.glob(os.path.join(crepe_dir, "*.npz")):
            base_name = os.path.splitext(os.path.basename(crepe_npz))[0]
            label_npz = os.path.join(label_dir, base_name + ".npz")
            if os.path.exists(label_npz):
                self.pairs.append((crepe_npz, label_npz))
        self.max_frames = max_frames

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        crepe_npz, label_npz = self.pairs[idx]
        
        # Load CREPE features
        data = np.load(crepe_npz)
        f0 = data["f0"]
        conf = data["conf"]
        feats = np.stack([f0, conf], axis=1).astype(np.float32)
        
        # Load labels
        lbl = np.load(label_npz)["labels"]
        targets = np.argmax(lbl, axis=1).astype(np.int64)
        
        # Ensure exact alignment by using minimum length
        min_len = min(len(feats), len(targets))
        feats = feats[:min_len]
        targets = targets[:min_len]
        
        if self.max_frames is not None:
            if len(feats) > self.max_frames:
                feats = feats[:self.max_frames]
                targets = targets[:self.max_frames]
            elif len(feats) < self.max_frames:
                pad_len = self.max_frames - len(feats)
                feats = np.pad(feats, ((0, pad_len), (0, 0)), mode="constant")
                targets = np.pad(targets, (0, pad_len), mode="constant", constant_values=88)
        
        return torch.from_numpy(feats), torch.from_numpy(targets)
