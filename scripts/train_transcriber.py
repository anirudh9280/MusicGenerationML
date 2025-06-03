#!/usr/bin/env python3
"""
train_transcriber.py

Train a small Bi-LSTM on top of CREPE features for piano transcription.
Usage:
    python train_transcriber.py \
      --crepe_dir /home/ubuntu/data/maestro_crepe \
      --label_dir /home/ubuntu/data/maestro_labels \
      --checkpoint_path /home/ubuntu/assignment2/transcriber_best.pt \
      --batch_size 8 \
      --epochs 10 \
      --val_split 0.1 \
      --max_frames None
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from maestro_dataset_fixed import MaestroFrameDataset

class FrameTranscriber(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, dropout=0.3, num_classes=89):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, T, 2]
        out, _ = self.lstm(x)       # out: [B, T, 2*hidden_dim]
        logits = self.fc(out)       # logits: [B, T, num_classes]
        return logits

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for feats, targets in loader:
        feats   = feats.to(device)       # [B, T, 2]
        targets = targets.to(device)     # [B, T]
        optimizer.zero_grad()
        logits = model(feats)            # [B, T, 89]
        loss   = criterion(logits.view(-1, 89), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    with torch.no_grad():
        for feats, targets in loader:
            feats   = feats.to(device)
            targets = targets.to(device)
            logits  = model(feats)                   # [B, T, 89]
            loss    = criterion(logits.view(-1, 89), targets.view(-1))
            total_loss += loss.item()
            preds = logits.argmax(dim=2)             # [B, T]
            correct += (preds == targets).sum().item()
            total   += targets.numel()
    return total_loss / len(loader), correct / total

def main(args):
    device = "cpu"
    print("Using device:", device)

    # 1. Load dataset and split
    ds = MaestroFrameDataset(
        crepe_dir=args.crepe_dir,
        label_dir=args.label_dir,
        max_frames=None if args.max_frames == "None" else int(args.max_frames)
    )
    n = len(ds)
    val_n = int(n * args.val_split)
    train_n = n - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # 2. Model, loss, optimizer
    model     = FrameTranscriber(hidden_dim=args.hidden_dim,
                                 num_layers=args.num_layers,
                                 dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}  TrainLoss: {tr_loss:.4f}  ValLoss: {vl_loss:.4f}  ValAcc: {vl_acc:.4f}")
        # Save best checkpoint
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"→ Saved best model: {args.checkpoint_path}  (ValAcc={vl_acc:.4f})")
    print("Training complete. Best ValAcc:", best_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crepe_dir",       required=True,
                        help="Directory of CREPE .npz files (features)")
    parser.add_argument("--label_dir",       required=True,
                        help="Directory of label .npz files (ground‐truth)")
    parser.add_argument("--checkpoint_path", default="transcriber_best.pt",
                        help="Where to save the best‐validation checkpoint")
    parser.add_argument("--batch_size",      type=int, default=8)
    parser.add_argument("--hidden_dim",      type=int, default=128)
    parser.add_argument("--num_layers",      type=int, default=2)
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--epochs",          type=int, default=10)
    parser.add_argument("--val_split",       type=float, default=0.1,
                        help="Fraction of data to reserve for validation")
    parser.add_argument("--max_frames",      default="None",
                        help="If set to an integer, pad/clip each example to this #frames")
    args = parser.parse_args()
    main(args) 