#!/usr/bin/env python3
"""
extract_crepe_features.py

For each WAV in MAESTRO, resample to 16 kHz, run TorchCREPE, and save
f0 (Hz) and confidence vectors to a .npz file.  
Usage:
    python extract_crepe_features.py \
      --input_dir /home/ubuntu/data/maestro_2004 \
      --output_dir /home/ubuntu/data/maestro_crepe \
      --crepe_model full \
      --device cuda
"""

import os
import glob
import argparse
import torch
import torchcrepe
import librosa
import numpy as np

def extract_for_wav(wav_path, npz_path, model="full", device="cuda"):
    # 1. Load WAV at 16 kHz mono
    y, _ = librosa.load(wav_path, sr=16000, mono=True)
    audio = torch.tensor(y, dtype=torch.float32)[None].to(device)

    # 2. Run TorchCREPE
    with torch.no_grad():
        f0, periodicity = torchcrepe.predict(
            audio,
            16000,
            model=model,
            hop_length=160,       # 10 ms frames
            fmin=65.41,           # C2
            fmax=1975.5,          # G6
            device=device,
            return_periodicity=True
        )
    # Convert to numpy on CPU
    f0_np   = f0[0].cpu().numpy().astype(np.float32)
    conf_np = periodicity[0].cpu().numpy().astype(np.float32)

    # 3. Save to .npz
    np.savez(npz_path, f0=f0_np, conf=conf_np)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    wav_paths = glob.glob(os.path.join(args.input_dir, "**", "*.wav"), recursive=True)
    for wav_path in wav_paths:
        rel = os.path.relpath(wav_path, args.input_dir)
        npz_out = os.path.join(args.output_dir, os.path.splitext(rel)[0] + ".npz")
        os.makedirs(os.path.dirname(npz_out), exist_ok=True)
        if os.path.exists(npz_out):
            continue
        print("Extracting:", rel)
        extract_for_wav(wav_path, npz_out, model=args.crepe_model, device=args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  required=True,
                        help="Root of MAESTRO WAV files")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write .npz CREPE features")
    parser.add_argument("--crepe_model", default="full",
                        help="TorchCREPE model to use: tiny | lite | full")
    parser.add_argument("--device", default="cuda",
                        help="Device for TorchCREPE: cuda or cpu")
    args = parser.parse_args()
    main(args) 