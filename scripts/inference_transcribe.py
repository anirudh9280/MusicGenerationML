#!/usr/bin/env python3
"""
inference_transcribe.py

Given a WAV file, extract CREPE features, run the trained FrameTranscriber,
and write out a symbolic MIDI file by merging consecutive same-pitch frames.

Usage:
    python inference_transcribe.py \
      /path/to/input.wav \
      /path/to/output.mid \
      --checkpoint_path /home/ubuntu/assignment2/transcriber_best.pt \
      --crepe_model full
"""

import os
import argparse
import torch
import torchcrepe
import librosa
import numpy as np
import pretty_midi
from train_transcriber import FrameTranscriber

def extract_crepe_on_audio(wav_path, model="full", device="cuda"):
    """
    Load WAV (resampled to 16 kHz), run TorchCREPE, return:
      - feats: np.array [T, 2] = [f0, confidence]
      - audio_length: int #samples in resampled audio
    """
    y, _ = librosa.load(wav_path, sr=16000, mono=True)
    audio = torch.tensor(y, dtype=torch.float32)[None].to(device)
    with torch.no_grad():
        f0, periodicity = torchcrepe.predict(
            audio, 16000,
            model=model,
            hop_length=160,
            fmin=65.41,
            fmax=1975.5,
            device=device,
            return_periodicity=True
        )
    f0_np   = f0[0].cpu().numpy()          # (T,)
    conf_np = periodicity[0].cpu().numpy() # (T,)
    feats = np.stack([f0_np, conf_np], axis=1).astype(np.float32)  # (T,2)
    return feats, len(y)

def frames_to_midi(preds, audio_length, hop=160, sr=16000, out_midi="out.mid"):
    """
    Convert frame‐wise predictions array [T] ∈ [0..88] to a MIDI file:
      - Merge consecutive frames of same pitch ≠ 88
      - Duration is from frame index start to frame index end
    """
    T = len(preds)
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    current_pitch = None
    start_frame   = None

    for t in range(T):
        p = int(preds[t])
        if p != current_pitch:
            # Close previous note
            if current_pitch is not None and current_pitch != 88:
                s = start_frame * hop / sr
                e = t * hop / sr
                midi_note = current_pitch + 21
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=midi_note,
                    start=s,
                    end=e
                )
                piano.notes.append(note)
            # Start a new note (if not no-note)
            if p != 88:
                start_frame = t
                current_pitch = p
            else:
                current_pitch = 88
                start_frame = None

    # Handle the last outstanding note
    if current_pitch is not None and current_pitch != 88:
        s = start_frame * hop / sr
        e = audio_length / sr
        midi_note = current_pitch + 21
        note = pretty_midi.Note(
            velocity=80,
            pitch=midi_note,
            start=s,
            end=e
        )
        piano.notes.append(note)

    pm.instruments.append(piano)
    pm.write(out_midi)
    print(f"Wrote MIDI: {out_midi}  ({len(piano.notes)} notes)")

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. Load trained FrameTranscriber
    model = FrameTranscriber(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Extract CREPE features for the input WAV
    feats, audio_len = extract_crepe_on_audio(
        args.wav_path,
        model=args.crepe_model,
        device=device
    )
    feats_t = torch.from_numpy(feats)[None].to(device)  # shape [1, T, 2]

    # 3. Forward‐pass through the model
    with torch.no_grad():
        logits = model(feats_t)                # [1, T, 89]
        preds  = logits.argmax(dim=2)[0].cpu().numpy()  # (T,)

    # 4. Convert frame predictions → MIDI
    frames_to_midi(
        preds,
        audio_len,
        hop=160,
        sr=16000,
        out_midi=args.output_midi
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path",         type=str,
                        help="Path to input WAV file (any sampling rate; auto-resampled)")
    parser.add_argument("output_midi",      type=str,
                        help="Where to write the generated MIDI")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained transcriber .pt file")
    parser.add_argument("--crepe_model",     default="full",
                        help="TorchCREPE model to use: tiny | lite | full")
    parser.add_argument("--hidden_dim",     type=int, default=128)
    parser.add_argument("--num_layers",     type=int, default=2)
    parser.add_argument("--dropout",        type=float, default=0.3)
    args = parser.parse_args()
    main(args) 