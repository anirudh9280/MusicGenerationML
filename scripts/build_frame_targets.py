#!/usr/bin/env python3
"""
Frame Target Generation Script
Converts MIDI files to frame-level piano roll labels for transcription training.

For each MIDI in MAESTRO, resample (if needed) and align to 10 ms frames.
Produce a .npz per example containing `labels` of shape (T, 89), where:
  - columns 0–87 correspond to pitches 21..108 (A0..C8)
  - column 88 corresponds to "no-note"
Usage:
    python build_frame_targets.py \
      --wav_dir /home/ubuntu/data/maestro_2004 \
      --midi_dir /home/ubuntu/data/maestro_2004 \
      --output_dir /home/ubuntu/data/maestro_labels \
      --sr 16000 \
      --hop_length 160
"""

import os
import glob
import argparse
import numpy as np
import librosa
import pretty_midi

def midi_to_frame_labels(midi_path, audio_length_samples, sr=16000, hop_length=160):
    """
    Build a (T, 89) array where T = ceil(audio_length_samples / hop_length).
    pitches 21..108 map to columns 0..87, and column 88 = no-note.
    """
    # Number of frames
    T = int(np.ceil(audio_length_samples / hop_length))
    labels = np.zeros((T, 89), dtype=np.uint8)

    pm = pretty_midi.PrettyMIDI(midi_path)
    for inst in pm.instruments:
        for note in inst.notes:
            start_idx = int(np.round(note.start * sr / hop_length))
            end_idx   = int(np.round(note.end   * sr / hop_length))
            # Clamp indices
            start_idx = max(0, min(T-1, start_idx))
            end_idx   = max(0, min(T,   end_idx))
            pitch = note.pitch
            if 21 <= pitch <= 108:
                labels[start_idx:end_idx, pitch - 21] = 1

    # Mark "no-note" frames
    active_any = labels[:, :88].any(axis=1)
    labels[~active_any, 88] = 1
    return labels

def main(args):
    """Main processing function"""
    print(f"🎼 Frame Target Generation")
    print(f"WAV directory: {args.wav_dir}")
    print(f"MIDI directory: {args.midi_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample rate: {args.sr}")
    print(f"Hop length: {args.hop_length} samples ({args.hop_length/args.sr*1000:.1f}ms)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all WAV files
    wav_paths = glob.glob(os.path.join(args.wav_dir, "**", "*.wav"), recursive=True)
    print(f"Found {len(wav_paths)} WAV files")
    
    if len(wav_paths) == 0:
        print("❌ No WAV files found!")
        return
    
    # Process each file
    successful = 0
    failed = 0
    total_frames = 0
    
    for wav_path in wav_paths:
        rel_path = os.path.relpath(wav_path, args.wav_dir)
        midi_path = os.path.join(args.midi_dir, os.path.splitext(rel_path)[0] + ".midi")
        
        # Try .mid extension if .midi doesn't exist
        if not os.path.exists(midi_path):
            midi_path = os.path.join(args.midi_dir, os.path.splitext(rel_path)[0] + ".mid")
        
        if not os.path.exists(midi_path):
            print(f"Warning: No MIDI file found for {rel_path}")
            failed += 1
            continue
        
        # Load audio to get length
        y, _ = librosa.load(wav_path, sr=args.sr, mono=True)
        audio_length_samples = len(y)
        
        # Generate frame labels
        labels = midi_to_frame_labels(midi_path, audio_length_samples,
                                      sr=args.sr, hop_length=args.hop_length)
        
        # Create output path
        output_path = os.path.join(args.output_dir, os.path.splitext(rel_path)[0] + ".npz")
        
        # Skip if already exists
        if os.path.exists(output_path) and not args.overwrite:
            continue
        
        # Save labels
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path, labels=labels)
        
        successful += 1
        total_frames += labels.shape[0]
    
    print(f"\n✅ Label generation complete!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total frames: {total_frames:,}")
    print(f"   Average frames per file: {total_frames/max(1,successful):.0f}")
    print(f"   Output directory: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate frame-level labels from MIDI")
    parser.add_argument("--wav_dir", required=True, help="Directory with WAV files")
    parser.add_argument("--midi_dir", required=True, help="Directory with MIDI files")
    parser.add_argument("--output_dir", required=True, help="Output directory for label NPZ files")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--hop_length", type=int, default=160, help="Hop length in samples")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    main(args) 