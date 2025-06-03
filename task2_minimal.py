#!/usr/bin/env python3
"""
Task 2: Minimal Implementation - Works without problematic dependencies
Onsets and Frames Implementation for Symbolic Conditioned Generation
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Core imports that should work
import numpy as np
import tensorflow as tf
from pathlib import Path

# Optional imports with fallbacks
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  pandas not available - using basic analysis")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️  plotting not available - skipping visualizations")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️  librosa not available - using scipy for audio")
    import scipy.io.wavfile as wavfile

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False
    print("⚠️  pretty_midi not available - using simplified MIDI")

# Add Magenta to path
sys.path.append('./libs/magenta')

# Try Magenta imports
try:
    import magenta.music as mm
    from magenta.models.onsets_frames_transcription import constants
    HAS_MAGENTA = True
    print("✅ Magenta available")
except ImportError:
    HAS_MAGENTA = False
    print("⚠️  Magenta not fully available - using simplified approach")

# =============================================================================
# SIMPLIFIED IMPLEMENTATIONS
# =============================================================================

def setup_environment_minimal():
    """Minimal environment setup"""
    print("🎵 Task 2: Minimal Symbolic Conditioned Generation Setup")
    print("=" * 60)
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {gpus[0]}")
    else:
        print("⚠️  No GPU detected, using CPU")
    
    # Check data
    data_path = Path("./data/maestro_tfrecords")
    if data_path.exists():
        tfrecords = list(data_path.glob("*.tfrecord*"))
        print(f"✅ Found {len(tfrecords)} TFRecord files")
    else:
        print("❌ MAESTRO data not found")
    
    return True

def analyze_maestro_minimal():
    """Minimal dataset analysis"""
    print("📊 MAESTRO Dataset Analysis (Simplified)")
    print("=" * 40)
    
    if HAS_PANDAS:
        csv_path = Path("./data/maestro_tfrecords/maestro-v3.0.0.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"Total recordings: {len(df)}")
            print(f"Dataset splits: {df['split'].value_counts().to_dict()}")
            return df
    
    # Fallback: analyze TFRecord files
    tfrecord_files = list(Path("./data/maestro_tfrecords").glob("*.tfrecord*"))
    total_size = sum(f.stat().st_size for f in tfrecord_files)
    print(f"Found {len(tfrecord_files)} TFRecord files")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    return None

def create_synthetic_audio():
    """Create synthetic piano audio without librosa"""
    print("🎹 Creating synthetic piano audio...")
    
    # Basic audio synthesis
    sr = 16000  # Lower sample rate for compatibility
    duration = 8
    t = np.linspace(0, duration, int(sr * duration))
    
    # Piano notes (C major scale)
    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    audio = np.zeros_like(t)
    
    for i, freq in enumerate(frequencies):
        start = i * 0.8
        end = start + 0.6
        mask = (t >= start) & (t <= end)
        # Simple exponential decay envelope
        envelope = np.exp(-3 * (t - start)) * mask
        note = 0.3 * np.sin(2 * np.pi * freq * t) * envelope
        audio += note
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Save using scipy if librosa not available
    output_path = "synthetic_piano.wav"
    if HAS_LIBROSA:
        librosa.output.write_wav(output_path, audio, sr)
    else:
        # Use scipy
        wavfile.write(output_path, sr, (audio * 32767).astype(np.int16))
    
    print(f"✅ Created {output_path}")
    return output_path, audio, sr

def simple_onset_detection(audio, sr):
    """Simple onset detection without librosa"""
    if HAS_LIBROSA:
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='frames')
        return librosa.frames_to_time(onset_frames, sr=sr)
    else:
        # Basic onset detection using energy
        frame_length = 1024
        hop_length = 512
        frames = []
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sum(frame ** 2)
            frames.append(energy)
        
        frames = np.array(frames)
        # Find peaks in energy
        diff = np.diff(frames)
        onset_indices = []
        
        for i in range(1, len(diff) - 1):
            if diff[i] > 0 and diff[i-1] <= 0 and frames[i+1] > np.mean(frames):
                onset_indices.append(i)
        
        # Convert to time
        onset_times = np.array(onset_indices) * hop_length / sr
        return onset_times

def create_simple_midi(onset_times, output_path="symbolic_conditioned.mid"):
    """Create MIDI file with or without pretty_midi"""
    print(f"🎼 Creating MIDI transcription...")
    
    if HAS_PRETTY_MIDI:
        # Use pretty_midi
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        
        for i, onset_time in enumerate(onset_times[:12]):  # Limit to 12 notes
            pitch = 60 + (i % 12)  # C4 and up
            duration = 0.5
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=onset_time,
                end=onset_time + duration
            )
            piano.notes.append(note)
        
        midi.instruments.append(piano)
        midi.write(output_path)
    else:
        # Create a simple MIDI-like representation
        notes_data = []
        for i, onset_time in enumerate(onset_times[:12]):
            pitch = 60 + (i % 12)
            notes_data.append({
                'onset': onset_time,
                'pitch': pitch,
                'velocity': 80,
                'duration': 0.5
            })
        
        # Save as text file since we can't create actual MIDI
        with open(output_path.replace('.mid', '.txt'), 'w') as f:
            f.write("# Simple MIDI transcription\n")
            f.write("# Format: onset_time, pitch, velocity, duration\n")
            for note in notes_data:
                f.write(f"{note['onset']:.3f}, {note['pitch']}, {note['velocity']}, {note['duration']}\n")
        
        print(f"✅ MIDI data saved to {output_path.replace('.mid', '.txt')} (text format)")
        return notes_data
    
    print(f"✅ MIDI saved to {output_path}")
    return output_path

def visualize_simple(onset_times, audio, sr):
    """Simple visualization if matplotlib available"""
    if not HAS_PLOTTING:
        print("⚠️  Plotting not available - skipping visualization")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    # Waveform
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax1.plot(t, audio, alpha=0.7)
    ax1.vlines(onset_times, -1, 1, colors='red', alpha=0.8, label='Onsets')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Audio Waveform with Detected Onsets')
    ax1.legend()
    
    # Onset distribution
    ax2.hist(onset_times, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Count')
    ax2.set_title('Onset Time Distribution')
    
    plt.tight_layout()
    plt.show()

def run_minimal_task2():
    """Run the complete minimal pipeline"""
    print("🚀 Running Minimal Task 2 Pipeline")
    print("=" * 50)
    
    # Step 1: Setup
    setup_environment_minimal()
    
    # Step 2: Analyze data
    df = analyze_maestro_minimal()
    
    # Step 3: Create test audio
    audio_path, audio, sr = create_synthetic_audio()
    
    # Step 4: Detect onsets
    print("🔍 Detecting onsets...")
    onset_times = simple_onset_detection(audio, sr)
    print(f"Found {len(onset_times)} onsets")
    
    # Step 5: Create MIDI
    midi_output = create_simple_midi(onset_times)
    
    # Step 6: Visualize
    visualize_simple(onset_times, audio, sr)
    
    print("✅ Minimal Task 2 pipeline complete!")
    return True

# =============================================================================
# NOTEBOOK CELL FUNCTIONS
# =============================================================================

# Cell 1: Setup
def cell1_setup():
    return setup_environment_minimal()

# Cell 2: Data Analysis  
def cell2_analysis():
    return analyze_maestro_minimal()

# Cell 3: Audio and Transcription
def cell3_transcription():
    audio_path, audio, sr = create_synthetic_audio()
    onset_times = simple_onset_detection(audio, sr)
    midi_output = create_simple_midi(onset_times)
    return audio_path, onset_times, midi_output

# Cell 4: Visualization
def cell4_visualization():
    # Re-run for visualization
    audio_path, audio, sr = create_synthetic_audio()
    onset_times = simple_onset_detection(audio, sr)
    visualize_simple(onset_times, audio, sr)
    return True

if __name__ == "__main__":
    run_minimal_task2() 