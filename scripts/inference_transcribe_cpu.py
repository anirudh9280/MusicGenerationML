import argparse
import os
import numpy as np
import torch
import librosa
import pretty_midi
from train_transcriber import FrameTranscriber

def extract_crepe_cpu(wav_path, model="tiny"):
    """Extract CREPE on CPU to avoid memory issues"""
    import torchcrepe
    
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    audio = torch.tensor(y, dtype=torch.float32)[None]
    
    # Use CPU and small model
    with torch.no_grad():
        f0, periodicity = torchcrepe.predict(
            audio, sr, model=model, hop_length=160,
            fmin=65.41, fmax=1975.5, device="cpu", return_periodicity=True
        )
    
    features = np.stack([f0[0].numpy(), periodicity[0].numpy()], axis=1)
    return features, len(y)

def frames_to_midi(preds, audio_len, out_midi, hop=160, sr=16000):
    """Convert frame predictions to MIDI"""
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    
    cur_pitch, start_frame = None, None
    
    for t, pred in enumerate(preds):
        pitch = int(pred)
        if pitch != cur_pitch:
            # End previous note
            if cur_pitch is not None and cur_pitch != 88:
                start_time = start_frame * hop / sr
                end_time = t * hop / sr
                note = pretty_midi.Note(velocity=80, pitch=cur_pitch + 21, 
                                      start=start_time, end=end_time)
                piano.notes.append(note)
            
            # Start new note
            if pitch != 88:  # Not silence
                start_frame = t
            cur_pitch = pitch
    
    # Handle final note
    if cur_pitch is not None and cur_pitch != 88:
        start_time = start_frame * hop / sr
        end_time = audio_len / sr
        note = pretty_midi.Note(velocity=80, pitch=cur_pitch + 21,
                              start=start_time, end=end_time)
        piano.notes.append(note)
    
    pm.instruments.append(piano)
    pm.write(out_midi)
    print(f"Generated MIDI: {out_midi} ({len(piano.notes)} notes)")

def main():
    model = FrameTranscriber(hidden_dim=128, num_layers=2, dropout=0.3)
    model.load_state_dict(torch.load("/root/MusicGenerationML/transcriber_best.pt", map_location="cpu"))
    model.eval()
    
    # Extract CREPE features
    features, audio_len = extract_crepe_cpu("/root/MusicGenerationML/data/maestro_2004/2004/2004_wav.midi.wav")
    
    # Run inference
    with torch.no_grad():
        feats_tensor = torch.from_numpy(features)[None]  # Add batch dim
        logits = model(feats_tensor)
        preds = logits.argmax(dim=2)[0].numpy()
    
    # Generate MIDI
    frames_to_midi(preds, audio_len, "/root/MusicGenerationML/symbolic_conditioned.mid")

if __name__ == "__main__":
    main()
