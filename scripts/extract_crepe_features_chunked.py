import os
import argparse
import glob
import numpy as np
import torch
import torchcrepe
import librosa

def extract_for_wav_chunked(wav_path, npz_out, model="tiny", device="cuda", chunk_duration=30):
    """Extract CREPE features in chunks to avoid GPU memory issues"""
    print(f"Extracting (chunked): {os.path.relpath(wav_path)}")
    
    # Load full audio
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    
    # Calculate chunk size
    chunk_samples = chunk_duration * sr
    hop_length = 160
    
    all_f0 = []
    all_conf = []
    
    # Process in chunks
    for start in range(0, len(y), chunk_samples):
        end = min(start + chunk_samples, len(y))
        chunk = y[start:end]
        
        if len(chunk) < hop_length:
            break
            
        # Convert to tensor
        audio_tensor = torch.tensor(chunk, dtype=torch.float32)[None].to(device)
        
        # Extract CREPE features for this chunk
        with torch.no_grad():
            f0, periodicity = torchcrepe.predict(
                audio_tensor, sr, model=model, hop_length=hop_length,
                fmin=65.41, fmax=1975.5, device=device, return_periodicity=True
            )
        
        all_f0.append(f0[0].cpu().numpy())
        all_conf.append(periodicity[0].cpu().numpy())
        
        # Clear cache after each chunk
        torch.cuda.empty_cache()
    
    # Concatenate all chunks
    f0_full = np.concatenate(all_f0)
    conf_full = np.concatenate(all_conf)
    
    # Save
    np.savez(npz_out, f0=f0_full, conf=conf_full)
    print(f"  Saved {npz_out} with {len(f0_full)} frames")

def main(args):
    wav_pattern = os.path.join(args.input_dir, "**", "*.wav")
    wav_files = glob.glob(wav_pattern, recursive=True)
    
    for wav_path in wav_files:
        rel_path = os.path.relpath(wav_path, args.input_dir)
        base_name = os.path.splitext(rel_path)[0]
        npz_out = os.path.join(args.output_dir, base_name + ".npz")
        
        os.makedirs(os.path.dirname(npz_out), exist_ok=True)
        
        if os.path.exists(npz_out):
            print(f"Skipping existing: {npz_out}")
            continue
            
        extract_for_wav_chunked(wav_path, npz_out, model=args.crepe_model, device=args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--crepe_model", default="tiny")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args)
