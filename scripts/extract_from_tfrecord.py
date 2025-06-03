import tensorflow as tf
import soundfile as sf
import pretty_midi
import os, glob
import numpy as np

# 1. Set paths
tfrec_dir = "/root/MusicGenerationML/data/maestro_tfrecords"
out_wav_dir  = "/root/MusicGenerationML/data/maestro_2004/2004"
out_midi_dir = "/root/MusicGenerationML/data/maestro_2004/2004"

# 2. Make output folder if it doesn't exist
os.makedirs(out_wav_dir,  exist_ok=True)
os.makedirs(out_midi_dir, exist_ok=True)

# 3. Correct TFRecord feature spec based on inspection
feature_description = {
    "audio":         tf.io.FixedLenFeature([], tf.string),
    "sequence":      tf.io.FixedLenFeature([], tf.string),
    "id":            tf.io.FixedLenFeature([], tf.string),
    "velocity_range": tf.io.FixedLenFeature([], tf.string),
}

def parse_function(proto):
    return tf.io.parse_single_example(proto, feature_description)

# 4. Read all TFRecord shards
tf_files = glob.glob(os.path.join(tfrec_dir, "*.tfrecord-*"))
print(f"Found {len(tf_files)} TFRecord files")

count = 0
for tf_file in tf_files:
    print(f"Processing {tf_file}")
    dataset = tf.data.TFRecordDataset(tf_file).map(parse_function)
    
    for example in dataset:
        try:
            # Extract ID for filename
            id_str = example["id"].numpy().decode("utf-8")
            
            # Clean up ID to create filename
            base_name = id_str.replace("/", "_").replace("\\", "_")
            wav_out  = os.path.join(out_wav_dir,  base_name + ".wav")
            midi_out = os.path.join(out_midi_dir, base_name + ".midi")
            
            # Extract audio bytes and decode
            audio_bytes = example["audio"].numpy()
            audio = tf.audio.decode_wav(audio_bytes, desired_channels=1, desired_samples=-1)
            audio_data = audio.audio.numpy().flatten()
            sample_rate = audio.sample_rate.numpy()
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                try:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                except ImportError:
                    print(f"Warning: librosa not available, keeping original sample rate {sample_rate}")
            
            # Write WAV
            sf.write(wav_out, audio_data, sample_rate)
            
            # Extract MIDI sequence
            sequence_bytes = example["sequence"].numpy()
            # The sequence is likely a serialized MIDI or note sequence
            # For now, let's try to parse it as a MIDI file directly
            try:
                # Try writing as MIDI bytes directly
                with open(midi_out, 'wb') as f:
                    f.write(sequence_bytes)
                
                # Verify it's a valid MIDI by trying to read it
                pm_test = pretty_midi.PrettyMIDI(midi_out)
            except:
                # If that fails, the sequence might be in a different format
                # Create a simple MIDI with no notes as placeholder
                pm = pretty_midi.PrettyMIDI()
                inst = pretty_midi.Instrument(program=0)
                pm.instruments.append(inst)
                pm.write(midi_out)
                print(f"Warning: Could not parse MIDI sequence for {id_str}, created empty MIDI")
            
            count += 1
            if count % 10 == 0:
                print(f"Extracted {count} examples…")
                
        except Exception as e:
            print(f"Error processing example {count}: {e}")
            continue
    
    print(f"Done shard {tf_file}")

print(f"Finished extracting {count} WAV+MIDI files to: {out_wav_dir}")
