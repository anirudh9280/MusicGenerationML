# Task 2 Scripts

## Usage Pipeline:
1. `extract_from_tfrecord.py` - Extract WAV/MIDI from TFRecords
2. `extract_crepe_features.py` - Extract CREPE features from audio
3. `build_frame_targets.py` - Convert MIDI to frame-level labels  
4. `train_transcriber.py` - Train Bi-LSTM transcriber
5. `inference_transcribe.py` - Generate symbolic_conditioned.mid

## Custom scripts created:
- `extract_from_tfrecord.py` - Handle MAESTRO TFRecord format
- `inspect_tfrecord.py` - Debug TFRecord structure
