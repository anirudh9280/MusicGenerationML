#!/bin/bash

# Vast.ai Setup Commands for Tasks 2 & 4
# Run these commands after SSH'ing into your Vast.ai instance

echo "=== Setting up Tasks 2 & 4 Environment on Vast.ai ==="

# 1. Create project structure
mkdir -p ~/MusicGenerationML/data/{maestro_tfrecords,edm_hse}
cd ~/MusicGenerationML

# 2. Setup Task 2 - MAESTRO TFRecord Data (Choose Option A or B)

echo "=== Task 2 Setup: MAESTRO Dataset ==="

# OPTION A: Download fresh on Vast.ai (RECOMMENDED - faster than upload)
echo "Option A: Downloading MAESTRO TFRecords directly on Vast.ai..."
cd data/maestro_tfrecords

# Install Google Cloud SDK if not available
if ! command -v gsutil &> /dev/null; then
    echo "Installing Google Cloud SDK..."
    curl https://sdk.cloud.google.com | bash
    exec -l $SHELL  # Reload shell
    source ~/google-cloud-sdk/path.bash.inc
fi

# Download the same TFRecord files we got locally
echo "Downloading MAESTRO TFRecord files (~16GB)..."
gsutil -m cp \
  "gs://magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0_ns_wav_train.tfrecord-0000[0-4]-of-00025" \
  "gs://magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0_ns_wav_validation.tfrecord-00000-of-00025" \
  "gs://magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0_ns_wav_test.tfrecord-00000-of-00025" \
  .

# Also download the CSV metadata
gsutil cp "gs://magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv" .

echo "MAESTRO TFRecord download complete!"

# OPTION B: Upload from local machine (uncomment if you prefer this)
# echo "Option B: Use this command from your LOCAL machine to upload:"
# echo "scp -r data/maestro_tfrecords/ root@<vast-ip>:~/MusicGenerationML/data/"
# echo "Replace <vast-ip> with your actual Vast.ai instance IP"
# echo "Then press Enter here to continue..."
# read -p "Press Enter after upload completes..."

cd ~/MusicGenerationML

# 3. Setup Task 4 - EDM Dataset

echo "=== Task 4 Setup: EDM Dataset ==="
cd data/edm_hse

echo "Downloading EDM-HSE dataset (~2.5GB)..."
wget -O edm_dataset.zip "https://zenodo.org/record/4740544/files/EDM-HSE.zip?download=1"

# Alternative if above doesn't work
if [ ! -f edm_dataset.zip ] || [ ! -s edm_dataset.zip ]; then
    echo "Trying alternative download method..."
    curl -L -o edm_dataset.zip "https://zenodo.org/record/4740544/files/EDM-HSE.zip"
fi

echo "Extracting EDM dataset..."
unzip edm_dataset.zip
rm edm_dataset.zip

echo "EDM dataset setup complete!"

# 4. Install Dependencies for Both Tasks

echo "=== Installing Python Dependencies ==="
cd ~/MusicGenerationML

# Task 2 dependencies (TensorFlow + Magenta)
pip install tensorflow==2.16.1 magenta librosa pretty_midi

# Task 4 dependencies (PyTorch + Audio)
pip install torch torchaudio librosa soundfile scipy numpy matplotlib wandb

# Common dependencies
pip install jupyter pandas seaborn matplotlib numpy scipy

# 5. Verify Setup

echo "=== Verifying Setup ==="

# Check datasets
echo "Task 2 - MAESTRO TFRecords:"
ls -lh data/maestro_tfrecords/
echo "Total size:"
du -sh data/maestro_tfrecords/

echo ""
echo "Task 4 - EDM Dataset:"
ls -lh data/edm_hse/
du -sh data/edm_hse/

# Check GPU and dependencies
echo ""
echo "=== System Check ==="
python3 -c "
import tensorflow as tf
import torch
import librosa
import magenta

print('=== Task 2 (TensorFlow/Magenta) ===')
print('TensorFlow version:', tf.__version__)
print('TensorFlow GPU available:', tf.test.is_gpu_available())
print('Magenta version:', magenta.__version__)

print('\n=== Task 4 (PyTorch) ===')
print('PyTorch version:', torch.__version__)
print('PyTorch GPU available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
print('LibROSA version:', librosa.__version__)

print('\n=== Ready for both tasks! ===')
"

echo "=== Setup Complete! ==="
echo "Task 2 data: ~/MusicGenerationML/data/maestro_tfrecords/"
echo "Task 4 data: ~/MusicGenerationML/data/edm_hse/"
echo "You can now start working on both notebooks!" 