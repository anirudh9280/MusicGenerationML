#!/bin/bash
set -e

echo "========================================="
echo "Music Generation ML Assignment Setup"
echo "Tasks 2 & 4: Symbolic + Continuous Generation"
echo "========================================="

# Create main directory structure
echo "Creating directory structure..."
mkdir -p task2_symbolic_conditioned
mkdir -p task4_continuous_conditioned
mkdir -p data
mkdir -p notebooks
mkdir -p results

echo "Directory structure created!"

# Clone repositories as submodules
echo "Cloning Magenta for Task 2 (Symbolic Generation)..."
if [ ! -d "task2_symbolic_conditioned/magenta" ]; then
    cd task2_symbolic_conditioned
    git clone https://github.com/magenta/magenta.git
    cd ..
fi

echo "Cloning Musika for Task 4 (Continuous Generation)..."
if [ ! -d "task4_continuous_conditioned/musika" ]; then
    cd task4_continuous_conditioned
    git clone https://github.com/marcoppasini/musika.git
    cd ..
fi

# Setup Python environments
echo "Setting up Python environment for Task 2 (TensorFlow/Magenta)..."
if [ ! -d "venv_task2" ]; then
    python3 -m venv venv_task2
fi

echo "Setting up Python environment for Task 4 (PyTorch/Musika)..."
if [ ! -d "venv_task4" ]; then
    python3 -m venv venv_task4
fi

# Install Task 2 dependencies (TensorFlow/Magenta)
echo "Installing Task 2 dependencies..."
source venv_task2/bin/activate
pip install --upgrade pip
pip install tensorflow==2.12.0
pip install magenta
pip install pretty_midi
pip install mir_eval
pip install librosa
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
pip install jupyter
pip install ipykernel
deactivate

# Install Task 4 dependencies (PyTorch/Musika)
echo "Installing Task 4 dependencies..."
source venv_task4/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install librosa
pip install soundfile
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
pip install jupyter
pip install ipykernel
pip install wandb
pip install hydra-core
deactivate

# Register Jupyter kernels
echo "Registering Jupyter kernels..."
source venv_task2/bin/activate
python -m ipykernel install --user --name=task2_magenta --display-name="Task 2 - Magenta (TensorFlow)"
deactivate

source venv_task4/bin/activate
python -m ipykernel install --user --name=task4_musika --display-name="Task 4 - Musika (PyTorch)"
deactivate

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Download MAESTRO dataset for Task 2:"
echo "   cd data && wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
echo ""
echo "2. Download EDM dataset for Task 4:"
echo "   cd data && wget [EDM dataset URL]"
echo ""
echo "3. Start Jupyter notebook:"
echo "   jupyter notebook --port=8888 --no-browser --allow-root"
echo ""
echo "Kernels available:"
echo "- Task 2 - Magenta (TensorFlow)"
echo "- Task 4 - Musika (PyTorch)" 