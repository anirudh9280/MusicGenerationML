# Music Generation ML Assignment

This repository implements **Tasks 2 & 4** from the CSE153 Music Generation assignment:

- **Task 2:** Symbolic Conditioned Generation (Onsets and Frames)
- **Task 4:** Continuous Conditioned Generation (Musika)

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Git
- ~8GB disk space for datasets

### Setup

1. **Clone and initialize submodules:**

```bash
git clone <your-repo-url>
cd MusicGenerationML
```

The repository already includes Magenta and Musika as Git submodules in `libs/`.

2. **Create virtual environments:**

**Task 2 Environment (TensorFlow + Magenta):**

```bash
python3 -m venv ~/env_task2
source ~/env_task2/bin/activate
pip install --upgrade pip
pip install tensorflow==2.16.1
pip install ipykernel
python -m ipykernel install --user --name=task2_env --display-name "Task2 TensorFlow"
deactivate
```

**Task 4 Environment (PyTorch + Musika):**

```bash
python3 -m venv ~/env_task4
source ~/env_task4/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install librosa matplotlib scipy tensorboard tqdm pydub huggingface-hub
pip install jupyterlab ipykernel
python -m ipykernel install --user --name=task4_env --display-name "Task4 PyTorch"
```

3. **Launch Jupyter Lab:**

```bash
source ~/env_task4/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Open http://localhost:8888 in your browser.

## 📁 Repository Structure

```
MusicGenerationML/
├── libs/                          # Git submodules
│   ├── magenta/                   # Magenta library for Task 2
│   └── musika/                    # Musika library for Task 4
├── Task2_OnsetsFrames.ipynb       # Task 2 notebook
├── Task4_Musika.ipynb            # Task 4 notebook
├── data/                         # Datasets (create this)
├── results/                      # Generated outputs
├── setup_assignment.sh           # Automated setup script
└── README.md                     # This file
```

## 🎵 Tasks Overview

### Task 2: Symbolic Conditioned Generation

- **Model:** Onsets and Frames (Magenta)
- **Input:** Audio recordings
- **Output:** MIDI transcriptions
- **Dataset:** MAESTRO
- **Kernel:** "Task2 TensorFlow"
- **Output File:** `symbolic_conditioned.mid`

### Task 4: Continuous Conditioned Generation

- **Model:** Musika (GAN-based)
- **Input:** Text/style conditioning
- **Output:** High-quality audio
- **Dataset:** Custom music dataset
- **Kernel:** "Task4 PyTorch"
- **Output File:** `continuous_conditioned.mp3`

## 📊 Datasets

### MAESTRO (Task 2)

```bash
cd data
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip maestro-v3.0.0-midi.zip
```

### Music Dataset (Task 4)

Follow instructions in the Task4 notebook for dataset preparation.

## 🛠 Development Workflow

1. **Start Jupyter Lab** with the Task 4 environment
2. **Open notebooks** and select appropriate kernels:
   - `Task2_OnsetsFrames.ipynb` → "Task2 TensorFlow"
   - `Task4_Musika.ipynb` → "Task4 PyTorch"
3. **Run experiments** following the 4-section structure:
   - Data analysis and preprocessing
   - Modeling
   - Evaluation
   - Related work discussion
4. **Generate outputs:**
   - `symbolic_conditioned.mid` (Task 2)
   - `continuous_conditioned.mp3` (Task 4)

## 📝 Assignment Requirements

### Deliverables

1. **Jupyter Notebook** (exported as HTML)
2. **Video Presentation** (~20 minutes)
3. **Generated Music Files**
   - `symbolic_conditioned.mid`
   - `continuous_conditioned.mp3`

### Presentation Structure (per task)

1. **Data Analysis:** Dataset exploration, preprocessing
2. **Modeling:** Architecture, implementation details
3. **Evaluation:** Metrics, baselines, results
4. **Related Work:** Literature review, comparisons

## 🚨 Troubleshooting

### Common Issues

**Environment conflicts:**

- Use separate virtual environments for each task
- TensorFlow (Task 2) and PyTorch (Task 4) can conflict

**Missing dependencies:**

- Some older packages may have compatibility issues
- Install core packages first, then add others as needed

**Kernel not showing in Jupyter:**

- Ensure you've run `python -m ipykernel install` in each environment
- Refresh Jupyter Lab browser page

**Memory issues:**

- Music models are memory-intensive
- Close other applications if needed
- Consider using smaller dataset samples for testing

### Getting Help

1. Check the notebook cell outputs for error details
2. Verify virtual environment activation
3. Ensure all required packages are installed
4. Review the assignment instructions for clarification

## 🎼 Expected Outputs

### Task 2 (Onsets and Frames)

- High-quality MIDI transcription of audio input
- Evaluation metrics showing transcription accuracy
- Comparison with baseline methods

### Task 4 (Musika)

- Generated audio responding to conditioning
- Quality metrics and subjective evaluation
- Demonstration of controllable generation

## 📚 References

- [Magenta Project](https://magenta.tensorflow.org/)
- [Musika Paper](https://arxiv.org/abs/2208.08706)
- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
- [Onsets and Frames Paper](https://arxiv.org/abs/1710.11153)

---

**Note:** This setup follows the assignment instructions for creating isolated environments with proper Jupyter kernel registration. Each task has its own environment to avoid dependency conflicts between TensorFlow and PyTorch.
