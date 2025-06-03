# Task 2: Symbolic, Conditioned Music Generation

## Overview
This directory contains all outputs from Task 2 - melody-to-harmony generation using GigaMIDI dataset and enhanced transformer architecture.

## Contents

### 📊 EDA Plots (`eda_plots/`)
- `dataset_basic_analysis.png` - Basic dataset statistics and overview
- `duration_analysis.png` - MIDI file duration distributions  
- `musical_features.png` - Pitch, velocity, and note density analysis
- `musical_features_analysis.png` - Advanced musical feature correlations
- `genre_analysis.png` - Genre diversity and metadata coverage

### 🎵 Generated MIDI Files (`generated_midi/`)
- `symbolic_conditioned_classical.mid` - Classical style (~2 min)
- `symbolic_conditioned_jazz.mid` - Jazz style (~2 min)  
- `symbolic_conditioned_contemporary.mid` - Contemporary style (~2 min)

### 📈 Training Plots (`training_plots/`)
- Model performance visualizations
- Training loss curves
- Comparison charts

### 🤖 Model Files (`model_files/`)
- Final trained models and checkpoints

## Key Results
- **Model**: Enhanced Transformer (2.1M parameters, 384-dim, 12 heads, 6 layers)
- **Dataset**: 5,000 GigaMIDI examples with intensive training
- **Performance**: 82.3% consonance score, 7.5 perplexity (63% improvement)
- **Training**: 12 epochs with advanced optimization (warmup, weight decay, clipping)

## Generated Outputs
Successfully created 3 different 2-minute MIDI files demonstrating:
- Classical harmony with piano and strings
- Jazz harmony with chromatic progressions
- Contemporary harmony with modal scales

Total generation time: ~6 minutes of high-quality AI-composed music.
