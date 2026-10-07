# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the Hierarchical Reasoning Model (HRM) repository - a novel recurrent neural architecture for complex reasoning tasks. HRM uses two interdependent recurrent modules: a high-level module for slow, abstract planning and a low-level module for rapid, detailed computations. The model is designed to solve challenging tasks like Sudoku puzzles, maze navigation, and the Abstraction and Reasoning Corpus (ARC) with minimal training data.

## Core Commands

### Training
```bash
# Single GPU training (laptop/smaller setup)
OMP_NUM_THREADS=8 python pretrain.py data_path=<DATA_PATH> epochs=<EPOCHS> eval_interval=<INTERVAL> global_batch_size=<BATCH_SIZE> lr=<LR> puzzle_emb_lr=<PUZZLE_LR>

# Multi-GPU distributed training (8-GPU setup)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py [args...]

# Quick Sudoku demo (single GPU)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

### Dataset Preparation
```bash
# ARC datasets
python dataset/build_arc_dataset.py  # ARC-1 (960 examples)
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000  # ARC-2 (1120 examples)

# Sudoku datasets
python dataset/build_sudoku_dataset.py  # Full version
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000  # 1000 examples

# Maze dataset
python dataset/build_maze_dataset.py  # 1000 examples
```

### Evaluation
```bash
# Evaluate trained model
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>

# Use arc_eval.ipynb notebook for ARC-specific evaluation and result inspection
```

### Dependencies Installation
```bash
pip install -r requirements.txt

# Flash Attention (required for CUDA extensions)
# For Hopper GPUs:
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention/hopper && python setup.py install

# For Ampere or earlier GPUs:
pip3 install flash-attn
```

## Architecture Overview

### Core Components

**Models Structure:**
- `models/hrm/hrm_act_v1.py` - Main HRM architecture with hierarchical reasoning
- `models/layers.py` - Core neural network layers (attention, embeddings, etc.)
- `models/losses.py` - Loss functions including ACT (Adaptive Computation Time)
- `models/sparse_embedding.py` - Sparse embedding implementations
- `models/common.py` - Shared model utilities

**Key Architecture Features:**
- **H_cycles/L_cycles**: High-level and low-level reasoning cycles
- **H_layers/L_layers**: Number of transformer layers for each level
- **ACT (Adaptive Computation Time)**: Dynamic halting mechanism
- **Hierarchical Processing**: Two-level reasoning with different timescales

### Configuration System

Uses Hydra for configuration management:
- `config/cfg_pretrain.yaml` - Main training configuration
- `config/arch/hrm_v1.yaml` - Architecture-specific settings
- Configuration is hierarchical and supports overrides via command line

### Dataset Architecture

**Dataset Classes:**
- `puzzle_dataset.py` - Core dataset loading and batching
- `dataset/build_*.py` - Dataset builders for different puzzle types
- `dataset/common.py` - Shared dataset utilities

**Supported Puzzle Types:**
- ARC (Abstraction and Reasoning Corpus) - Pattern recognition
- Sudoku - Constraint satisfaction
- Maze - Path planning

### Training Pipeline

**Key Files:**
- `pretrain.py` - Main training script with distributed support
- `evaluate.py` - Model evaluation script
- Uses W&B for experiment tracking (`wandb login` required)

**Training Features:**
- Distributed training support via `torchrun`
- Automatic checkpointing during evaluation intervals
- Adaptive learning rate scheduling
- Mixed precision training support

## Important Notes

- **CUDA Required**: Project requires CUDA extensions and Flash Attention
- **Small Sample Learning**: Models can achieve high performance with ~1000 training examples
- **Variance**: Small-sample learning has ±2 point accuracy variance
- **Early Stopping**: For extreme datasets, use early stopping when training accuracy approaches 100% to avoid numerical instability
- **Memory**: Multi-GPU setups assume 8-GPU configuration; adjust `global_batch_size` for different setups

## Checkpoints

Pre-trained checkpoints available on HuggingFace:
- ARC-AGI-2: `sapientinc/HRM-checkpoint-ARC-2`
- Sudoku Extreme: `sapientinc/HRM-checkpoint-sudoku-extreme`  
- Maze 30x30: `sapientinc/HRM-checkpoint-maze-30x30-hard`