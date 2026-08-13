# FSDL 2022 — ML Engineering Practices Lab

A self-contained notebook covering essential software engineering practices for machine learning projects, built around a minimal PyTorch training pipeline. Based on Lab 05 of the Full Stack Deep Learning (FSDL) 2022 course.

## Overview

This notebook trains a small neural network on a toy classification task, then walks through a series of practical ML engineering concepts that go beyond just writing model code — covering code quality, testing, and performance profiling.

## What's Inside

- **Toy training pipeline** — a 2-layer MLP trained on randomly generated data using PyTorch, Adam optimizer, and Cross-Entropy loss
- **Code formatting** — using `black` to auto-format messy Python code
- **Linting** — using `flake8` to catch style issues and potential bugs
- **Simulated pre-commit checks** — chaining formatting and linting into a single quality-check function
- **Shell scripting** — writing and running basic and error-safe (`set -euxo pipefail`) Bash scripts
- **Unit testing** — writing tests with `pytest`
- **Doctests** — inline example-based tests within function docstrings
- **Dataset sanity checks** — verifying data shape and type correctness
- **Memorization test** — overfitting a model on a tiny data subset to confirm it can learn at all
- **Performance profiling** — using PyTorch's built-in `torch.profiler` to measure CPU time per operation
- **TensorBoard integration** — exporting profiler traces for visual analysis
- **DataLoader benchmarking** — comparing training data loading speed across different `num_workers` settings (0, 2, 4)

## Requirements

- Python 3.x
- PyTorch, TorchVision, TorchAudio
- `pytest`, `black`, `flake8`
- `tensorboard`

Install dependencies:
```bash
pip install torch torchvision torchaudio pytest black flake8 tensorboard
```

## Usage

Open the notebook in Jupyter or Google Colab and run the cells sequentially. Each section is labeled by concept and can be run independently after the initial setup and training cells.

## Purpose

This notebook was built as a practical, hands-on reference for the "unglamorous but essential" parts of shipping ML code — formatting, linting, testing, and profiling — rather than model architecture or training tricks.
