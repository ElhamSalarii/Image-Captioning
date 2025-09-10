# Image Captioning with CNN-RNN Architecture — Repository Description
This repository contains an implementation framework for an image captioning system using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The system takes images as input and generates natural language descriptions of their content.

Image captioning sits at the intersection of computer vision and natural language processing, combining feature extraction from images with sequence modeling for text generation. This implementation follows the encoder-decoder architecture:


## 1) Project Overview

**Goal:** Provide a complete workflow for building, training, and evaluating an image captioning model using a CNN encoder and an RNN/LSTM decoder.


- An encoder (CNN) extracts visual features from input images
- A decoder (RNN/LSTM/GRU) generates captions word-by-word based on these features

- **Task:** Image captioning – generate descriptive sentences for given images.
- **Architecture:** CNN encoder + Recurrent Neural Network (RNN, e.g., LSTM/GRU) decoder.
- **Dataset (examples):** This project uses the Flickr8k dataset, which contains:

Approximately 8,000 images
5 different captions for each image (40,000 captions total)
A diverse range of scenes, objects, and actions
The download_flickr.py script handles:

Downloading the images and captions
Preprocessing captions (cleaning, normalization)
Creating train/validation/test splits
Organizing files in the expected directory structure
- **Output:** Trained model checkpoints, generated captions, and evaluation metrics (BLEU, METEOR, CIDEr, ROUGE).

> Tip: If you have a specific dataset, replace the example paths and dataset name in the configs.

Student TODO Sections
Throughout this project, you'll need to implement various components marked with TODO comments. Here's a summary of what you'll be working on:

Core Implementation Tasks
Data Processing (download_flickr.py):

Process captions from the Flickr8k dataset
Create train/val/test splits
Encoder (encoder.py):

Initialize CNN backbones (ResNet, MobileNet)
Create projection layers for feature vectors
Decoder (decoder.py):

Implement the RNN/LSTM/GRU decoder
Create word embedding layers
Implement the caption generation logic with teacher forcing
Implement greedy decoding for inference
Caption Model (caption_model.py):

Integrate encoder and decoder
Implement the forward pass
Implement caption generation
Data Utilities:

Build vocabulary and tokenization functions (vocabulary.py)
Create dataset loaders and transformations (dataset.py)
Implement evaluation metrics (metrics.py)
Create training and validation loops (trainer.py)
Notebook Implementation Tasks
Each notebook contains specific TODOs:

Data Exploration: Implement functions to visualize and analyze the dataset
Feature Extraction: Implement feature extraction using pre-trained CNNs
Model Training: Implement model training, parameter counting, and caption generation
Evaluation: Implement metrics calculation and results visualization

## 2) Key Features

- CNN-RNN encoder-decoder architecture for image captioning
- Image feature extraction with a pre-trained CNN (e.g., ResNet-50) fine-tuned or frozen
- Flexible decoder (LSTM/GRU) with attention (optional)
- End-to-end training loop with data loading, batching, and logging
- Inference script to generate captions for new images
- Evaluation utilities for standard captioning metrics (BLEU, CIDEr, METEOR, ROUGE)
- Reproducibility: config-driven experiments, logging, and checkpoints

## 3) Quick Start

### Prerequisites
- Python >= 3.7
- PyTorch or TensorFlow (depending on the implementation)
- CUDA toolkit (optional but recommended for GPU acceleration)
- 
### Installation
- 1) Clone this repository:

- git clone
  ```bash
 https://github.com/Mound21k/image-captioning.git

cd image-captioning
 ```

 - Plain text
- Create a virtual environment and install dependencies:

- python -m venv venv
- source venv/bin/activate  # On Windows: venv\Scripts\activate
- pip install -r requirements.txt
- Plain text
- Download the Flickr8k dataset:

- python data/download_flickr.py --data_dir ./data

### Running a Baseline

- Train the model:
```bash
python train_captioning.py
```

- Validate / Evaluate:
```bash
python evaluate_captioning.py --split val
```

- Generate captions for new images:
```bash
python generate_caption.py --images path/to/images --output outputs/captions.json
```

> Tip: Use a config file (config.yaml or config.json) to switch datasets, model sizes, learning rates, and hyperparameters easily.

## 4) Configuration and Reproducibility

- **Configuration:** Use a minimal, readable config (e.g., `config.yaml` or `config.json`) that includes:
  - Model architecture and hyperparameters (encoder type, decoder type, embedding size, hidden size)
  - Attention mechanism (if used) and tokenizer settings
  - Dataset paths (train/val/test) and preprocessing options
  - Training hyperparameters (epochs, batch_size, learning_rate)
- **Seed:** Set a fixed random seed for reproducibility.
- **Logging:** Log training progress, validation metrics, and system info (CUDA version, PyTorch/TensorFlow version).

Example YAML snippet:
```yaml
model:
  encoder: "resnet50"
  decoder: "lstm"
  embed_size: 256
  hidden_size: 512
  attention: true
training:
  epochs: 20
  batch_size: 64
  learning_rate: 1e-4
data:
  train_path: "data/coco/train.json"
  val_path: "data/coco/val.json"
  image_folder: "data/coco/images"
tokenizer:
  vocab_size: 10000
  max_seq_length: 20
```

## 5) Directory Structure (high level)

- `train_captioning.py` – training entry point
- `evaluate_captioning.py` – evaluation scripts and metrics
- `generate_caption.py` – inference script
- `data/` – datasets and preprocessing
- `models/` – saved model checkpoints
- `outputs/` – generated captions and logs
- `docs/` – optional documentation
- `requirements.txt` – dependencies
- `README.md` – this file

## 6) Evaluation Metrics

- BLEU, METEOR, CIDEr, ROUGE

## 7) refrences
- 1. [DL4032_ElhamSalari-403155005_HW03.pdf](https://github.com/user-attachments/files/22248140/DL4032_ElhamSalari-403155005_HW03.pdf)
-  2. 
[Uploading xuc15.pdf…]()
