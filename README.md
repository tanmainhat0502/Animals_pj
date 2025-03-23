# Animals Classification Project

This project is a deep learning framework for classifying animal images using various convolutional neural network (CNN) architectures. It supports multiple backbones such as ResNet, Inception, VGG, AlexNet, EfficientNet, and MobileNet, and includes tools for data preparation, model training, evaluation, and inference.

## Project Overview

- **Goal**: Build a flexible and extensible system to classify animals from images.
- **Features**:
  - Custom dataset splitting (train/val/test).
  - Multiple CNN backbones implemented from scratch.
  - Configurable hyperparameters and image transformations.
  - Training with logging (TensorBoard) and checkpoint saving.
  - Evaluation with accuracy/loss visualization.
  - Inference on single images.

## Project Structure
Animals_pj/
├── configs/                 
│   ├── dataset.yaml         
│   ├── transforms.yaml     
│   ├── hyperparameters.yaml 
├── data/                  
│   ├── raw/               
│   ├── processed/
├── models
│   ├── backbones.py
│   ├── efficientnet.py
│   ├── inceptionnet.py
│   ├── mobilenet.py
│   ├── model.py
│   └── resnet.py
├── scripts
│   ├── evaluate.py
│   ├── inference.py
│   ├── prepare_data.py
│   └── train.py
└── utils
    ├── data_utils.py
    ├── logging.py
    └── transforms.py
