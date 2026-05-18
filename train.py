"""
train.py

Training loop for both stages:
  - Stage 1: U-Net segmentation model (models/unet.py)
  - Stage 2: Classifier on extracted thickness features (models/classifier.py)

Loads hyperparameters from config.yaml.
Logs metrics to Weights & Biases.
"""
