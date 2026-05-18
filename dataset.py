"""
dataset.py

PyTorch Dataset class for loading preprocessed OCT scans.
Reads image paths and labels from data/labels.csv,
applies train/val/test splits from data/splits.csv,
and feeds samples into the DataLoader for training and evaluation.
"""
