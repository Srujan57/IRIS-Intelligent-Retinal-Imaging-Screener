"""
augmentation.py

Applies image augmentation to preprocessed OCT scans during training.
Transforms include:
  - Horizontal flip
  - Random rotation (+/- 10 degrees)
  - Brightness and contrast jitter
  - Gaussian noise injection

Used in the training loop to improve generalization and
address class imbalance between healthy and ED-pattern samples.
"""
