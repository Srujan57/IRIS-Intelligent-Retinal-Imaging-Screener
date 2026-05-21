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

from torchvision import transforms


def get_train_transform(config):
    return transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[config["normalize_mean"]],
            std=[config["normalize_std"]]
        )
    ])


def get_val_transform(config):
    return transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[config["normalize_mean"]],
            std=[config["normalize_std"]]
        )
    ])
