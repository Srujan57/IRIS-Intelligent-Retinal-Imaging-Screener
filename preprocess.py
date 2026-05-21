"""
preprocess.py

Loads raw OCT scans, resizes to config image_size,
normalizes pixel values, and saves processed images
ready for segmentation.
"""
import os
import torch
import yaml
from PIL import Image
from torchvision import transforms


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_transform(config):
    return transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[config["normalize_mean"]],
            std=[config["normalize_std"]]
        )
    ])


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("L")
    return transform(image)


def preprocess_folder(input_dir, output_dir, transform):
    os.makedirs(output_dir, exist_ok=True)

    processed = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir,
                filename.replace(".jpg", ".pt")
                        .replace(".jpeg", ".pt")
                        .replace(".png", ".pt")
            )
            tensor = preprocess_image(input_path, transform)
            torch.save(tensor, output_path)
            processed += 1

    print(f"Done: {processed} images processed → {output_dir}")


if os.path.exists("/kaggle/input"):
    KERMANY_DIR = "/kaggle/input/kermany2018/"
    OCT5K_DIR   = "/kaggle/input/oct5k-iris/"
else:
    KERMANY_DIR = "data/kermany/"
    OCT5K_DIR   = "data/oct5k/"


if __name__ == "__main__":
    config = load_config()
    transform = get_transform(config)

    preprocess_folder(KERMANY_DIR, "data/processed/kermany", transform)
    preprocess_folder(OCT5K_DIR,   "data/processed/oct5k",   transform)
