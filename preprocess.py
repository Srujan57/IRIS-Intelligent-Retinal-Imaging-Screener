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


def resolve_dataset_dir(kaggle_path, local_path):
    """
    Resolve each dataset's input directory independently, rather than
    switching both datasets on a single global Kaggle/local flag.
    This lets datasets live in different places at once — e.g. a large
    dataset processed on Kaggle and a smaller one processed locally.
    """
    return kaggle_path if os.path.exists(kaggle_path) else local_path


KERMANY_DIR = resolve_dataset_dir("/kaggle/input/kermany2018/", "data/kermany/")
OCT5K_DIR   = resolve_dataset_dir("/kaggle/input/oct5k-iris/", "data/oct5k/")

DATASETS = [
    ("kermany", KERMANY_DIR, "data/processed/kermany"),
    ("oct5k",   OCT5K_DIR,   "data/processed/oct5k"),
]


if __name__ == "__main__":
    config = load_config()
    transform = get_transform(config)

    for name, input_dir, output_dir in DATASETS:
        if os.path.isdir(input_dir):
            preprocess_folder(input_dir, output_dir, transform)
        else:
            print(
                f"Skipping {name}: '{input_dir}' not found locally. "
                f"If processing this dataset elsewhere (e.g. Kaggle), "
                f"copy the resulting .pt tensors into '{output_dir}' "
                f"before running dataset.py."
            )
