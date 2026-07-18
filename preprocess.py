"""
preprocess.py

Loads raw OCT scans, resizes to config image_size,
normalizes pixel values, and saves processed images
ready for segmentation.
"""
import glob
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
    """
    Recursively walks input_dir so nested dataset layouts (e.g. Kermany2018's
    kermany2018/OCT2017/train/CNV/*.jpeg structure) are found, not just files
    sitting directly in input_dir. The subfolder path is folded into the
    output filename so images with the same name in different subfolders
    (e.g. two datasets both having an "img1.jpeg") don't overwrite each other.
    """
    os.makedirs(output_dir, exist_ok=True)

    processed = 0
    skipped = 0
    for root, dirs, filenames in os.walk(input_dir):
        # Skip macOS zip artifacts (__MACOSX/) entirely — they only ever
        # contain AppleDouble resource-fork files, never real images.
        dirs[:] = [d for d in dirs if d != "__MACOSX"]

        for filename in filenames:
            # AppleDouble files ("._foo.jpeg") are metadata sidecars for
            # "foo.jpeg", not images — PIL can't open them, skip them.
            if filename.startswith("._"):
                continue

            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                input_path = os.path.join(root, filename)

                rel_dir = os.path.relpath(root, input_dir)
                prefix = "" if rel_dir == "." else rel_dir.replace(os.sep, "_") + "_"
                out_name = (
                    prefix + filename
                    .rsplit(".", 1)[0] + ".pt"
                )
                output_path = os.path.join(output_dir, out_name)

                try:
                    tensor = preprocess_image(input_path, transform)
                except Exception as e:
                    # Don't let one corrupt/unreadable file kill an
                    # 84k-image run — log it and keep going.
                    print(f"  Skipping unreadable file {input_path}: {e}")
                    skipped += 1
                    continue

                torch.save(tensor, output_path)
                processed += 1

    print(f"Done: {processed} images processed, {skipped} skipped → {output_dir}")


def find_shallowest_with_subdirs(root, required_subdirs):
    """
    Some dataset zips (Kermany2018 included) contain the real image tree
    duplicated inside themselves — e.g. both "kermany2018/OCT2017/" and a
    nested re-zip artifact at "kermany2018/oct2017/OCT2017/" with the same
    train/test/val/CNV/DME/DRUSEN/NORMAL contents. Walking the whole dataset
    root would process every image twice. Instead, find every folder that
    directly contains all of required_subdirs (case-insensitive, since the
    duplicate differs only in casing) and return the shallowest match, which
    is the real top-level copy — deeper matches are re-zip duplicates.
    """
    candidates = []
    for dirpath, dirs, _files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        present = {d.strip().lower() for d in dirs}
        if required_subdirs <= present:
            candidates.append(dirpath)

    if not candidates:
        return root

    candidates.sort(key=lambda p: p.count(os.sep))
    return candidates[0]


def resolve_dataset_dir(dataset_name, kaggle_root_hint, local_path, dedupe_subdirs=None):
    """
    Resolve each dataset's input directory independently, rather than
    switching both datasets on a single global Kaggle/local flag.
    This lets datasets live in different places at once — e.g. a large
    dataset processed on Kaggle and a smaller one processed locally.

    Kaggle's mount depth for a given dataset isn't consistent — it can be
    "/kaggle/input/<name>/" or nested several levels deeper (e.g.
    "/kaggle/input/datasets/<owner>/<name>/") depending on how the dataset
    was attached. Rather than hardcode one depth, search for a directory
    named dataset_name anywhere under /kaggle/input.

    dedupe_subdirs: optional set of subfolder names (e.g. {"train","test","val"})
    used to pick the real top-level image folder when a dataset's archive
    contains a duplicated nested copy of itself (see
    find_shallowest_with_subdirs above).
    """
    if os.path.isdir(kaggle_root_hint):
        matches = [
            m for m in glob.glob(f"/kaggle/input/**/{dataset_name}", recursive=True)
            if os.path.isdir(m)
        ]
        if matches:
            root = matches[0]
            if dedupe_subdirs:
                return find_shallowest_with_subdirs(root, dedupe_subdirs)
            return root
    return local_path


KERMANY_DIR = resolve_dataset_dir(
    "kermany2018", "/kaggle/input", "data/kermany/",
    dedupe_subdirs={"train", "test", "val"}
)
OCT5K_DIR = resolve_dataset_dir("oct5k-iris", "/kaggle/input", "data/oct5k/")

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
