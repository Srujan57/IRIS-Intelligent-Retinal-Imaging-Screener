# IRIS: Intelligent Retinal Imaging Screener

A two-stage deep learning pipeline for early eating disorder detection via retinal OCT biomarkers.

## How It Works
1. `preprocess.py` — normalizes and resizes raw OCT scans
2. `segment.py` — U-Net segments retinal layers (RNFL, GCL, macular, choroidal) and extracts thickness measurements
3. `train.py` — trains the classifier on extracted thickness features
4. `evaluate.py` — reports per-class AUC, macro F1, and confusion matrix

## Datasets
- **Kermany OCT** — kaggle.com/datasets/paultimothymooney/kermany2018
- **OCT5k** — nature.com/articles/s41597-024-04259-z

## Setup
```bash
pip install -r requirements.txt
```

## Pipeline
```
OCT scan → preprocess.py → segment.py → classifier → healthy / ED-pattern
```

## Paper
IRIS: A Deep Learning Pipeline for Early Eating Disorder Detection via Retinal OCT Biomarkers
