"""
segment.py

Runs the U-Net model on preprocessed OCT scans to segment
retinal layers (RNFL, GCL, macular, choroidal) and extract
thickness measurements per layer per scan.
Outputs a feature vector for each scan passed to the classifier.
"""
