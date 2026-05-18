"""
quantize.py

Applies Quantization-Aware Training (QAT) to the classifier.
Runs bit-width sweep (FP32 -> INT8 -> INT4) and logs
accuracy at each precision level.
"""
