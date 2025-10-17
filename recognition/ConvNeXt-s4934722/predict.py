"""
predict.py
Author: Thanh Nhat Quang Mai - 49347227
Date: 17th October 2025
Description:
    Loads a trained ConvNeXt model from modules.py and performs
    inference on new ADNI images using the same transforms as dataset.py.
"""

import os
import torch
import torch.nn as nn
from PIL import Image
import argparse

from dataset import build_transform
from modules import ConvNeXt    

# Device and constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["AD", "NC"]
MODEL_PATH = "./best_convnext.pth"


def load_model(model_path=MODEL_PATH):
    """Load the trained custom ConvNeXt model."""
    print(f"Loading model from: {model_path}")
    model = ConvNeXt(num_classes=2)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
    return model
