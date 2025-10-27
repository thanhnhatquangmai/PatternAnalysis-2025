"""
predict.py
Author: Thanh Nhat Quang Mai - 49347227
Date: 17th October 2025
Description:
    Loads a trained ConvNeXt model from modules.py and performs
    inference on new ADNI images using the same transforms as dataset.py.
    Generates GradCAM visualizations using torchcam for interpretability.
"""

import os
import torch
import torch.nn as nn
from PIL import Image
import argparse
import matplotlib.pyplot as plt

from dataset import build_transform
from modules import ConvNeXt    

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


# Device and constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["AD", "NC"]
MODEL_PATH = "./best_convnext.pth"
IMAGE_PATH = "./images/ADNI"

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


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


def predict_image(model, image_path, save_cam=False, cam_extractor=None):
    '''Run inference on a single image.'''
    model.eval()
    img = Image.open(image_path).convert("RGB")
    transform = build_transform(is_train=False)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    if save_cam:
        with torch.set_grad_enabled(True):
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)

            # Generate CAM
            activation_map = cam_extractor(pred.squeeze().item(), outputs)

            # Convert activation map tensor -> PIL image
            cam_img = to_pil_image(activation_map[0].squeeze().cpu(), mode='F')

            # Overlay heatmap on original image
            result = overlay_mask(img, cam_img, alpha=0.5)

            # Save result
            cam_path = os.path.splitext(image_path)[0] + "_cam.jpg"
            result.save(cam_path)
            print(f"Saved CAM: {cam_path}")

    else:
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)

    return CLASS_NAMES[pred.item()], conf.item()


def predict_folder(model, folder_path):
    """Run inference on all images in a folder."""
    print(f"\nRunning inference on folder: {folder_path}\n")
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            fpath = os.path.join(folder_path, fname)
            label, conf = predict_image(model, fpath, save_cam=False)  # skip CAM for batch
            color = GREEN if label == "AD" else RED
            print(f"{fname:<30} -> {color}{label}{RESET} ({conf:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Predict AD/NC using custom ConvNeXt model")
    parser.add_argument("--path", default=IMAGE_PATH, help="Path to image or folder of images")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to trained model (.pth)")
    parser.add_argument("--cam", action="store_true", help="Generate and save CAM for single image prediction")
    args = parser.parse_args()

    model = load_model(args.model)

    if os.path.isfile(args.path):
        # Initialize GradCAM only for single image
        cam_extractor = GradCAM(model, target_layer=model.stages[-1]) if args.cam else None
        label, conf = predict_image(model, args.path, save_cam=args.cam, cam_extractor=cam_extractor)
        color = GREEN if label == "AD" else RED
        print(f"\nPrediction: {color}{label}{RESET} (Confidence: {conf:.3f})")
    elif os.path.isdir(args.path):
        # Normal folder inference — no CAM
        predict_folder(model, args.path)
    else:
        print("Invalid path. Please provide a valid image file or folder.")


if __name__ == "__main__":
    main()
