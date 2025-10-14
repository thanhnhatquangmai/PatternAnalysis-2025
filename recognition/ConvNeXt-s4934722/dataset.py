"""
dataset.py
Author: Thanh Nhat Quang Mai - 49347227
Date: 14th October 2025
Description:
    Data loading utilities for the ADNI Alzheimer's dataset.
    - Provides PyTorch DataLoader for training and testing.
    - Includes data transformations with basic augmentations for training.
    - Supports grayscale-to-3-channel conversion, resizing, normalization.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DEFAULT_ROOT_DIR = os.path.abspath("./ADNI")
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 2

def get_dataloader(
    is_train: bool,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    root_dir: str = DEFAULT_ROOT_DIR,
    validate_split: float = 0.1
):
    """
    Returns PyTorch DataLoaders for ADNI dataset (train, val, or test).

    Args:
        is_train (bool): Whether to load training/validation or test data.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker processes for data loading.
        root_dir (str): Root directory containing the ADNI dataset.
        validate_split (float): Fraction of training data to use for validation.

    Returns:
        DataLoader | tuple(DataLoader, DataLoader): 
            - If is_train=True, returns (train_loader, val_loader)
            - If is_train=False, returns test_loader
    """
    if is_train:
        dir = os.path.join(root_dir, "AD_NC", "train")
    else:
        dir = os.path.join(root_dir, "AD_NC", "test")

    # Load full dataset first (no transform yet)
    base_dataset = datasets.ImageFolder(root=dir)

    if is_train:
        validate_size = int(len(base_dataset) * validate_split)
        train_size = len(base_dataset) - validate_size
        indices = torch.randperm(len(base_dataset)).tolist()
        train_indices = indices[:train_size]
        validate_indices = indices[train_size:]

        # Create separate datasets with independent transforms
        train_dataset = Subset(
            type(base_dataset)(root=base_dataset.root, transform=build_transform(is_train=True)),
            train_indices
        )
        validate_dataset = Subset(
            type(base_dataset)(root=base_dataset.root, transform=build_transform(is_train=False)),
            validate_indices
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        validate_dataloader = DataLoader(
            dataset=validate_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_dataloader, validate_dataloader

    else:
        # Test loader
        transform = build_transform(is_train=False)
        dataset = datasets.ImageFolder(root=dir, transform=transform)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        return dataloader

def build_transform(is_train: bool) -> transforms.Compose:
    """
    Build a torchvision transform pipeline for the ADNI dataset.

    Args:
        is_train (bool): Whether to build a transform for training or testing.

    Returns:
        transforms.Compose: Transformation pipeline.
    """
    ADNI_DEFAULT_MEAN = 0.116
    ADNI_DEFAULT_STD = 0.225
    IMAGE_SIZE = 224

    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=(IMAGE_SIZE, IMAGE_SIZE),
                scale=(0.8, 1.0), # Randomly zoom between 80-100%
                ratio=(0.9, 1.1) # Slightly vary aspect ratio
            ), # Resize to model input size
            transforms.Grayscale(num_output_channels=3), # Repeat grayscale to 3 channels
            transforms.RandomHorizontalFlip(), # Simple augmentation
            transforms.RandomRotation(10), # Small random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05)
            ), # Small translation to shift position
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[ADNI_DEFAULT_MEAN] * 3, # 3 channels
                std=[ADNI_DEFAULT_STD] * 3
            ),
            transforms.RandomErasing(p=0.25)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[ADNI_DEFAULT_MEAN] * 3, # 3 channels
                std=[ADNI_DEFAULT_STD] * 3
            )
        ])

    return transform