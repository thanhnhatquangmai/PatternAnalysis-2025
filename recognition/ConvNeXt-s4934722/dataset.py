"""
dataset.py
Author: Thanh Nhat Quang Mai - 49347227
Date: 14th October 2025
Description:
    Data loading utilities for the ADNI Alzheimer's dataset.
    - Provides PyTorch DataLoader for training and testing.
    - Splits train/validation by patient ID to prevent data leakage.
    - Includes data transformations with augmentations for training.
    - Supports grayscale-to-3-channel conversion, resizing, normalization.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

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
        validate_split (float): Fraction of patients to use for validation.

    Returns:
        DataLoader | tuple(DataLoader, DataLoader): 
            - If is_train=True, returns (train_loader, val_loader)
            - If is_train=False, returns test_loader
    """
    if is_train:
        data_dir = os.path.join(root_dir, "AD_NC", "train")
    else:
        data_dir = os.path.join(root_dir, "AD_NC", "test")

    base_dataset = datasets.ImageFolder(root=data_dir)

    if is_train:
        # Extract patient IDs from filenames (e.g., "218391_94.jpg" -> "218391")
        all_paths = [sample[0] for sample in base_dataset.samples]
        patient_ids = [os.path.basename(p).split("_")[0] for p in all_paths]

        # Map patient IDs to dataset indices
        id_to_indices = {}
        for idx, pid in enumerate(patient_ids):
            id_to_indices.setdefault(pid, []).append(idx)

        unique_patients = list(id_to_indices.keys())

        # Split by patient IDs (so no patient appears in both sets)
        train_patients, val_patients = train_test_split(
            unique_patients, test_size=validate_split, random_state=42, shuffle=True
        )

        # Gather indices for each split
        train_indices = [i for pid in train_patients for i in id_to_indices[pid]]
        val_indices = [i for pid in val_patients for i in id_to_indices[pid]]

        # Create independent datasets with different transforms
        train_dataset = Subset(
            type(base_dataset)(root=base_dataset.root, transform=build_transform(is_train=True)),
            train_indices
        )
        val_dataset = Subset(
            type(base_dataset)(root=base_dataset.root, transform=build_transform(is_train=False)),
            val_indices
        )

        # DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        print(f"Split {len(unique_patients)} patients -> "
            f"{len(train_patients)} train, {len(val_patients)} val")
        print(f"Train images: {len(train_indices)}, Val images: {len(val_indices)}")

        return train_loader, val_loader

    else:
        # Test DataLoader (no split)
        transform = build_transform(is_train=False)
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)

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
        return transforms.Compose([
            transforms.RandomResizedCrop(
                size=(IMAGE_SIZE, IMAGE_SIZE),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[ADNI_DEFAULT_MEAN] * 3,
                std=[ADNI_DEFAULT_STD] * 3
            ),
            transforms.RandomErasing(p=0.25)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[ADNI_DEFAULT_MEAN] * 3,
                std=[ADNI_DEFAULT_STD] * 3
            )
        ])
