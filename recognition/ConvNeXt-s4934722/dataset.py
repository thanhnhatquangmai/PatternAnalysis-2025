'''
dataset.py
Author: Thanh Nhat Quang Mai - 49347227
Description:
    Data loading utilities for the ADNI Alzheimer's dataset.
    - Provides PyTorch DataLoader for training and testing.
    - Includes data transformations with basic augmentations for training.
    - Supports grayscale-to-3-channel conversion, resizing, normalization.
'''

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEFAULT_ROOT_DIR = os.path.abspath("./ADNI")
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 0

def get_data_loader(
        is_train: bool,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        root_dir: str = DEFAULT_ROOT_DIR
) -> DataLoader:
    """
    Returns a PyTorch DataLoader for the ADNI dataset.

    Args:
        is_train (bool): Whether to return the training or test loader.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker processes for data loading.
        root_dir (str): Root directory containing the ADNI dataset.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    # Determine directory based on train/test
    if is_train:
        dir = os.path.join(root_dir, "AD_NC", "train")
    else:
        dir = os.path.join(root_dir, "AD_NC", "test")

    # Build transform
    transform = build_transform(is_train)

    # Print transform steps
    print("Transform = ")
    for t in transform.transforms:
        print(t)
    print("---------------------------")

    # Create dataset and dataloader
    dataset = datasets.ImageFolder(root=dir, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
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
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Resize to model input size
            transforms.Grayscale(num_output_channels=3), # Repeat grayscale to 3 channels
            transforms.RandomHorizontalFlip(), # Simple augmentation
            transforms.RandomRotation(10), # Small random rotation
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[ADNI_DEFAULT_MEAN] * 3, # 3 channels
                std=[ADNI_DEFAULT_STD] * 3
            )
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