import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader():
    ...

def build_transform(is_train: bool):
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