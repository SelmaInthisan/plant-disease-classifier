"""PyTorch data loaders for the binary PlantVillage task."""
import json
import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXPECTED_CLASS_TO_IDX = {"Diseased": 0, "Healthy": 1}


def get_transforms(img_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def create_data_loaders(data_dir="data", batch_size=32, img_size=224, num_workers=0):
    train_transform, eval_transform = get_transforms(img_size)
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_transform)

    for name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        if dataset.class_to_idx != EXPECTED_CLASS_TO_IDX:
            raise RuntimeError(
                f"{name} class mapping is {dataset.class_to_idx}; expected {EXPECTED_CLASS_TO_IDX}. "
                "Re-run src/data_download.py."
            )

    loaders = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
    return (*loaders, EXPECTED_CLASS_TO_IDX)
