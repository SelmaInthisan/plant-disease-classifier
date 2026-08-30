"""
Model training pipeline with transfer learning, validation tracking, and model checkpointing.
Trains MobileNetV2 / ResNet for binary plant disease classification (Healthy vs Diseased).
"""
import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model.network import create_model
from src.dataset import create_data_loaders

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_model(
    data_dir: str = "data",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 3e-4,
    backbone: str = "mobilenet_v2",
    output_model: str = "model/plant_classifier.pth",
    seed: int = 42
):
    set_seed(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using compute device: {device}")

    # Create loaders
    train_loader, val_loader, test_loader, class_to_idx = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=224
    )
    num_classes = len(class_to_idx)
    print(f"Loaded {num_classes} classes: {class_to_idx}")

    # Instantiate model
    model = create_model(num_classes=num_classes, backbone=backbone, pretrained=True)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # Fine-tune the pretrained MobileNetV2 end-to-end with a conservative learning rate.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epochs": []
    }

    best_val_acc = 0.0
    start_time = time.time()

    print("\n=======================================================")
    print("      Starting Plant Classifier Model Training")
    print("=======================================================")
    print(f"Backbone: {backbone} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "class_to_idx": class_to_idx,
                "backbone": backbone,
                "num_classes": num_classes
            }, output_model)
            print(f"  → Saved new best model (Val Acc: {val_acc*100:.2f}%) to {output_model}")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.2f} minutes. Best Val Acc: {best_val_acc*100:.2f}%")

    with open("reports/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history, best_val_acc

def main():
    parser = argparse.ArgumentParser(description="Train Plant Disease Classification Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--backbone", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "resnet18"], help="Backbone architecture")
    parser.add_argument("--output_model", type=str, default="model/plant_classifier.pth", help="Output path for best model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        output_model=args.output_model,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
