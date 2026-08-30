"""
Comprehensive model evaluation script.
Calculates Accuracy, Precision, Recall, F1-score, Confusion Matrix, and generates publication plots.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model.network import create_model
from src.dataset import create_data_loaders

def plot_learning_curves(
    history_file: str = "reports/training_history.json",
    output_path: str = "reports/learning_curves.png",
    loss_path: str = "reports/training_loss.png",
    acc_path: str = "reports/accuracy.png"
):
    if not os.path.exists(history_file):
        print(f"Warning: {history_file} not found. Skipping learning curves plot.")
        return

    with open(history_file, "r") as f:
        history = json.load(f)

    epochs = history["epochs"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = [a * 100 for a in history["train_acc"]]
    val_acc = [a * 100 for a in history["val_acc"]]

    # 1. Combined 2-panel plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    axes[0].plot(epochs, train_loss, 'o-', color='#2563eb', label='Train Loss', linewidth=2.5, markersize=5)
    axes[0].plot(epochs, val_loss, 's-', color='#dc2626', label='Val Loss', linewidth=2.5, markersize=5)
    axes[0].set_title("Training vs Validation Loss", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=11, fontweight='bold')
    axes[0].legend(frameon=True, fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Accuracy subplot
    axes[1].plot(epochs, train_acc, 'o-', color='#16a34a', label='Train Accuracy', linewidth=2.5, markersize=5)
    axes[1].plot(epochs, val_acc, 's-', color='#9333ea', label='Val Accuracy', linewidth=2.5, markersize=5)
    axes[1].set_title("Training vs Validation Accuracy", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Accuracy (%)", fontsize=11, fontweight='bold')
    axes[1].legend(frameon=True, fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved combined learning curves plot to {output_path}")

    # 2. Individual Loss plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss, 'o-', color='#2563eb', label='Train Loss', linewidth=2.5)
    plt.plot(epochs, val_loss, 's-', color='#dc2626', label='Val Loss', linewidth=2.5)
    plt.title("Cross Entropy Loss vs Epoch", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # 3. Individual Accuracy plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_acc, 'o-', color='#16a34a', label='Train Accuracy', linewidth=2.5)
    plt.plot(epochs, val_acc, 's-', color='#9333ea', label='Val Accuracy', linewidth=2.5)
    plt.title("Model Accuracy vs Epoch", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()

def evaluate_model(
    model_path: str = "model/plant_classifier.pth",
    data_dir: str = "data",
    output_dir: str = "reports"
):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Evaluating model on device: {device}")

    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {"Healthy": 0, "Diseased": 1})
    backbone = checkpoint.get("backbone", "mobilenet_v2")
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # Instantiate and load model
    model = create_model(num_classes=num_classes, backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test data
    _, _, test_loader, _ = create_data_loaders(data_dir=data_dir, batch_size=32)

    all_preds = []
    all_targets = []
    all_probs = []

    print("Running inference on holdout test partition...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Metrics computation
    overall_acc = accuracy_score(all_targets, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)

    class_metrics = {}
    for i, c_name in enumerate(class_names):
        class_metrics[c_name] = {
            "precision": float(round(precision_per_class[i], 4)),
            "recall": float(round(recall_per_class[i], 4)),
            "f1_score": float(round(f1_per_class[i], 4)),
            "support": int(support_per_class[i])
        }

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Plot Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Confusion Matrix — Plant Disease Classification\n(Healthy vs Diseased)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Predicted Label", fontsize=11, fontweight='bold')
    plt.ylabel("Actual True Label", fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

    # Plot learning curves
    plot_learning_curves()

    # Save summary JSON files
    summary = {
        "overall_accuracy": float(round(overall_acc, 4)),
        "macro_precision": float(round(precision_macro, 4)),
        "macro_recall": float(round(recall_macro, 4)),
        "macro_f1": float(round(f1_macro, 4)),
        "weighted_precision": float(round(precision_weighted, 4)),
        "weighted_recall": float(round(recall_weighted, 4)),
        "weighted_f1": float(round(f1_weighted, 4)),
        "classes": class_names,
        "per_class_metrics": class_metrics,
        "confusion_matrix": cm.tolist(),
        "total_test_samples": int(len(all_targets)),
        "model_architecture": backbone
    }

    summary_file = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("           MODEL TEST EVALUATION SUMMARY")
    print("="*60)
    print(f"Target Classification : Binary (Healthy vs Diseased)")
    print(f"Holdout Test Samples  : {len(all_targets)}")
    print(f"Overall Accuracy      : {overall_acc*100:.2f}%")
    print(f"Macro Precision       : {precision_macro*100:.2f}%")
    print(f"Macro Recall          : {recall_macro*100:.2f}%")
    print(f"Macro F1-Score        : {f1_macro*100:.2f}%")
    print(f"Weighted F1-Score     : {f1_weighted*100:.2f}%")
    print("="*60)
    print("\nClassification Report:\n")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

    return summary

if __name__ == "__main__":
    evaluate_model()
