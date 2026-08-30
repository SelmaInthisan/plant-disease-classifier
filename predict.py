"""
Command-line inference script for Plant Disease Classification.
Usage:
    python predict.py <path_to_leaf_image> [--model model/plant_classifier.pth]
"""
import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from model.network import create_model

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_inference_model(model_path: str = "model/plant_classifier.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please train the model first.")

    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {"Healthy": 0, "Diseased": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    backbone = checkpoint.get("backbone", "mobilenet_v2")
    num_classes = len(class_to_idx)

    model = create_model(num_classes=num_classes, backbone=backbone, pretrained=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return model, transform, idx_to_class, device

def predict_single_image(image_path: str, model_path: str = "model/plant_classifier.pth"):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)

    model, transform, idx_to_class, device = load_inference_model(model_path)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error reading image: {e}")
        sys.exit(1)

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    pred_class = idx_to_class[predicted_idx.item()]
    conf_pct = confidence.item() * 100

    print("=" * 45)
    print("      PLANT LEAF DIAGNOSTIC RESULT")
    print("=" * 45)
    print(f"Image File  : {image_path}")
    print(f"Prediction  : {pred_class}")
    print(f"Confidence  : {conf_pct:.2f}%")
    print("-" * 45)
    print("Class Probabilities:")
    for idx, cls_name in idx_to_class.items():
        prob = probabilities[idx].item() * 100
        print(f"  • {cls_name:<10}: {prob:6.2f}%")
    print("=" * 45)

    return {
        "prediction": pred_class,
        "confidence": round(confidence.item(), 4),
        "class_probabilities": {
            cls_name: round(probabilities[idx].item(), 4)
            for idx, cls_name in idx_to_class.items()
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Predict Plant Leaf Disease (Healthy vs Diseased)")
    parser.add_argument("image", type=str, help="Path to input plant leaf image")
    parser.add_argument("--model", type=str, default="model/plant_classifier.pth", help="Path to trained model checkpoint")
    args = parser.parse_args()

    predict_single_image(args.image, args.model)

if __name__ == "__main__":
    main()
