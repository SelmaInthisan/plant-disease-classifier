import json
from pathlib import Path

def md(text):
    return {"cell_type":"markdown","metadata":{},"source":text.splitlines(True)}
def code(text):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(True)}

cells = [
md("""# Plant Disease Classification — MobileNetV2
**Dataset:** `mohanty/PlantVillage` (`color`)
**Task:** Binary image classification — Healthy vs Diseased
"""),
md("""## 1. Dataset and split
The official dataset provides a predefined 80/20 train/test split that preserves `leaf_id`. The project keeps that test split and creates validation only from the official training split using a leaf-safe stratified split.
"""),
code("""from datasets import load_dataset
ds = load_dataset('mohanty/PlantVillage', 'color')
print(ds)
"""),
md("""## 2. Binary labels
Every label ending in `___healthy` is mapped to **Healthy**. Every other PlantVillage disease label is mapped to **Diseased**.
"""),
code("""def binary_label(label_name):
    return 'Healthy' if str(label_name).endswith('___healthy') else 'Diseased'
"""),
md("""## 3. Preprocessing and augmentation
Training uses 224×224 inputs, random resized crop, horizontal/vertical flips, rotation, color jitter, and ImageNet normalization. Validation/test use deterministic resize and ImageNet normalization only.
"""),
code("""from torchvision import transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
"""),
md("""## 4. MobileNetV2 transfer learning
Use ImageNet-pretrained MobileNetV2 and replace the classifier with a two-output head for Healthy/Diseased.
"""),
code("""from torchvision import models
import torch.nn as nn
base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
in_features = base.classifier[1].in_features
base.classifier = nn.Sequential(
    nn.Dropout(0.3), nn.Linear(in_features, 256), nn.ReLU(True),
    nn.BatchNorm1d(256), nn.Dropout(0.2), nn.Linear(256, 2)
)
"""),
md("""## 5. Training
Run the project's reproducible training script:
`python src/train.py --epochs 15 --batch_size 32 --lr 0.0003`
"""),
md("""## 6. Evaluation
Run `python src/evaluate.py` to generate accuracy, precision, recall, F1-score and the confusion matrix from the official test split.
"""),
md("""## 7. Deployment
After the corrected model is trained and evaluated, deploy the FastAPI/Gradio application to a free Hugging Face Space. Do not upload the local dataset directory.
""")]
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.x"}},"nbformat":4,"nbformat_minor":5}
Path('notebooks/Plant_Disease_Classification.ipynb').write_text(json.dumps(nb, indent=2), encoding='utf-8')
