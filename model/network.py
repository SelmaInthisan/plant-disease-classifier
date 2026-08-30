import torch
import torch.nn as nn
from torchvision import models

class PlantClassifier(nn.Module):
    """
    Transfer learning classifier for Plant Disease detection (Healthy vs Diseased)
    using MobileNetV2 or ResNet.
    """
    def __init__(self, num_classes: int = 2, backbone: str = "mobilenet_v2", pretrained: bool = True, freeze_backbone: bool = False):
        super(PlantClassifier, self).__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.feature_extractor = models.mobilenet_v2(weights=weights)
            in_features = self.feature_extractor.classifier[1].in_features
            
            if freeze_backbone:
                for param in self.feature_extractor.features.parameters():
                    param.requires_grad = False
                    
            self.feature_extractor.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes)
            )
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.feature_extractor = models.resnet18(weights=weights)
            in_features = self.feature_extractor.fc.in_features
            
            if freeze_backbone:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                    
            self.feature_extractor.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

def create_model(num_classes: int = 2, backbone: str = "mobilenet_v2", pretrained: bool = True, freeze_backbone: bool = False) -> PlantClassifier:
    return PlantClassifier(num_classes=num_classes, backbone=backbone, pretrained=pretrained, freeze_backbone=freeze_backbone)
