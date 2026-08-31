"""
Memory-efficient model loading and inference service.
"""

import os
import io
import time
import json
from typing import Dict, Any

from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from model.network import create_model
from app.disease_info import get_disease_info


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ModelInferenceService:

    _instance = None

    def __init__(
        self,
        model_path: str = "model/plant_classifier.pth",
        class_indices_path: str = "model/class_indices.json"
    ):

        # CPU is the correct choice for Render free hosting.
        self.device = torch.device("cpu")

        self.model_path = model_path
        self.class_indices_path = class_indices_path

        self.model = None
        self.idx_to_class = {}
        self.class_to_idx = {}
        self.num_classes = 0

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])

        # IMPORTANT:
        # Model is NOT loaded during initialization.
        # It will be loaded only when predict() is called.

    @classmethod
    def get_instance(cls):

        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    def load_model(self):

        if self.model is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}"
            )

        print("Loading MobileNetV2 checkpoint...", flush=True)

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=False
        )

        # --------------------------------------------------
        # Load class mapping
        # --------------------------------------------------

        checkpoint_classes = (
            checkpoint.get("class_to_idx")
            if isinstance(checkpoint, dict)
            else None
        )

        if checkpoint_classes is not None:

            self.class_to_idx = {
                str(k): int(v)
                for k, v in checkpoint_classes.items()
            }

        elif os.path.exists(self.class_indices_path):

            with open(
                self.class_indices_path,
                "r",
                encoding="utf-8"
            ) as f:

                file_classes = json.load(f)

            self.class_to_idx = {
                str(k): int(v)
                for k, v in file_classes.items()
            }

        else:

            self.class_to_idx = {
                "Diseased": 0,
                "Healthy": 1
            }

        self.idx_to_class = {
            int(v): k
            for k, v in self.class_to_idx.items()
        }

        self.num_classes = len(self.class_to_idx)

        # --------------------------------------------------
        # Create model
        # --------------------------------------------------

        backbone = (
            checkpoint.get("backbone", "mobilenet_v2")
            if isinstance(checkpoint, dict)
            else "mobilenet_v2"
        )

        self.model = create_model(
            num_classes=self.num_classes,
            backbone=backbone,
            pretrained=False
        )

        state_dict = (
            checkpoint.get("model_state_dict")
            if isinstance(checkpoint, dict)
            else checkpoint
        )

        self.model.load_state_dict(
            state_dict,
            strict=True
        )

        # Release checkpoint memory immediately.
        del checkpoint
        del state_dict

        self.model = self.model.to(self.device)
        self.model.eval()

        print(
            f"Model loaded on {self.device}",
            flush=True
        )

        print(
            f"Class mapping: {self.class_to_idx}",
            flush=True
        )

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:

        # Load model only when prediction is requested.
        self.load_model()

        start_time = time.perf_counter()

        # --------------------------------------------------
        # Load image
        # --------------------------------------------------

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        # --------------------------------------------------
        # Inference
        # --------------------------------------------------

        with torch.inference_mode():

            logits = self.model(tensor)

            probabilities = F.softmax(
                logits,
                dim=1
            ).squeeze(0)

        # --------------------------------------------------
        # Class probabilities
        # --------------------------------------------------

        class_probs = {}

        for idx, cls_name in self.idx_to_class.items():

            class_probs[cls_name] = float(
                round(
                    probabilities[idx].item(),
                    4
                )
            )

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        top_prob, top_idx = torch.max(
            probabilities,
            dim=0
        )

        predicted_class = self.idx_to_class.get(
            top_idx.item(),
            "Diseased"
        )

        confidence_val = float(
            round(
                top_prob.item(),
                4
            )
        )

        # --------------------------------------------------
        # Disease information
        # --------------------------------------------------

        info = get_disease_info(
            predicted_class
        )

        inference_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # Release temporary tensors.
        del tensor
        del logits
        del probabilities

        return {

            "prediction": predicted_class,

            "confidence": confidence_val,

            "class_probabilities": class_probs,

            "display_name":
                f"{predicted_class} Plant Leaf",

            "crop":
                info.get(
                    "crop",
                    "Foliage Crop"
                ),

            "condition":
                info.get(
                    "condition",
                    f"{predicted_class} Leaf"
                ),

            "is_healthy":
                predicted_class == "Healthy",

            "confidence_percentage":
                f"{confidence_val * 100:.1f}%",

            "severity":
                info.get(
                    "severity",
                    "None"
                    if predicted_class == "Healthy"
                    else "Moderate"
                ),

            "pathogen_type":
                info.get(
                    "pathogen_type",
                    "None"
                ),

            "symptoms":
                info.get(
                    "symptoms",
                    []
                ),

            "causes":
                info.get(
                    "causes",
                    ""
                ),

            "prevention":
                info.get(
                    "prevention",
                    []
                ),

            "organic_treatment":
                info.get(
                    "organic_treatment",
                    ""
                ),

            "chemical_treatment":
                info.get(
                    "chemical_treatment",
                    ""
                ),

            "inference_time_ms":
                round(
                    inference_time_ms,
                    2
                )
        }