import os
import sys
import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.model_loader import ModelInferenceService
from app.disease_info import DISEASE_DATABASE, get_disease_info

client = TestClient(app)

def create_test_image_bytes(color=(34, 139, 34)) -> bytes:
    img = Image.new("RGB", (224, 224), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "device" in data
    assert data["classes_count"] == 2
    assert "Healthy" in data["classes"]
    assert "Diseased" in data["classes"]

def test_classes_endpoint():
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert len(data["classes"]) == 2
    for cls in data["classes"]:
        assert "class_name" in cls
        assert "display_name" in cls
        assert "is_healthy" in cls
        assert "symptoms" in cls
        assert "organic_treatment" in cls

def test_samples_endpoint():
    response = client.get("/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) > 0

def test_predict_endpoint_valid_image():
    img_bytes = create_test_image_bytes(color=(40, 160, 40))
    response = client.post(
        "/predict",
        files={"file": ("test_leaf.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Healthy", "Diseased"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "class_probabilities" in data
    assert "Healthy" in data["class_probabilities"]
    assert "Diseased" in data["class_probabilities"]
    assert "is_healthy" in data
    assert "symptoms" in data
    assert "prevention" in data
    assert "organic_treatment" in data
    assert "chemical_treatment" in data
    assert data["inference_time_ms"] > 0

def test_predict_endpoint_invalid_file_type():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"This is not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

def test_predict_endpoint_empty_file():
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_disease_info_database():
    for key, info in DISEASE_DATABASE.items():
        assert "prediction" in info
        assert "crop" in info
        assert "condition" in info
        assert "is_healthy" in info
        assert "symptoms" in info
        assert "prevention" in info
        assert "organic_treatment" in info
        assert "chemical_treatment" in info

def test_inference_service_singleton():
    service1 = ModelInferenceService.get_instance()
    service2 = ModelInferenceService.get_instance()
    assert service1 is service2
