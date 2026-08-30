"""Backend inference helper alias."""
from app.model_loader import ModelInferenceService

def get_inference_service():
    return ModelInferenceService.get_instance()
