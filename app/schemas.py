from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class: Healthy or Diseased")
    confidence: float = Field(..., description="Prediction confidence score between 0 and 1")
    class_probabilities: Dict[str, float] = Field(..., description="Probability breakdown across classes")
    display_name: str = Field(..., description="Human-readable title")
    crop: str = Field(..., description="Crop group")
    condition: str = Field(..., description="Diagnostic condition")
    is_healthy: bool = Field(..., description="Boolean health status")
    confidence_percentage: str = Field(..., description="Confidence formatted as percentage string")
    severity: str = Field(..., description="Condition severity level")
    pathogen_type: str = Field(..., description="Pathogen category or None")
    symptoms: List[str] = Field(..., description="Key observed symptoms")
    causes: str = Field(..., description="Environmental or biological causes")
    prevention: List[str] = Field(..., description="Recommended preventative cultural practices")
    organic_treatment: str = Field(..., description="Organic remedy guidelines")
    chemical_treatment: str = Field(..., description="Chemical control guidelines")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str
    classes_count: int
    classes: List[str]
