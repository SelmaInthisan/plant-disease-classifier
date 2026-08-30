"""
FastAPI Application for Plant Disease Classification System.
Provides REST API endpoints and serves the modern web interface.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.schemas import PredictionResponse, HealthResponse
from app.model_loader import ModelInferenceService
from app.disease_info import DISEASE_DATABASE, get_disease_info

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize and warm up inference service on startup
    ModelInferenceService.get_instance()
    yield

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Classification API 🌿",
    description="Deep-learning-powered plant leaf health diagnosis and binary disease classification service.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for cross-origin web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for frontend and sample images
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", summary="Root web interface")
async def serve_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Plant Disease Classification API is running. Visit /docs for OpenAPI documentation."}

@app.get("/health", response_model=HealthResponse, summary="API Health Check")
async def health_check():
    service = ModelInferenceService.get_instance()
    return HealthResponse(
        status="healthy",
        model_loaded=service.model is not None,
        device=str(service.device),
        version="1.0.0",
        classes_count=service.num_classes,
        classes=list(service.class_to_idx.keys())
    )

@app.get("/classes", summary="List supported target classes and diagnosis information")
async def list_classes():
    service = ModelInferenceService.get_instance()
    results = []
    for cls_name in service.idx_to_class.values():
        info = get_disease_info(cls_name)
        results.append({
            "class_name": cls_name,
            **info
        })
    return {"total_classes": len(results), "classes": results}

@app.get("/samples", summary="List sample leaf images for instant testing")
async def list_samples():
    samples_json = static_dir / "samples" / "samples.json"
    if samples_json.exists():
        with open(samples_json, "r") as f:
            samples = json.load(f)
            return {"samples": samples}
    return {"samples": []}

@app.post("/predict", response_model=PredictionResponse, summary="Classify a plant leaf image into Healthy or Diseased")
async def predict_image(
    file: UploadFile = File(..., description="Plant leaf image (JPG, PNG, WEBP)")
):
    # Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload a valid image file (JPG, PNG, WEBP)."
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
        
        if len(contents) > 15 * 1024 * 1024:  # 15MB limit
            raise HTTPException(status_code=400, detail="Uploaded image exceeds 15MB size limit.")

        service = ModelInferenceService.get_instance()
        prediction = service.predict(contents)
        return prediction

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/download", summary="Download complete project ZIP archive")
async def download_project_zip():
    zip_path = static_dir / "download" / "plant-disease-classifier.zip"
    if zip_path.exists():
        return FileResponse(
            path=str(zip_path),
            filename="plant-disease-classifier.zip",
            media_type="application/zip"
        )
    raise HTTPException(status_code=404, detail="Project ZIP archive not found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
