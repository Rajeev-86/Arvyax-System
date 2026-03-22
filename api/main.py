"""FastAPI backend for ArvyaX emotional intelligence system."""

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

# Add parent package to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import EmotionalInferencePipeline
from src import __version__
from .schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="ArvyaX Emotional Intelligence API",
    description="Predict emotional state, intensity, and actionable recommendations",
    version=__version__
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
_pipeline: Optional[EmotionalInferencePipeline] = None


def get_pipeline() -> EmotionalInferencePipeline:
    """Get or initialize the inference pipeline."""
    global _pipeline
    if _pipeline is None:
        print("Initializing EmotionalInferencePipeline...")
        _pipeline = EmotionalInferencePipeline(use_slm=False, use_template_fallback=True)
    return _pipeline


@app.on_event("startup")
async def startup():
    """Initialize pipeline on startup."""
    try:
        get_pipeline()
        print("✅ API startup complete")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        pipeline = get_pipeline()
        return HealthResponse(
            status="healthy",
            version=__version__,
            models_loaded=True
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            version=__version__,
            models_loaded=False
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction.

    Args:
        request: PredictionRequest with user input

    Returns:
        PredictionResponse with predictions and recommendations
    """
    try:
        pipeline = get_pipeline()

        # Convert request to dict
        input_data = request.model_dump()

        # Get prediction
        result = pipeline.predict_single(
            data=input_data,
            generate_message=request.generate_message
        )

        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    file: UploadFile = File(...),
    generate_messages: bool = False
):
    """
    Make batch predictions from CSV file.

    Args:
        file: CSV file with records
        generate_messages: Whether to generate messages

    Returns:
        Batch predictions
    """
    try:
        pipeline = get_pipeline()

        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf8")))

        # Validate required columns
        required_cols = [
            'journal_text', 'ambience_type', 'duration_min',
            'energy_level', 'stress_level', 'time_of_day', 'reflection_quality'
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Get predictions
        results_df = pipeline.predict_batch(df, generate_messages=generate_messages)

        # Convert to list of dicts
        predictions = [
            PredictionResponse(**row)
            for _, row in results_df.iterrows()
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/predict/raw")
async def predict_raw(data: dict):
    """
    Raw dict-based prediction (for API clients not using Pydantic schema).

    Args:
        data: Dict with required fields

    Returns:
        Raw prediction dict
    """
    try:
        pipeline = get_pipeline()
        result = pipeline.predict_single(data=data, generate_message=data.get('generate_message', False))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "ArvyaX Emotional Intelligence API",
        "version": __version__,
        "description": "Predict emotional state and get personalized recommendations",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "predict_raw": "POST /predict/raw",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
