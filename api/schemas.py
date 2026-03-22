"""Pydantic request/response models for FastAPI."""

from typing import Optional, Dict
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for single prediction."""

    journal_text: str = Field(..., description="User's reflection text")
    ambience_type: str = Field(..., description="Ambience type: forest, ocean, mountain, rain, cafe")
    duration_min: int = Field(..., ge=1, le=180, description="Session duration in minutes")
    sleep_hours: Optional[float] = Field(None, ge=0, le=24, description="Hours slept last night")
    energy_level: int = Field(..., ge=1, le=5, description="Energy level 1-5")
    stress_level: int = Field(..., ge=1, le=5, description="Stress level 1-5")
    time_of_day: str = Field(..., description="Time: morning, afternoon, evening, night")
    previous_day_mood: Optional[str] = Field(None, description="Previous mood")
    face_emotion_hint: Optional[str] = Field(None, description="Face emotion hint")
    reflection_quality: str = Field(..., description="Quality: clear, vague, conflicted")
    generate_message: bool = Field(False, description="Whether to generate supportive message")


class PredictionResponse(BaseModel):
    """Response model for single prediction."""

    predicted_state: str = Field(..., description="Predicted emotional state")
    predicted_intensity: int = Field(..., description="Predicted intensity 1-5")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    uncertain_flag: int = Field(..., description="1 if uncertain, 0 otherwise")
    what_to_do: str = Field(..., description="Recommended action")
    when_to_do: str = Field(..., description="When to do it")
    message: Optional[str] = Field(None, description="Supportive message if requested")
    state_probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    data: list[Dict] = Field(..., description="List of prediction dictionaries")
    generate_messages: bool = Field(False, description="Whether to generate messages")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    models_loaded: bool
