"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    """Inference request."""
    prompt: str = Field(..., description="Text prompt for the VLM")
    image_url: Optional[str] = Field(None, description="URL or path to image")
    max_new_tokens: int = Field(512, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")
    do_sample: bool = Field(True, description="Whether to sample")


class InferResponse(BaseModel):
    """Inference response."""
    text: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float
    model: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    model_loaded: bool
    device: str
    gpu_memory_used_mb: float


class ModelsResponse(BaseModel):
    """Available models response."""
    models: list[str]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
