"""FastAPI application — VLM inference server."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from vlm_bench.models.base import VLMInput
from vlm_bench.models.factory import ModelFactory
from vlm_bench.server.schemas import (
    HealthResponse,
    InferRequest,
    InferResponse,
    ModelsResponse,
)

logger = logging.getLogger(__name__)

# Global model instance
_model = None
_model_name = None
_max_batch_size = 8


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, unload on shutdown."""
    global _model
    logger.info(f"Loading model: {_model_name}")
    _model = ModelFactory.create(_model_name)
    _model.load_model()
    logger.info("Model loaded successfully")
    yield
    logger.info("Unloading model")
    _model.unload()


def create_app(model_name: str, max_batch_size: int = 8) -> FastAPI:
    """Create FastAPI application with VLM model."""
    global _model_name, _max_batch_size
    _model_name = model_name
    _max_batch_size = max_batch_size

    app = FastAPI(
        title="vlm-bench API",
        description="Vision-Language Model inference server",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        import torch
        return HealthResponse(
            status="healthy",
            model=_model_name,
            model_loaded=_model is not None and _model.is_loaded,
            device=str(_model.device) if _model else "N/A",
            gpu_memory_used_mb=torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
        )

    @app.get("/models", response_model=ModelsResponse)
    async def list_models():
        """List available models."""
        return ModelsResponse(models=ModelFactory.list_models())

    @app.post("/infer", response_model=InferResponse)
    async def infer(request: InferRequest):
        """Run VLM inference."""
        if _model is None or not _model.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        start = time.perf_counter()

        vlm_input = VLMInput(
            image_path=request.image_url,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
        )

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _model.generate, vlm_input)

        return InferResponse(
            text=result.text,
            tokens_generated=result.tokens_generated,
            latency_ms=result.latency_ms,
            tokens_per_second=result.tokens_per_second,
            model=_model_name,
        )

    return app
