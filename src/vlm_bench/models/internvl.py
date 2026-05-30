"""InternVL-2 model implementation."""

from __future__ import annotations

import time
from typing import Optional

import torch
from PIL import Image

from vlm_bench.models.base import BaseVLM, VLMInput, VLMOutput
from vlm_bench.models.factory import register_model


@register_model("internvl-2")
@register_model("internvl")
class InternVLModel(BaseVLM):
    """InternVL-2 — Open-source multimodal model for understanding.

    Supports OCR, document understanding, chart analysis, and general VQA.
    Variants: 8B (InternVL2-8B), 26B (InternVL2-26B).
    """

    SUPPORTED_VARIANTS = {"8b", "26b"}

    def __init__(self, model_name: str, device: str = "cuda:0", dtype: str = "float16", **kwargs):
        super().__init__(model_name, device, dtype)
        self.variant = kwargs.get("variant", "8b")

    def load_model(self, cache_dir: Optional[str] = None) -> None:
        """Load InternVL model."""
        from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

        model_id = f"OpenGVLab/InternVL2-8B"
        if self.variant == "26b":
            model_id = "OpenGVLab/InternVL2-26B"

        kwargs = {"torch_dtype": self.dtype, "trust_remote_code": True}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        self._model = AutoModel.from_pretrained(model_id, **kwargs).to(self.device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self._image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._loaded = True

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Process image through InternVL vision encoder."""
        if not self._loaded:
            raise RuntimeError("Model not loaded.")
        pixel_values = self._image_processor(images=image, return_tensors="pt").pixel_values
        return pixel_values.to(self.device, dtype=self.dtype)

    def generate(self, vlm_input: VLMInput) -> VLMOutput:
        """Generate text using InternVL."""
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        image = vlm_input.image
        if image is None and vlm_input.image_path:
            image = Image.open(vlm_input.image_path).convert("RGB")

        if image is None:
            raise ValueError("No image provided")

        # InternVL uses a different prompt format
        question = f"<image>\n{vlm_input.prompt}"
        pixel_values = self.encode_image(image)

        generation_config = dict(
            max_new_tokens=vlm_input.max_new_tokens,
            do_sample=vlm_input.do_sample,
            temperature=vlm_input.temperature,
            top_p=vlm_input.top_p,
        )

        start = time.perf_counter()
        with torch.no_grad():
            response = self._model.chat(
                self._tokenizer,
                pixel_values,
                question,
                generation_config,
            )
        elapsed = (time.perf_counter() - start) * 1000

        # Estimate tokens
        tokens = len(self._tokenizer.encode(response))

        return VLMOutput(
            text=response,
            tokens_generated=tokens,
            latency_ms=elapsed,
            tokens_per_second=tokens / (elapsed / 1000) if elapsed > 0 else 0,
        )
