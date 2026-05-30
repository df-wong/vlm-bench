"""LLaVA v1.6 model implementation."""

from __future__ import annotations

import time
from typing import Optional

import torch
from PIL import Image

from vlm_bench.models.base import BaseVLM, VLMInput, VLMOutput
from vlm_bench.models.factory import register_model


@register_model("llava-v1.6")
@register_model("llava")
class LLaVAModel(BaseVLM):
    """LLaVA v1.6 (Large Language and Vision Assistant) implementation.

    Supports 7B and 13B variants with Mistral/Vicuna backbones.
    Optimized for ROCm with FlashAttention-ROCm integration.
    """

    SUPPORTED_VARIANTS = {"7b", "13b"}

    def __init__(self, model_name: str, device: str = "cuda:0", dtype: str = "float16", **kwargs):
        super().__init__(model_name, device, dtype)
        self.variant = kwargs.get("variant", "7b")
        self.attn_impl = kwargs.get("attn_implementation", None)
        self._image_processor = None

    def load_model(self, cache_dir: Optional[str] = None) -> None:
        """Load LLaVA model from HuggingFace Hub."""
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        model_id = f"llava-hf/llava-v1.6-mistral-7b-hf"
        if self.variant == "13b":
            model_id = f"llava-hf/llava-v1.6-vicuna-13b-hf"

        kwargs = {"torch_dtype": self.dtype, "device_map": self.device}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if self.attn_impl:
            kwargs["attn_implementation"] = self.attn_impl

        self._model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._loaded = True

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Process PIL image through LLaVA image processor."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        inputs = self._processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].to(self.device, dtype=self.dtype)

    def generate(self, vlm_input: VLMInput) -> VLMOutput:
        """Generate text from image + prompt using LLaVA."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image = vlm_input.image
        if image is None and vlm_input.image_path:
            image = Image.open(vlm_input.image_path).convert("RGB")

        if image is None:
            raise ValueError("No image provided")

        # Build conversation prompt
        prompt = f"[INST] <image>\n{vlm_input.prompt} [/INST]"

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        # Measure memory before
        mem_before = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

        # Generate
        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=vlm_input.max_new_tokens,
                temperature=vlm_input.temperature,
                top_p=vlm_input.top_p,
                do_sample=vlm_input.do_sample,
                repetition_penalty=vlm_input.repetition_penalty,
            )
        elapsed = (time.perf_counter() - start) * 1000

        # Decode
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        text = self._processor.decode(generated_ids, skip_special_tokens=True)
        tokens_gen = len(generated_ids)

        mem_after = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

        return VLMOutput(
            text=text,
            tokens_generated=tokens_gen,
            latency_ms=elapsed,
            tokens_per_second=tokens_gen / (elapsed / 1000) if elapsed > 0 else 0,
            memory_used_mb=mem_after - mem_before,
        )
