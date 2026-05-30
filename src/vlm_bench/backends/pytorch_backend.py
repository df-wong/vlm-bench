"""Standard PyTorch inference backend."""

from __future__ import annotations

from typing import Any

import torch

from vlm_bench.backends.base import BaseBackend, BackendInfo


class PyTorchBackend(BaseBackend):
    """Vanilla PyTorch inference with optional torch.compile."""

    def __init__(self, device: str = "cuda:0", compile_model: bool = False, compile_mode: str = "reduce-overhead"):
        super().__init__(device)
        self.compile_model = compile_model
        self.compile_mode = compile_mode

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="pytorch",
            version=torch.__version__,
            device=self.device,
            supports_cuda_graph=self.compile_model,
        )

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optionally compile model with torch.compile."""
        model = model.to(self.device)
        if self.compile_model:
            model = torch.compile(model, mode=self.compile_mode)
            self._compiled = True
        return model.eval()

    def run_inference(self, model: torch.nn.Module, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run standard PyTorch inference."""
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            output = model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return {"output": output}
