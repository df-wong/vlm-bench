"""Tests for model factory and base classes."""

import pytest

from vlm_bench.models.factory import ModelFactory


class TestModelFactory:
    def test_list_models(self):
        models = ModelFactory.list_models()
        assert len(models) > 0
        assert any("llava" in m for m in models)

    def test_create_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create("nonexistent-model")

    def test_create_llava(self):
        model = ModelFactory.create("llava-v1.6", device="cpu")
        assert model.model_name == "llava-v1.6"
        assert not model.is_loaded

    def test_create_internvl(self):
        model = ModelFactory.create("internvl-2", device="cpu")
        assert model.model_name == "internvl-2"
        assert not model.is_loaded

    def test_model_repr(self):
        model = ModelFactory.create("llava-v1.6", device="cpu")
        assert "llava" in repr(model).lower()
        assert "not loaded" in repr(model)
