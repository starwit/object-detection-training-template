"""The project's params.yaml is valid for the installed trainer."""

from __future__ import annotations

from pathlib import Path

from object_detector_trainer import load_config

PARAMS_YAML = Path(__file__).parents[1] / "params.yaml"


def test_params_validate_including_every_model() -> None:
    config = load_config(PARAMS_YAML)

    assert config.train.model in config.models
    for model_key in config.models:
        config.model_options(model_key)
