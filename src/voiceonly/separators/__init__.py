"""Model registry: the only place that knows which candidates exist."""

from __future__ import annotations

import importlib.util

from .base import ModelInfo, SeparationError, SpeechSeparator
from .models import (
    AUDIOSEP_MODELS,
    AudioSeparatorModel,
    DeepFilterNet,
    HTDemucs,
    HTDemucsFT,
    Passthrough,
)

REGISTRY: dict[str, type[SpeechSeparator]] = {
    cls.info.key: cls
    for cls in (Passthrough, HTDemucs, HTDemucsFT, AudioSeparatorModel, *AUDIOSEP_MODELS,
                DeepFilterNet)
}

_IMPORT_NAMES = {"demucs": "demucs", "deepfilternet": "df", "audio-separator": "audio_separator"}


def missing_requirements(key: str) -> list[str]:
    return [r for r in REGISTRY[key].info.requires
            if importlib.util.find_spec(_IMPORT_NAMES.get(r, r)) is None]


def install_hint(key: str) -> str:
    if key == "deepfilternet":
        return "It pins numpy<2; use a separate Python 3.11 env (docs/decisions/0004)."
    extra = ("roformer" if "audio-separator" in REGISTRY[key].info.requires
             else {"htdemucs": "demucs", "htdemucs_ft": "demucs"}.get(key, key))
    return f"Install with: uv sync --extra {extra}"


def create(key: str, device: str = "auto", **options) -> SpeechSeparator:
    if key not in REGISTRY:
        raise SeparationError(f"unknown model '{key}'. Known: {', '.join(REGISTRY)}")
    missing = missing_requirements(key)
    if missing:
        raise SeparationError(f"model '{key}' needs {', '.join(missing)}. {install_hint(key)}")
    return REGISTRY[key](device=device, **options)


__all__ = ["REGISTRY", "ModelInfo", "SeparationError", "SpeechSeparator", "create",
           "missing_requirements"]
