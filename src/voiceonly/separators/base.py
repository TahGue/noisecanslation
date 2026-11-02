"""The model interface every candidate implements.

This mirrors the `SpeechSeparator` trait planned for the Rust core, so the
capabilities recorded here carry over unchanged.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class ModelInfo:
    key: str
    name: str
    version: str
    sample_rate: int
    channels: int
    # Pipeline chunking. Models with their own internal chunking still get long
    # chunks so memory stays bounded on long files.
    chunk_seconds: float
    overlap_seconds: float
    # Future context the model needs; 0 for offline-only models where it is
    # not meaningful. Recorded for live-mode planning.
    lookahead_seconds: float | None = None
    streaming: bool = False
    # Sliding window the live engine feeds per step; None = not usable live.
    live_window_seconds: float | None = None
    license: str = "unknown"
    requires: list[str] = field(default_factory=list)
    # Stems separate_stems() returns, in order; "voice" first. The default split is
    # the voice estimate and the rest of the mix.
    stems: list[str] = field(default_factory=lambda: ["voice", "background"])
    # Smallest live step this model can sustain on an RTX 4090.
    live_min_hop_seconds: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)


# One model call on the GPU at a time. Live engines can briefly overlap (stopping one
# while the next starts) and a CUDA-graph capture is invalidated by GPU work from
# another thread, so every live model call and model load holds this lock.
GPU_LOCK = threading.RLock()


class SeparationError(RuntimeError):
    pass


class SpeechSeparator(ABC):
    info: ModelInfo

    def __init__(self, device: str = "auto") -> None:
        self.device = device

    def initialize(self) -> None:
        """Load weights. Called once before the first chunk."""

    @abstractmethod
    def separate(self, chunk: np.ndarray) -> np.ndarray:
        """Return the speech estimate for one chunk.

        Input and output are float32 (frames, info.channels) at info.sample_rate,
        with the same number of frames.
        """

    def separate_stems(self, chunk: np.ndarray) -> np.ndarray:
        """All stems for one chunk, shaped (len(info.stems), frames, channels)."""
        voice = np.asarray(self.separate(chunk), dtype=np.float32)
        return np.stack([voice, chunk - voice])

    def close(self) -> None:
        """Release GPU memory."""

    def runtime_info(self) -> dict:
        return {}


def resolve_torch_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
