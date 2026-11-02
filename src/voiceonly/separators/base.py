"""The model interface every candidate implements.

This mirrors the `SpeechSeparator` trait planned for the Rust core, so the
capabilities recorded here carry over unchanged.
"""

from __future__ import annotations

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

    def to_dict(self) -> dict:
        return asdict(self)


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
