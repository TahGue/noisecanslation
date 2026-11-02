"""Low-overhead MRX inference for live use.

MRX computes three STFT resolutions, runs encoders, three bidirectional LSTM
stacks and nine mask decoders, then nine inverse STFTs. The network core (encoders
-> LSTMs -> masks) is captured as a CUDA graph (optionally fp16); the STFTs stay
eager and the inverse STFT uses the sync-free version from fast_demucs.
"""

from __future__ import annotations

import numpy as np
import torch

from .fast_demucs import _hann, istft


class MRXGraphRunner:
    """Separates a fixed number of frames into the model's sources (music, speech, sfx)."""

    def __init__(self, model, frames: int, channels: int, half: bool) -> None:
        self.model, self.frames, self.half = model, frames, half
        self.device = next(model.parameters()).device
        self.static_specs = [s.clone() for s in self._stfts(
            torch.zeros(channels, frames, device=self.device))]
        self.graph = None
        with torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._core()
            torch.cuda.current_stream().wait_stream(stream)
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    self.static_masks = self._core()
                self.graph = graph
            except Exception:
                torch.cuda.synchronize()
                self.graph = None  # eager fallback: same result, more CPU time

    def _stfts(self, audio: torch.Tensor) -> list[torch.Tensor]:
        hop = self.model._n_hop
        return [torch.stft(audio, n, hop, window=_hann(n, audio), win_length=n, normalized=True,
                           center=True, return_complex=True, pad_mode="reflect")
                for n in self.model._window_lengths]

    def _core(self) -> list[list[torch.Tensor]]:
        """Masks per source and resolution, each shaped like its spectrogram."""
        m = self.model
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            encoded = [enc(spec) for spec, enc in zip(self.static_specs, m._encoders)]
            cross = m._cross_net(encoded)
            return [[dec(cross).transpose(-1, -2).reshape(spec.shape).float()
                     for spec, dec in zip(self.static_specs, decoders)]
                    for decoders in m._decoders]

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """(frames, channels) mix -> (n_sources, frames, channels) in model order."""
        with torch.inference_mode():
            audio = torch.from_numpy(np.ascontiguousarray(chunk.T)).to(self.device)
            for static, spec in zip(self.static_specs, self._stfts(audio)):
                static.copy_(spec)
            if self.graph is not None:
                self.graph.replay()
                masks = self.static_masks
            else:
                masks = self._core()
            hop = self.model._n_hop
            sources = []
            for src_masks in masks:
                total = 0
                for spec, mask in zip(self.static_specs, src_masks):
                    n_fft = 2 * spec.shape[-2] - 2
                    total = total + istft(spec * mask, n_fft, hop, _hann(n_fft, mask), self.frames)
                sources.append(total)
            return torch.stack(sources).cpu().numpy().transpose(0, 2, 1)
