"""Low-overhead BS-RoFormer inference for live use.

BS-RoFormer runs ~1,500 small kernels per forward from Python (62 band MLPs,
einops reshapes, 12 axial transformer blocks). In live mode that Python work holds
the GIL long enough to starve the audio capture thread, which then delivers audio
in bursts. Capturing the network core (band split -> transformers -> masks) as a
CUDA graph turns each step into one call; the STFT and inverse STFT stay eager.
Output is identical to the model's own forward.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import pack, rearrange, unpack


class RoFormerGraphRunner:
    """Separates a fixed number of frames with one BS-RoFormer (num_stems == 1)."""

    def __init__(self, model, frames: int, half: bool) -> None:
        self.model, self.frames, self.half = model, frames, half
        self.device = next(model.parameters()).device
        self.kwargs = dict(model.stft_kwargs)
        self.window = model.stft_window_fn().to(self.device)
        self.channels = model.audio_channels
        spec = self._stft(torch.zeros(self.channels, frames, device=self.device))
        self.static_in = torch.zeros_like(self._features(spec))
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
                with torch.cuda.graph(graph):
                    self.static_out = self._core()
                self.graph = graph
            except Exception:
                torch.cuda.synchronize()
                self.graph = None  # eager fallback: same result, more CPU time

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        """(channels, frames) -> real STFT (1, freqs*channels, time, 2), as the model packs it."""
        spec = torch.stft(audio, **self.kwargs, window=self.window, return_complex=True)
        spec = torch.view_as_real(spec)[None]  # b s f t c
        return rearrange(spec, "b s f t c -> b (f s) t c")

    @staticmethod
    def _features(spec: torch.Tensor) -> torch.Tensor:
        return rearrange(spec, "b f t c -> b t (f c)").contiguous()

    def _core(self) -> torch.Tensor:
        m = self.model
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            x = m.band_split(self.static_in)
            for block in m.layers:
                if len(block) == 3:
                    linear, time_t, freq_t = block
                    x, ps = pack([x], "b * d")
                    x = linear(x)
                    (x,) = unpack(x, ps, "b * d")
                else:
                    time_t, freq_t = block
                x = rearrange(x, "b t f d -> b f t d")
                x, ps = pack([x], "* t d")
                x = time_t(x)
                (x,) = unpack(x, ps, "* t d")
                x = rearrange(x, "b f t d -> b t f d")
                x, ps = pack([x], "* f d")
                x = freq_t(x)
                (x,) = unpack(x, ps, "* f d")
            x = m.final_norm(x)
            return torch.stack([fn(x) for fn in m.mask_estimators], dim=1)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """(frames, channels) mix -> (frames, channels) voice estimate."""
        with torch.inference_mode():
            audio = torch.from_numpy(np.ascontiguousarray(chunk.T)).to(self.device)
            spec = self._stft(audio)
            self.static_in.copy_(self._features(spec))
            if self.graph is not None:
                self.graph.replay()
                mask = self.static_out
            else:
                mask = self._core()
            mask = rearrange(mask.float(), "b n t (f c) -> b n f t c", c=2)
            est = torch.view_as_complex(spec[:, None].contiguous()) * torch.view_as_complex(
                mask.contiguous())
            est = rearrange(est, "b n (f s) t -> (b n s) f t", s=self.channels)
            out = torch.istft(est, **self.kwargs, window=self.window, length=self.frames)
            return out.cpu().numpy().T
