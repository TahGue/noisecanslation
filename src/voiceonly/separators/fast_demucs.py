"""Low-latency HTDemucs inference: no CPU syncs, CUDA graph replay, optional fp16.

demucs does several things per forward that stall the GPU pipeline (debug asserts
that read tensors back, STFT windows and position tables rebuilt on the CPU, and
torch.istft's envelope check). The patches below replace them with numerically
equivalent versions so a fixed-length forward can be captured as a CUDA graph.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_patched = False
_cache: dict = {}


def _hann(n: int, ref: torch.Tensor) -> torch.Tensor:
    key = ("hann", n, ref.device, ref.dtype)
    if key not in _cache:
        _cache[key] = torch.hann_window(n, device=ref.device, dtype=ref.dtype)
    return _cache[key]


def _pad1d(x, paddings, mode="constant", value=0.0):
    # demucs.hdemucs.pad1d minus its asserts, which force a GPU->CPU read.
    length = x.shape[-1]
    left, right = paddings
    if mode == "reflect":
        max_pad = max(left, right)
        if length <= max_pad:
            extra = max_pad - length + 1
            extra_right = min(right, extra)
            extra_left = extra - extra_right
            paddings = (left - extra_left, right - extra_right)
            x = F.pad(x, (extra_left, extra_right))
    return F.pad(x, paddings, mode, value)


def _spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    z = torch.stft(x, n_fft * (1 + pad), hop_length or n_fft // 4, window=_hann(n_fft, x),
                   win_length=n_fft, normalized=True, center=True, return_complex=True,
                   pad_mode="reflect")
    _, freqs, frames = z.shape
    return z.view(*other, freqs, frames)


def istft(z: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor, length: int):
    """torch.istft(normalized=True, center=True) without the CPU-side envelope check."""
    n_frames = z.shape[-1]
    frames = torch.fft.irfft(z, n=n_fft, dim=1) * (n_fft ** 0.5) * window[None, :, None]
    out_len = n_fft + hop * (n_frames - 1)
    ola = F.fold(frames, output_size=(1, out_len), kernel_size=(1, n_fft),
                 stride=(1, hop))[:, 0, 0]
    key = ("env", n_fft, hop, n_frames, window.device, window.dtype)
    if key not in _cache:
        w2 = (window ** 2)[None, :, None].expand(1, n_fft, n_frames)
        env = F.fold(w2, output_size=(1, out_len), kernel_size=(1, n_fft),
                     stride=(1, hop))[0, 0, 0]
        _cache[key] = torch.where(env > 1e-11, env, torch.ones_like(env))
    start = n_fft // 2
    return (ola / _cache[key])[:, start:start + length]


def _ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    if pad:
        raise NotImplementedError("ispectro with pad != 0")
    x = istft(z, n_fft, hop_length or n_fft // 4, _hann(n_fft, z.real), length)
    return x.view(*other, x.shape[-1])


def apply_patches() -> None:
    global _patched
    if _patched:
        return
    import demucs.hdemucs as hdemucs
    import demucs.htdemucs as htdemucs
    import demucs.transformer as transformer

    original_2d = transformer.create_2d_sin_embedding

    def cached_2d(d_model, height, width, device="cpu", max_period=10000):
        key = ("pos2d", d_model, height, width, str(device), max_period)
        if key not in _cache:
            _cache[key] = original_2d(d_model, height, width, device, max_period)
        return _cache[key]

    hdemucs.pad1d = htdemucs.pad1d = _pad1d
    htdemucs.spectro, htdemucs.ispectro = _spectro, _ispectro
    transformer.create_2d_sin_embedding = cached_2d
    _patched = True


class GraphRunner:
    """Runs one HTDemucs model on a fixed number of frames by replaying a CUDA graph.

    Calling it returns every source, shaped (n_sources, frames, channels), in the
    model's own order (model.sources).
    """

    def __init__(self, model, frames: int, half: bool) -> None:
        apply_patches()
        self.model, self.frames, self.half = model, frames, half
        training = int(model.segment * model.samplerate)
        # Shorter windows than training: run at the true length instead of padding to it.
        model.use_train_segment = frames >= training
        device = next(model.parameters()).device
        self.static_in = torch.zeros(1, model.audio_channels, frames, device=device)
        self.graph = None
        with torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._forward()
            torch.cuda.current_stream().wait_stream(stream)
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.static_out = self._forward()
                self.graph = graph
            except Exception:
                torch.cuda.synchronize()
                self.graph = None  # eager fallback: same result, a little slower

    def _forward(self) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            return self.model(self.static_in)[0]

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            self.static_in.copy_(torch.from_numpy(np.ascontiguousarray(chunk.T))[None])
            if self.graph is not None:
                self.graph.replay()
                out = self.static_out
            else:
                out = self._forward()
            return out.float().cpu().numpy().transpose(0, 2, 1)
