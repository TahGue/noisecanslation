"""Low-overhead TIGER-DnR inference for live use.

TIGER-DnR is three TIGER networks (dialogue, effects, music); each predicts all
three sources and the wrapper keeps one output per network. TIGER's forward builds
its STFT window on the CPU every call and runs ~40k small kernels, so a short live
window is dominated by overhead. Here the STFT stays eager, the band split ->
separator -> mask part of all three networks is captured as one CUDA graph
(optionally fp16), and the inverse STFT is the sync-free one from fast_demucs.
The maths is TIGER.forward's, reorganised; see test_fast_tiger_matches_model.
"""

from __future__ import annotations

import numpy as np
import torch

from .fast_demucs import _hann, istft

# (network attribute, source index it is kept for) in our stem order: voice, music, effects.
NETS = (("dialog", 2), ("music", 0), ("effect", 1))


def _masked_spec(net, spec_ri: torch.Tensor, source: int) -> torch.Tensor:
    """One TIGER network's estimate of `source`, as (batch, 2, freqs, frames) real/imag."""
    batch = spec_ri.shape[0]
    features, bands = [], []
    start = 0
    for i, width in enumerate(net.band_width):
        band = spec_ri[:, :, start:start + width].contiguous()  # B, 2, BW, T
        bands.append(band)
        features.append(net.BN[i](band.view(batch, width * 2, -1)))
        start += width
    sep = net.separator(torch.stack(features, 1).view(batch, net.nband, net.feature_dim, -1))
    sep = sep.view(batch, net.nband, net.feature_dim, -1)
    real_parts, imag_parts = [], []
    k = net.num_output
    for i, width in enumerate(net.band_width):
        out = net.mask[i](sep[:, i]).view(batch, 2, 2, k, width, -1)
        mask = out[:, 0] * torch.sigmoid(out[:, 1])  # B, 2, K, BW, T
        m_re, m_im = mask[:, 0], mask[:, 1]
        # force the masks of all sources to sum to 1 (as TIGER does), then keep one
        m_re = m_re - (m_re.sum(1, keepdim=True) - 1) / k
        m_im = m_im - m_im.sum(1, keepdim=True) / k
        m_re, m_im = m_re[:, source], m_im[:, source]  # B, BW, T
        s_re, s_im = bands[i][:, 0].float(), bands[i][:, 1].float()
        real_parts.append(s_re * m_re.float() - s_im * m_im.float())
        imag_parts.append(s_re * m_im.float() + s_im * m_re.float())
    return torch.stack([torch.cat(real_parts, 1), torch.cat(imag_parts, 1)], 1)


class TigerGraphRunner:
    """Separates a fixed number of frames into voice, music and effects."""

    def __init__(self, model, frames: int, channels: int, half: bool) -> None:
        self.model, self.frames, self.channels, self.half = model, frames, channels, half
        first = model.dialog
        self.n_fft, self.hop = first.win, first.stride
        self.device = next(model.parameters()).device
        spec = self._stft(torch.zeros(channels, frames, device=self.device))
        self.static_in = spec.clone()
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
                    self.static_out = self._core()
                self.graph = graph
            except Exception:
                torch.cuda.synchronize()
                self.graph = None  # eager fallback: same result, more CPU time

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        """(channels, frames) -> (channels, 2, freqs, frames) real/imag, TIGER's convention."""
        spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop,
                          window=_hann(self.n_fft, audio), return_complex=True)
        return torch.stack([spec.real, spec.imag], 1)

    def _core(self) -> torch.Tensor:
        # The three networks are independent and each underuses the GPU (many small
        # kernels), so they run side by side on their own streams.
        if not hasattr(self, "streams"):
            self.streams = [torch.cuda.Stream() for _ in NETS]
        main = torch.cuda.current_stream()
        est = [None] * len(NETS)
        for i, ((name, source), stream) in enumerate(zip(NETS, self.streams)):
            stream.wait_stream(main)
            with torch.cuda.stream(stream), torch.autocast(
                    "cuda", dtype=torch.float16, enabled=self.half):
                est[i] = _masked_spec(getattr(self.model, name), self.static_in, source).float()
        for stream in self.streams:
            main.wait_stream(stream)
        for tensor in est:
            # Made on a side stream, consumed on the main one: tell the caching
            # allocator, or it may hand the memory out again while still being read.
            tensor.record_stream(main)
        return torch.stack(est, 0)  # 3 stems, channels, 2, freqs, frames

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """(frames, channels) mix -> (3, frames, channels): voice, music, effects."""
        with torch.inference_mode():
            audio = torch.from_numpy(np.ascontiguousarray(chunk.T)).to(self.device)
            self.static_in.copy_(self._stft(audio))
            if self.graph is not None:
                self.graph.replay()
                est = self.static_out
            else:
                est = self._core()
            stems, channels, _, freqs, frames = est.shape
            z = torch.complex(est[:, :, 0], est[:, :, 1]).reshape(stems * channels, freqs, frames)
            out = istft(z, self.n_fft, self.hop, _hann(self.n_fft, est), self.frames,
                        normalized=False)
            return out.view(stems, channels, -1).cpu().numpy().transpose(0, 2, 1)
