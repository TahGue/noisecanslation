"""Candidate model wrappers. Heavy dependencies are imported in initialize()."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import ModelInfo, SeparationError, SpeechSeparator, resolve_torch_device


class Passthrough(SpeechSeparator):
    """Returns the mix unchanged. Checks the pipeline and gives the no-op baseline."""

    info = ModelInfo(
        key="passthrough", name="Identity", version="1", sample_rate=48000, channels=2,
        chunk_seconds=10.0, overlap_seconds=1.0, lookahead_seconds=0.0,
        streaming=True, license="n/a", live_window_seconds=2.0,
    )

    def separate(self, chunk: np.ndarray) -> np.ndarray:
        return chunk


class _TorchSeparator(SpeechSeparator):
    def initialize(self) -> None:
        import torch

        self.torch = torch
        self.device = resolve_torch_device(self.device)
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

    def runtime_info(self) -> dict:
        info = {"device": self.device}
        if self.device.startswith("cuda"):
            info["gpu"] = self.torch.cuda.get_device_name()
            info["peak_vram_mb"] = round(self.torch.cuda.max_memory_allocated() / 2**20, 1)
        return info

    def close(self) -> None:
        self.model = None
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


class HTDemucs(_TorchSeparator):
    """HTDemucs 'vocals' stem. A music-trained model, so it is a baseline, not a dialogue model."""

    info = ModelInfo(
        key="htdemucs", name="HTDemucs (vocals stem)", version="htdemucs",
        sample_rate=44100, channels=2, chunk_seconds=60.0, overlap_seconds=2.0,
        # 4 s scored as well as the full 7.8 s training window on the live test clip
        # and runs ~2x faster (20 ms per step in fp16 on an RTX 4090).
        license="MIT", requires=["demucs"], live_window_seconds=4.0,
    )

    def __init__(self, device: str = "auto", half: bool = True) -> None:
        super().__init__(device)
        self.half = half
        self.runners: dict[int, object] = {}

    def initialize(self) -> None:
        super().initialize()
        from demucs.pretrained import get_model

        self.model = get_model(self.info.version).to(self.device).eval()
        if "vocals" not in self.model.sources:
            raise SeparationError(f"model has no vocals stem: {self.model.sources}")
        self.stem = self.model.sources.index("vocals")
        # get_model() always returns a bag; a bag of one can run its model directly.
        models = getattr(self.model, "models", [self.model])
        self.single = models[0] if len(models) == 1 else None
        self.segment_frames = (int(float(self.single.segment) * self.info.sample_rate)
                               if self.single is not None else 0)
        self.torch.backends.cuda.matmul.allow_tf32 = True
        self.torch.backends.cudnn.allow_tf32 = True

    def _runner(self, frames: int):
        """CUDA-graph runner for one live window length (built on first use)."""
        if frames not in self.runners:
            import copy

            from .fast_demucs import GraphRunner

            # Own copy: the runner changes padding behaviour, which offline runs must not see.
            model = copy.deepcopy(self.single)
            self.runners[frames] = GraphRunner(model, frames, self.stem, self.half)
        return self.runners[frames]

    def runtime_info(self) -> dict:
        info = super().runtime_info()
        if self.runners:
            info["live_fp16"] = self.half
            info["live_cuda_graph"] = all(r.graph is not None for r in self.runners.values())
        return info

    def close(self) -> None:
        self.runners.clear()
        super().close()

    def separate(self, chunk: np.ndarray) -> np.ndarray:
        from demucs.apply import apply_model

        if (self.single is not None and self.device.startswith("cuda")
                and chunk.shape[0] <= self.segment_frames):
            return self._runner(chunk.shape[0])(chunk)  # live path
        x = self.torch.from_numpy(chunk.T.copy())[None].to(self.device)
        with self.torch.inference_mode():
            if self.single is not None and chunk.shape[0] <= self.segment_frames:
                # Live path: one forward pass. shifts=0, since a random shift would push
                # the input past the training length and double the work.
                out = apply_model(self.single, x, device=self.device, split=False, shifts=0,
                                  progress=False)
            else:
                out = apply_model(self.model, x, device=self.device, split=True,
                                  overlap=0.25, progress=False)
        return out[0, self.stem].cpu().numpy().T


class HTDemucsFT(HTDemucs):
    """Fine-tuned bag of four HTDemucs models: better, but ~4x the compute. Offline only."""

    info = ModelInfo(
        key="htdemucs_ft", name="HTDemucs fine-tuned (vocals)", version="htdemucs_ft",
        sample_rate=44100, channels=2, chunk_seconds=60.0, overlap_seconds=2.0,
        license="MIT", requires=["demucs"],
    )


class DeepFilterNet(_TorchSeparator):
    """DeepFilterNet 3 speech enhancement. Low-latency candidate for live mode."""

    info = ModelInfo(
        key="deepfilternet", name="DeepFilterNet", version="3", sample_rate=48000,
        channels=1, chunk_seconds=20.0, overlap_seconds=0.5, lookahead_seconds=0.04,
        streaming=True, license="MIT OR Apache-2.0", requires=["deepfilternet"],
        live_window_seconds=2.0,
    )

    def initialize(self) -> None:
        super().initialize()
        from df.enhance import init_df

        self.model, self.df_state, _ = init_df()
        sr = self.df_state.sr()
        if sr != self.info.sample_rate:
            raise SeparationError(f"DeepFilterNet reports {sr} Hz, expected {self.info.sample_rate}")

    def separate(self, chunk: np.ndarray) -> np.ndarray:
        from df.enhance import enhance

        x = self.torch.from_numpy(chunk.T.copy())
        y = enhance(self.model, self.df_state, x)
        return y.cpu().numpy().T


class AudioSeparatorModel(SpeechSeparator):
    """Any vocals-capable checkpoint run through the audio-separator package (file API).

    Subclasses below pin one checkpoint each. Community checkpoints often have no
    clear license: fine for local evaluation, check before redistributing.
    """

    CHECKPOINT = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

    info = ModelInfo(
        key="roformer", name="BS-RoFormer Viperx 1297", version=CHECKPOINT,
        sample_rate=44100, channels=2, chunk_seconds=120.0, overlap_seconds=4.0,
        license="per-checkpoint, verify", requires=["audio-separator"],
    )

    def __init__(self, device: str = "auto", checkpoint: str | None = None) -> None:
        super().__init__(device)
        if checkpoint:
            self.info = ModelInfo(**{**self.info.to_dict(), "version": checkpoint})

    def initialize(self) -> None:
        import os

        from audio_separator.separator import Separator

        from ..audio_io import ensure_ffmpeg_on_path

        if ensure_ffmpeg_on_path() is None:
            raise SeparationError("audio-separator needs ffmpeg: uv sync --extra roformer")
        # ONNX models run on onnxruntime-gpu (CUDA 12), which reuses the CUDA/cuDNN DLLs
        # loaded by the cu128 PyTorch wheel; import torch first so they are found.
        import onnxruntime
        import torch  # noqa: F401

        onnxruntime.preload_dlls()
        cache =Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VoiceOnly" / "models"
        self.tmp = tempfile.TemporaryDirectory(prefix="voiceonly-audiosep-")
        self.sep = Separator(output_dir=self.tmp.name, output_format="WAV",
                             model_file_dir=str(cache), log_level=30)
        self.sep.load_model(model_filename=self.info.version)

    def separate(self, chunk: np.ndarray) -> np.ndarray:
        tmp = Path(self.tmp.name)
        src = tmp / "chunk.wav"
        sf.write(src, chunk, self.info.sample_rate, subtype="FLOAT")
        outputs = [tmp / Path(p).name for p in self.sep.separate(str(src))]
        try:
            vocal = next(p for p in outputs if "vocal" in p.name.lower())
        except StopIteration:
            raise SeparationError(f"no vocals stem among {[p.name for p in outputs]}") from None
        y, sr = sf.read(vocal, dtype="float32", always_2d=True)
        for p in outputs:
            p.unlink(missing_ok=True)
        if sr != self.info.sample_rate:
            raise SeparationError(f"checkpoint output is {sr} Hz")
        if y.shape[0] < chunk.shape[0]:
            y = np.pad(y, ((0, chunk.shape[0] - y.shape[0]), (0, 0)))
        return y[: chunk.shape[0]]

    def close(self) -> None:
        self.tmp.cleanup()


def _audiosep(key: str, name: str, checkpoint: str) -> type[AudioSeparatorModel]:
    info = ModelInfo(**{**AudioSeparatorModel.info.to_dict(), "key": key, "name": name,
                        "version": checkpoint})
    return type(f"AudioSep_{key.replace('-', '_')}", (AudioSeparatorModel,), {"info": info})


AUDIOSEP_MODELS = [
    _audiosep("bs-roformer", "BS-RoFormer Viperx 1297", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
    _audiosep("mel-roformer", "Mel-RoFormer vocals (Kim)", "vocals_mel_band_roformer.ckpt"),
    _audiosep("mel-roformer-big", "Mel-RoFormer Big Beta 5e", "melband_roformer_big_beta5e.ckpt"),
    _audiosep("mdx23c", "MDX23C InstVoc HQ", "MDX23C-8KFFT-InstVoc_HQ.ckpt"),
    _audiosep("mdx-kim2", "MDX-Net Kim Vocal 2", "Kim_Vocal_2.onnx"),
    _audiosep("mdx-vocft", "MDX-Net Voc FT", "UVR-MDX-NET-Voc_FT.onnx"),
]
