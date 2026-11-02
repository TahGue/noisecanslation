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
        stems=["voice", "drums", "bass", "other"],
    )
    # HTDemucs source name for each of our stems.
    SOURCE_FOR_STEM = {"voice": "vocals", "drums": "drums", "bass": "bass", "other": "other"}

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
        self.order = [self.model.sources.index(self.SOURCE_FOR_STEM[s]) for s in self.info.stems]
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
            self.runners[frames] = GraphRunner(model, frames, self.half)
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
        return self._sources(chunk)[self.stem]

    def separate_stems(self, chunk: np.ndarray) -> np.ndarray:
        return self._sources(chunk)[self.order]

    def _sources(self, chunk: np.ndarray) -> np.ndarray:
        """All model sources, (n_sources, frames, channels), in model.sources order."""
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
        return out[0].cpu().numpy().transpose(0, 2, 1)


class HTDemucsFT(HTDemucs):
    """Fine-tuned bag of four HTDemucs models: better, but ~4x the compute. Offline only."""

    info = ModelInfo(
        key="htdemucs_ft", name="HTDemucs fine-tuned (vocals)", version="htdemucs_ft",
        sample_rate=44100, channels=2, chunk_seconds=60.0, overlap_seconds=2.0,
        license="MIT", requires=["demucs"], stems=["voice", "drums", "bass", "other"],
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
            # Outputs are "<input>_(<Stem>)_<model>.wav". Match the stem in brackets:
            # checkpoint names such as Kim_Vocal_2 contain "vocal" in every file name.
            vocal = next(p for p in outputs if "(vocals)" in p.name.lower())
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


class RoFormerDirect(_TorchSeparator):
    """BS-RoFormer run in-process on the GPU (fp16), without audio-separator's file API.

    Uses audio-separator's model class and its downloaded checkpoint + YAML config.
    Much heavier than HTDemucs (160 M parameters): live it needs a short window
    and a larger step.
    """

    CHECKPOINT = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

    info = ModelInfo(
        key="bs-roformer-live", name="BS-RoFormer (in-process, fp16)", version=CHECKPOINT,
        sample_rate=44100, channels=2, chunk_seconds=8.0, overlap_seconds=1.0,
        license="per-checkpoint, verify", requires=["audio-separator"],
        live_window_seconds=1.5, live_min_hop_seconds=0.15,
    )

    def __init__(self, device: str = "auto", half: bool = True) -> None:
        super().__init__(device)
        self.half = half

    def initialize(self) -> None:
        import inspect
        import os
        import tempfile as _tempfile

        import yaml
        from audio_separator.separator import Separator
        from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer
        from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import (
            MelBandRoformer,
        )

        from ..audio_io import ensure_ffmpeg_on_path

        super().initialize()
        ensure_ffmpeg_on_path()  # audio-separator checks for ffmpeg on construction
        cache = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VoiceOnly" / "models"
        with _tempfile.TemporaryDirectory() as tmp:
            # Downloads checkpoint + config if missing; the config name does not always
            # follow the checkpoint name (e.g. config_melband_roformer_big_beta5e.yaml).
            fetch = Separator(output_dir=tmp, model_file_dir=str(cache), log_level=30)
            _, _, _, model_path, yaml_name = fetch.download_model_files(self.info.version)
        # The YAML stores band sizes as Python tuples, which safe_load rejects.
        loader = type("TupleLoader", (yaml.SafeLoader,), {})
        loader.add_constructor("tag:yaml.org,2002:python/tuple",
                               lambda ld, node: tuple(ld.construct_sequence(node)))
        cfg = yaml.load((cache / yaml_name).read_text(encoding="utf-8"), Loader=loader)["model"]
        cls = MelBandRoformer if "num_bands" in cfg else BSRoformer
        accepted = inspect.signature(cls.__init__).parameters
        model = cls(**{k: v for k, v in cfg.items() if k in accepted})
        state = self.torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        self.is_bs = cls is BSRoformer
        self.torch.backends.cuda.matmul.allow_tf32 = True

    LIVE_MAX_SECONDS = 4.0  # chunks up to this length use the CUDA-graph runner

    def _runner(self, frames: int):
        if not hasattr(self, "runners"):
            self.runners = {}
        if frames not in self.runners:
            from .fast_roformer import RoFormerGraphRunner

            self.runners[frames] = RoFormerGraphRunner(self.model, frames, self.half)
        return self.runners[frames]

    def runtime_info(self) -> dict:
        info = super().runtime_info()
        runners = getattr(self, "runners", {})
        if runners:
            info["live_cuda_graph"] = all(r.graph is not None for r in runners.values())
        return info

    def close(self) -> None:
        self.runners = {}
        super().close()

    def separate(self, chunk: np.ndarray) -> np.ndarray:
        torch = self.torch
        if (self.is_bs and self.device.startswith("cuda")
                and chunk.shape[0] <= self.LIVE_MAX_SECONDS * self.info.sample_rate):
            return self._runner(chunk.shape[0])(chunk)  # live path (BS-RoFormer only)
        x = torch.from_numpy(np.ascontiguousarray(chunk.T))[None].to(self.device)
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=self.half and self.device.startswith("cuda")):
            y = self.model(x)[0].float()
        if y.ndim == 3:  # (stems, channels, frames) for multi-stem checkpoints: voice first
            y = y[0]
        n = chunk.shape[0]
        if y.shape[-1] < n:
            y = torch.nn.functional.pad(y, (0, n - y.shape[-1]))
        return y[:, :n].cpu().numpy().T


def _roformer_live(key: str, name: str, checkpoint: str) -> type[RoFormerDirect]:
    info = ModelInfo(**{**RoFormerDirect.info.to_dict(), "key": key, "name": name,
                        "version": checkpoint})
    return type(f"RoFormerDirect_{key.replace('-', '_')}", (RoFormerDirect,), {"info": info})


# In-process (live-capable) RoFormers. Viperx 1297 / Kim / Big Beta 5e are the long-
# standing references; Resurrection (unwa), FV7b (gabox) and becruily are 2025 releases.
ROFORMER_LIVE_MODELS = [
    _roformer_live("bs-roformer-resurrection-live", "BS-RoFormer Resurrection (unwa)",
                   "bs_roformer_vocals_resurrection_unwa.ckpt"),
    _roformer_live("mel-roformer-live", "Mel-RoFormer vocals (Kim)",
                   "vocals_mel_band_roformer.ckpt"),
    _roformer_live("mel-roformer-big-live", "Mel-RoFormer Big Beta 5e (unwa)",
                   "melband_roformer_big_beta5e.ckpt"),
    _roformer_live("mel-roformer-fv7b-live", "Mel-RoFormer vocals FV7b (gabox)",
                   "mel_band_roformer_vocals_fv7b_gabox.ckpt"),
    _roformer_live("mel-roformer-becruily-live", "Mel-RoFormer vocals (becruily)",
                   "mel_band_roformer_vocals_becruily.ckpt"),
]


class MRXCinematic(_TorchSeparator):
    """MRX three-stem soundtrack separation (speech / music / effects), MERL cocktail-fork.

    Trained on Divide-and-Remaster, so it separates sound effects from music, which
    music-trained models cannot. Code vendored under separators/vendor (MIT).
    """

    WEIGHTS = "default_mrx_pre_trained_weights.pth"
    URL = ("https://media.githubusercontent.com/media/merlresearch/cocktail-fork-separation/"
           "main/checkpoints/" + WEIGHTS)
    TARGET_LUFS = -27.0  # level the model was trained at (DnR average)

    info = ModelInfo(
        key="mrx", name="MRX cinematic (voice / music / effects)", version=WEIGHTS,
        sample_rate=44100, channels=2, chunk_seconds=30.0, overlap_seconds=1.0,
        license="MIT (MERL cocktail-fork)", requires=[], live_window_seconds=3.0,
        live_min_hop_seconds=0.05, stems=["voice", "music", "effects"],
    )
    ORDER = [1, 0, 2]  # model output is music, speech, sfx

    def initialize(self) -> None:
        import os
        import urllib.request

        from .vendor.mrx import MRX

        super().initialize()
        path = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VoiceOnly" / "models" / "mrx"
        weights = path / self.WEIGHTS
        if not weights.exists():
            path.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.URL, weights.with_suffix(".part"))
            weights.with_suffix(".part").rename(weights)
        model = MRX()
        model.load_state_dict(self.torch.load(weights, map_location="cpu", weights_only=True))
        self.model = model.to(self.device).eval()
        self.torch.backends.cudnn.benchmark = True

    def _gain(self, chunk: np.ndarray) -> float:
        """Input gain to the training loudness (RMS approximation of LUFS)."""
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        level = 20 * np.log10(max(rms, 1e-6)) - 0.7
        return float(10 ** (np.clip(self.TARGET_LUFS - level, -20, 30) / 20))

    def separate(self, chunk: np.ndarray) -> np.ndarray:
        return self.separate_stems(chunk)[0]

    LIVE_MAX_SECONDS = 6.0  # chunks up to this length use the CUDA-graph runner

    def __init__(self, device: str = "auto", half: bool = True) -> None:
        super().__init__(device)
        self.half = half
        self.runners: dict[int, object] = {}

    def _runner(self, frames: int):
        if frames not in self.runners:
            from .fast_mrx import MRXGraphRunner

            self.runners[frames] = MRXGraphRunner(self.model, frames, self.info.channels,
                                                  self.half)
        return self.runners[frames]

    def close(self) -> None:
        self.runners = {}
        super().close()

    def separate_stems(self, chunk: np.ndarray) -> np.ndarray:
        torch = self.torch
        g = self._gain(chunk)
        if (self.device.startswith("cuda")
                and chunk.shape[0] <= self.LIVE_MAX_SECONDS * self.info.sample_rate):
            y = self._runner(chunk.shape[0])(chunk * g)  # (3, frames, channels)
            return y[self.ORDER] / g
        x = torch.from_numpy(np.ascontiguousarray(chunk.T) * g).to(self.device)
        with torch.inference_mode():
            y = self.model(x)  # (3, channels, frames)
        return (y[self.ORDER].float().cpu().numpy() / g).transpose(0, 2, 1)


class TigerCinematic(_TorchSeparator):
    """TIGER-DnR (Tsinghua, MIT): dialogue / music / effects from film soundtracks.

    Three small TIGER networks (4.2 M parameters in total), trained on Divide and
    Remaster. Beats MRX on every stem in our tests (voice 15.9 vs 10.9 dB). It is a
    mono model: stereo input is downmixed and the stems are returned on both channels.
    Code vendored under separators/vendor (MIT); weights from JusperLee/TIGER-DnR.
    """

    REPO = "JusperLee/TIGER-DnR"
    LIVE_MAX_SECONDS = 6.0  # chunks up to this length use the CUDA-graph runner

    info = ModelInfo(
        key="tiger", name="TIGER-DnR cinematic (voice / music / effects)",
        version="JusperLee/TIGER-DnR", sample_rate=44100, channels=2, chunk_seconds=36.0,
        overlap_seconds=4.0, license="MIT (code), Apache-2.0 (weights card)",
        requires=["safetensors"], live_window_seconds=2.0, live_min_hop_seconds=0.15,
        stems=["voice", "music", "effects"],
    )

    def __init__(self, device: str = "auto", half: bool = False) -> None:
        super().__init__(device)
        self.half = half  # fp16 is slower for TIGER (its many norms run in fp32)
        self.runners: dict[int, object] = {}

    def initialize(self) -> None:
        import json
        import os

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from .vendor.tiger_dnr import TIGERDNR

        super().initialize()
        cache = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VoiceOnly" / "models" / "tiger"
        files = {}
        for name in ("config.json", "model.safetensors"):
            local = cache / name
            files[name] = local if local.exists() else Path(
                hf_hub_download(self.REPO, name, local_dir=cache))
        cfg = json.loads(files["config.json"].read_text(encoding="utf-8"))
        args = TIGERDNR.__init__.__code__.co_varnames
        model = TIGERDNR(**{k: v for k, v in cfg.items() if k in args})
        model.load_state_dict(load_file(files["model.safetensors"]))
        self.model = model.to(self.device).eval()

    def _runner(self, frames: int):
        if frames not in self.runners:
            from .fast_tiger import TigerGraphRunner

            self.runners[frames] = TigerGraphRunner(self.model, frames, 1, self.half)
        return self.runners[frames]

    def close(self) -> None:
        self.runners = {}
        super().close()

    def separate(self, chunk: np.ndarray) -> np.ndarray:
        return self.separate_stems(chunk)[0]

    def separate_stems(self, chunk: np.ndarray) -> np.ndarray:
        mono = chunk.mean(axis=1, keepdims=True).astype(np.float32)
        if (self.device.startswith("cuda")
                and chunk.shape[0] <= self.LIVE_MAX_SECONDS * self.info.sample_rate):
            stems = self._runner(chunk.shape[0])(mono)  # (3, frames, 1)
        else:
            torch = self.torch
            x = torch.from_numpy(mono.T.copy()).to(self.device)[None]  # (1, 1, frames)
            with torch.inference_mode():
                voice, effects, music = self.model(x)  # TIGER's own 12 s / 4 s chunking
            stems = torch.stack([voice, music, effects]).reshape(3, -1, 1).cpu().numpy()
        return np.repeat(stems, chunk.shape[1], axis=2)


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
