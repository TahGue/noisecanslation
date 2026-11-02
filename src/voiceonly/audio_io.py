"""Decoding, resampling and writing audio.

Audio is always float32 shaped (frames, channels).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SNDFILE_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}


@dataclass
class SourceInfo:
    path: str
    sample_rate: int
    channels: int
    frames: int
    duration_s: float
    codec: str | None = None
    container: str | None = None
    decoder: str = "soundfile"

    def to_dict(self) -> dict:
        return asdict(self)


def ensure_ffmpeg_on_path() -> str | None:
    """Make `ffmpeg` callable by name, using imageio-ffmpeg's bundled binary if needed.

    Some libraries (audio-separator) shell out to plain `ffmpeg`. The bundled binary
    has a versioned file name, so a copy named ffmpeg.exe goes in a cache folder that
    is put on PATH for this process. Returns the ffmpeg path, or None if unavailable.
    """
    import os

    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    src = Path(imageio_ffmpeg.get_ffmpeg_exe())
    shim_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VoiceOnly" / "bin"
    shim = shim_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not shim.exists() or shim.stat().st_size != src.stat().st_size:
        shim_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, shim)
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    return str(shim)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe(path: str | Path) -> SourceInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in SNDFILE_EXTS:
        try:
            info = sf.info(str(path))
            return SourceInfo(str(path), info.samplerate, info.channels, info.frames,
                              info.frames / info.samplerate, info.subtype, info.format)
        except RuntimeError:
            pass  # fall through to ffmpeg
    if not ffmpeg_available():
        raise RuntimeError(
            f"Cannot read {path.name}: format needs ffmpeg, which is not on PATH "
            "(winget install Gyan.FFmpeg)."
        )
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels,codec_name,duration:format=format_name,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    if not data.get("streams"):
        raise RuntimeError(f"{path.name} has no audio stream")
    st = data["streams"][0]
    sr = int(st["sample_rate"])
    dur = float(st.get("duration") or data["format"]["duration"])
    return SourceInfo(str(path), sr, int(st["channels"]), int(round(dur * sr)), dur,
                      st.get("codec_name"), data["format"].get("format_name"), "ffmpeg")


def load(path: str | Path) -> tuple[np.ndarray, SourceInfo]:
    """Decode the first audio stream at its native rate and channel layout."""
    info = probe(path)
    if info.decoder == "soundfile":
        audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    else:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
             "-f", "f32le", "-acodec", "pcm_f32le", "-"],
            capture_output=True, check=True,
        )
        audio = np.frombuffer(out.stdout, dtype=np.float32).reshape(-1, info.channels).copy()
    info.frames = audio.shape[0]
    info.duration_s = info.frames / info.sample_rate
    return audio, info


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    ratio = Fraction(sr_out, sr_in).limit_denominator(1000)
    out = resample_poly(audio, ratio.numerator, ratio.denominator, axis=0)
    # Pin the length so durations stay exact after a round trip.
    target = int(round(audio.shape[0] * sr_out / sr_in))
    if out.shape[0] < target:
        out = np.pad(out, ((0, target - out.shape[0]), (0, 0)))
    return out[:target].astype(np.float32)


def remix_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    have = audio.shape[1]
    if have == channels:
        return audio
    if channels == 1:
        return audio.mean(axis=1, keepdims=True)
    if have == 1:
        return np.repeat(audio, channels, axis=1)
    if channels == 2:
        # Simple fold-down for >2 channels: keep L/R, add centre at -3 dB when present.
        lr = audio[:, :2].copy()
        if have >= 3:
            lr += 0.7071 * audio[:, 2:3]
        return lr
    raise ValueError(f"Cannot remix {have} channels to {channels}")


def save(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subtype = "FLOAT" if path.suffix.lower() == ".wav" else "PCM_24"
    data = audio if subtype == "FLOAT" else np.clip(audio, -1.0, 1.0)
    sf.write(str(path), data, sample_rate, subtype=subtype)
