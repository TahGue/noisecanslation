"""Offline extraction: decode -> adapt to model -> chunked separation -> restore -> check."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__, audio_io, quality
from .chunking import process_chunked
from .separators import SpeechSeparator


def extract(audio: np.ndarray, sample_rate: int, separator: SpeechSeparator,
            on_progress=None) -> tuple[np.ndarray, dict]:
    """Return speech at the source rate and channel layout (up to stereo), plus timings."""
    info = separator.info
    t0 = time.perf_counter()
    x = audio_io.remix_channels(audio_io.resample(audio, sample_rate, info.sample_rate),
                                info.channels)
    t_prep = time.perf_counter()

    chunk = int(round(info.chunk_seconds * info.sample_rate))
    overlap = int(round(info.overlap_seconds * info.sample_rate))
    total = x.shape[0]
    blocks, done = [], 0
    for block in process_chunked(lambda s, n: x[s:s + n], total, separator.separate,
                                 chunk, overlap):
        blocks.append(block)
        done += block.shape[0]
        if on_progress:
            on_progress(done / total)
    y = np.concatenate(blocks) if blocks else np.zeros((0, info.channels), np.float32)
    t_sep = time.perf_counter()

    out_channels = audio.shape[1] if audio.shape[1] <= 2 else info.channels
    y = audio_io.remix_channels(audio_io.resample(y, info.sample_rate, sample_rate), out_channels)
    t_end = time.perf_counter()

    duration = audio.shape[0] / sample_rate
    timings = {
        "prepare_s": t_prep - t0,
        "separate_s": t_sep - t_prep,
        "restore_s": t_end - t_sep,
        "total_s": t_end - t0,
        "audio_s": duration,
        "rtf": (t_end - t0) / duration if duration else None,
    }
    return y, timings


def run_file(input_path: str | Path, out_dir: str | Path, separator: SpeechSeparator,
             on_progress=None) -> dict:
    out_dir = Path(out_dir)
    audio, src = audio_io.load(input_path)

    t0 = time.perf_counter()
    separator.initialize()
    load_s = time.perf_counter() - t0
    try:
        speech, timings = extract(audio, src.sample_rate, separator, on_progress)
        runtime = separator.runtime_info()
    finally:
        separator.close()

    qc = quality.check(speech, audio.shape[0], src.sample_rate, reference=audio)
    audio_io.save(out_dir / "speech.wav", speech, src.sample_rate)
    audio_io.save(out_dir / "speech.flac", speech, src.sample_rate)

    metadata = {
        "voiceonly_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": src.to_dict(),
        "output": {"sample_rate": src.sample_rate, "channels": int(speech.shape[1]),
                   "frames": int(speech.shape[0]), "files": ["speech.wav", "speech.flac"]},
        "model": separator.info.to_dict(),
        "quality": qc,
        "platform": {"os": platform.platform(), "python": platform.python_version()},
    }
    benchmark = {"model_load_s": load_s, **timings, **runtime}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2))
    return {"metadata": metadata, "benchmark": benchmark}
