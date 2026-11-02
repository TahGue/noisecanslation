"""Synthetic DnR-layout items for pipeline and metric testing.

The "speech" is a voiced, syllable-modulated harmonic signal, not real speech.
Use it to test plumbing, never to choose a model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import audio_io


def make_item(rng: np.random.Generator, seconds: float, sr: int) -> dict[str, np.ndarray]:
    t = np.arange(int(seconds * sr)) / sr
    n = t.size

    f0 = 110 + 60 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6)) + rng.uniform(0, 40)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 12))
    syllables = np.clip(np.sin(2 * np.pi * rng.uniform(3, 5) * t), 0, None) ** 2
    phrases = (np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 6)) > -0.4).astype(float)
    speech = voice * syllables * phrases

    chord = [rng.choice([196.0, 220.0, 261.6, 293.7, 329.6]) * r for r in (1, 1.26, 1.5)]
    music = sum(np.sin(2 * np.pi * f * t) for f in chord) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t))
    sfx = np.cumsum(rng.standard_normal(n)) * 0.002
    sfx -= np.convolve(sfx, np.ones(sr // 50) / (sr // 50), mode="same")
    for start in rng.integers(0, n - sr // 4, size=int(seconds / 2)):
        burst = rng.standard_normal(sr // 4) * np.exp(-np.linspace(0, 8, sr // 4))
        sfx[start:start + sr // 4] += burst

    def stereo(x: np.ndarray, level_db: float, pan: float) -> np.ndarray:
        x = x / (np.sqrt(np.mean(x**2)) + 1e-9) * 10 ** (level_db / 20)
        return np.stack([x * (1 - pan), x * (1 + pan)], axis=1).astype(np.float32)

    stems = {
        "speech": stereo(speech, -20, 0.0),
        "music": stereo(music, rng.uniform(-28, -20), rng.uniform(-0.5, 0.5)),
        "sfx": stereo(sfx, rng.uniform(-30, -22), rng.uniform(-0.5, 0.5)),
    }
    mix = stems["speech"] + stems["music"] + stems["sfx"]
    # Same gain on every stem keeps mix == sum(stems) while peaking at -3 dBFS.
    gain = min(1.0, 0.708 / float(np.abs(mix).max()))
    stems = {k: v * gain for k, v in stems.items()}
    stems["mix"] = mix * gain
    return stems


def write_set(out_dir: str | Path, items: int = 5, seconds: float = 12.0,
              sr: int = 48000, seed: int = 0) -> list[Path]:
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(items):
        d = Path(out_dir) / f"item_{i:03d}"
        for name, audio in make_item(rng, seconds, sr).items():
            audio_io.save(d / f"{name}.wav", audio, sr)
        paths.append(d)
    return paths
