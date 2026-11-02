"""Build a small DnR-layout test set from TTS voice clips + synthetic backgrounds.

    uv run python scripts/make_testset.py data/tts_raw data/testset

Each item: the voice line twice with pauses, mixed with synthetic music + effects
at a given background-to-voice level. Synthetic backgrounds contain no singing, so
this set cannot show whether a model keeps vocals in songs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from voiceonly import audio_io, synthetic

SR = 48000
# (voice clip, background level relative to voice in dB, seed)
ITEMS = [("a", -5.0, 1), ("b", 0.0, 2), ("c", 3.0, 3), ("d", 8.0, 4)]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def main(raw_dir: str, out_dir: str) -> None:
    for name, bg_db, seed in ITEMS:
        voice, info = audio_io.load(Path(raw_dir) / f"{name}.wav")
        voice = audio_io.remix_channels(audio_io.resample(voice, info.sample_rate, SR), 2)
        gap = np.zeros((int(1.5 * SR), 2), np.float32)
        speech = np.concatenate([gap, voice, gap, voice, gap])
        stems = synthetic.make_item(np.random.default_rng(seed), speech.shape[0] / SR, SR)
        n = min(speech.shape[0], stems["music"].shape[0])
        speech = speech[:n] / rms(speech[:n]) * 0.1
        bg = stems["music"][:n] + stems["sfx"][:n]
        scale = 0.1 * 10 ** (bg_db / 20) / rms(bg)
        music, sfx = stems["music"][:n] * scale, stems["sfx"][:n] * scale
        mix = speech + music + sfx
        g = 0.7 / float(np.abs(mix).max())
        d = Path(out_dir) / f"{name}_bg{bg_db:+.0f}dB"
        for stem, audio in {"speech": speech, "music": music, "sfx": sfx, "mix": mix}.items():
            audio_io.save(d / f"{stem}.wav", (audio * g).astype(np.float32), SR)
        print(f"{d.name}: {n / SR:.1f} s")


if __name__ == "__main__":
    main(*sys.argv[1:3])
