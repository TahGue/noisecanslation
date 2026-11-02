"""End-to-end live test on real devices (driver-free chain).

Plays a reference mix to SINK (as any app would), runs `voiceonly live` capturing
SINK's loopback and playing to LISTEN, records LISTEN's loopback, then reports:
measured delay, glitches, tonal balance vs the offline-simulated output, and how
far background sits below speech during voice pauses.

    uv run python scripts/live_e2e.py --preset fast
    uv run python scripts/live_e2e.py --preset clean --sink PHL --listen WH-1000XM4

You will hear the filtered clip on LISTEN; SINK should be an output nobody hears.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundcard as sc
from scipy.signal import fftconvolve, welch

from voiceonly import audio_io, separators
from voiceonly.live import PRESETS, find_device, simulate

ROOT = Path(__file__).resolve().parents[1]


def frame_energy(x: np.ndarray, n: int = 4800) -> np.ndarray:
    return (x[: len(x) // n * n].reshape(-1, n) ** 2).mean(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="clean", choices=list(PRESETS))
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--sink", default="Realtek")
    ap.add_argument("--listen", default="WH-1000XM4")
    ap.add_argument("--item", default=str(ROOT / "data/tts/item_000"))
    ap.add_argument("--gain-db", type=float, default=-12.0)
    ap.add_argument("--json", help="append the result as a JSON line to this file")
    args = ap.parse_args()

    item = Path(args.item)
    mix, sr = audio_io.load(item / "mix.wav")
    speech, _ = audio_io.load(item / "speech.wav")
    sr = sr.sample_rate

    # Expected output: the same engine run offline, aligned to the input.
    sep = separators.create(args.model)
    sep.initialize()
    x = audio_io.resample(mix, sr, sep.info.sample_rate)
    expected, _ = simulate(x, sep, PRESETS[args.preset].config())
    expected = audio_io.resample(expected, sep.info.sample_rate, sr).mean(1)
    sep.close()

    seconds = mix.shape[0] / sr
    live = subprocess.Popen(
        ["uv", "run", "voiceonly", "live", "-i", f"loopback:{args.sink}", "-o", args.listen,
         "-m", args.model, "--preset", args.preset, "--duration", str(seconds + 14)],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    time.sleep(10)  # model load + CUDA graph capture

    rec: list[np.ndarray] = []

    def record() -> None:
        ctypes.windll.ole32.CoInitializeEx(None, 0)
        mic = sc.get_microphone(args.listen, include_loopback=True)
        with mic.recorder(samplerate=sr, channels=2, blocksize=sr // 100) as r:
            for _ in range(int((seconds + 3) * 100)):
                rec.append(r.record(numframes=sr // 100))

    t = threading.Thread(target=record)
    t.start()
    sd.play(mix * 10 ** (args.gain_db / 20), sr, device=find_device(args.sink, "output"),
            extra_settings=sd.WasapiSettings(auto_convert=True), blocking=True)
    t.join()
    out, err = live.communicate(timeout=60)
    summary = next((ln for ln in (out + err).replace("\r", "\n").splitlines()
                    if ln.startswith("steps ")), "")
    stats = {k: int(v) for k, v in re.findall(r"(underruns|dropped|drift drops) (\d+)", summary)}

    got = np.concatenate(rec).mean(1)
    lag = int(np.argmax(np.abs(fftconvolve(got, expected[::-1])))) - (len(expected) - 1)
    seg = got[max(lag, 0):max(lag, 0) + len(expected)]
    ref = expected[:len(seg)]

    f, pg = welch(seg, sr, nperseg=4096)
    _, pr = welch(ref, sr, nperseg=4096)
    bands = {f"{lo}-{hi}": round(10 * np.log10(pg[(f >= lo) & (f < hi)].sum()
                                               / pr[(f >= lo) & (f < hi)].sum()) - args.gain_db, 1)
             for lo, hi in [(0, 1000), (1000, 4000), (4000, 8000), (8000, 16000)]}

    es, eo = frame_energy(speech.mean(1)[:len(seg)]), frame_energy(seg)
    pause, talk = es < es.max() * 1e-4, es > es.max() * 1e-2
    result = {
        "preset": args.preset, "model": args.model,
        "delay_s": round(lag / sr, 3),
        "underruns": stats.get("underruns"), "dropped": stats.get("dropped"),
        "drift_drops": stats.get("drift drops"),
        "band_balance_db": bands,
        "pause_background_db": round(10 * np.log10(eo[pause].mean() / eo[talk].mean()), 1),
    }
    print(json.dumps(result))
    if args.json:
        with open(args.json, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
