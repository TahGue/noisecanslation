"""System check: what this machine can run, and how well.

`voiceonly doctor` prints it; the app shows the same report. Each check is guarded
so a missing library becomes a line in the report, never a crash.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from . import separators

BENCHMARKS = Path(__file__).with_name("benchmarks.json")

# Platform support for the live pipeline (capture + routing + hotkey), by OS.
LIVE_SUPPORT = {
    "Windows": "supported (WASAPI loopback, default-output switching, global hotkey)",
    "Linux": "not yet: needs a PipeWire/PulseAudio capture + routing backend",
    "Darwin": "not yet: macOS has no system loopback (needs BlackHole or Core Audio taps)",
}

# Per-model notes that do not depend on this machine.
MODEL_NOTES = {
    "htdemucs": ("MIT", "CUDA fast path; CPU/Apple GPU offline"),
    "tiger": ("MIT code, Apache-2.0 weights", "CUDA fast path; CPU offline (slow)"),
    "bs-roformer-live": ("community checkpoint, license unclear",
                         "CUDA fast path; needs a strong NVIDIA GPU live"),
}


@dataclass
class Check:
    name: str
    ok: bool | None  # None = informational
    detail: str


def load_benchmarks() -> dict:
    try:
        return json.loads(BENCHMARKS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"meta": {}, "models": {}}


def system_checks() -> list[Check]:
    checks = [Check("OS", None, platform.platform()),
              Check("Python", None, sys.version.split()[0])]
    system = platform.system()
    live = LIVE_SUPPORT.get(system, "unknown platform")
    checks.append(Check("Live filtering on this OS", system == "Windows", live))
    try:
        import torch

        cuda = torch.cuda.is_available()
        detail = f"torch {torch.__version__}"
        if cuda:
            props = torch.cuda.get_device_properties(0)
            detail += (f", CUDA {torch.version.cuda}, {props.name}, "
                       f"{props.total_memory / 2**30:.0f} GB VRAM")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            detail += ", Apple GPU (MPS): offline only, no CUDA fast paths"
        else:
            detail += ", no GPU: offline processing only, live mode too slow"
        checks.append(Check("GPU", cuda, detail))
    except ImportError:
        checks.append(Check("GPU", False, "PyTorch not installed (uv sync --extra demucs)"))
    if system == "Windows":
        try:
            import soundcard as sc

            outs = [m.name for m in sc.all_microphones(include_loopback=True) if m.isloopback]
            checks.append(Check("Audio capture", bool(outs),
                                f"{len(outs)} outputs can be captured (loopback)"))
        except Exception as e:  # noqa: BLE001 - report, don't crash
            checks.append(Check("Audio capture", False, f"soundcard unavailable: {e}"))
    from .audio_io import ffmpeg_available

    checks.append(Check("ffmpeg (video files)", ffmpeg_available() or None,
                        "found" if ffmpeg_available() else "not on PATH: WAV/FLAC/OGG only"))
    return checks


def model_checks(gpu_name: str | None = None) -> list[dict]:
    bench = load_benchmarks()
    same_gpu = gpu_name and bench.get("meta", {}).get("gpu") == gpu_name
    rows = []
    for key, (license_, runtime) in MODEL_NOTES.items():
        info = separators.REGISTRY[key].info
        missing = separators.missing_requirements(key)
        b = bench.get("models", {}).get(key, {})
        live = b.get("live", {}).get("balanced") or next(iter(b.get("live", {}).values()), None)
        rows.append({
            "key": key, "name": info.name, "stems": info.stems, "license": license_,
            "runtime": runtime, "installed": not missing,
            "install": separators.install_hint(key) if missing else "",
            "params_m": b.get("params_m"),
            "offline_voice": b.get("offline", {}).get("voice"),
            "live": live,
            "realtime": b.get("realtime", {}).get("balanced"),
            "measured_here": bool(same_gpu),
        })
    return rows


def report() -> str:
    lines = ["System"]
    checks = system_checks()
    for c in checks:
        mark = {True: "ok ", False: "!! ", None: "   "}[c.ok]
        lines.append(f"  {mark}{c.name}: {c.detail}")
    gpu_name = None
    try:
        import torch

        gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    except ImportError:
        pass
    lines += ["", "Models (benchmark: TTS test set, balanced preset)"]
    for m in model_checks(gpu_name):
        status = "installed" if m["installed"] else f"missing: {m['install']}"
        lines.append(f"  {m['key']:17s} {status}; stems {', '.join(m['stems'])}; {m['license']}")
        lv = m["live"]
        if lv:
            where = "measured on this GPU" if m["measured_here"] else "measured on another GPU"
            rt = m.get("realtime") or {}
            delay = (f"{rt['delay_s']:.2f} s delay, {rt['glitches']} glitches in real time"
                     if rt else f"{lv['engine_delay_s']:.2f} s delay (preset)")
            lines.append(f"  {'':17s} voice {lv['voice']:.1f} dB live / "
                         f"{m['offline_voice']:.1f} offline, {lv['step_ms']:.0f} ms per "
                         f"{lv['hop_ms']:.0f} ms step ({lv['headroom']:.1f}x), "
                         f"{lv['vram_mb']} MB VRAM, {delay} ({where})")
        lines.append(f"  {'':17s} {m['runtime']}")
    return "\n".join(lines)
