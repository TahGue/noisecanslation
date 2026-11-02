"""Benchmark every live-capable model: size, offline quality, live quality, speed, memory.

    uv run python scripts/model_benchmark.py                      # the full sweep
    uv run python scripts/model_benchmark.py -m tiger -m htdemucs --rt-output Realtek

Each model runs in its own process (a crash or hang is recorded, the sweep goes on).
Phases: offline quality (all items), live-sim per preset (two items), and a real-time
engine run (capture -> GPU -> playback to --rt-output) that counts glitches and the
true delay. --rt-output must be an output nobody listens to.

Writes src/voiceonly/benchmarks.json (shown in the app) and docs/benchmarks.md.
Test set: data/testset (scripts/make_testset.py): TTS voices over synthetic music and
effects at -5..+8 dB. Numbers are comparable between models, not absolute.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from voiceonly import audio_io, metrics, separators
from voiceonly.live import StreamingProcessor, presets_for

ROOT = Path(__file__).resolve().parents[1]
# The sweep: light (TIGER, MRX), middle (HTDemucs, MDX), heavy (RoFormers), mixing
# long-standing references with 2025 community checkpoints.
SWEEP = ["tiger", "mrx", "htdemucs", "htdemucs_ft", "mdx-vocft", "mdx-kim2", "mdx23c",
         "bs-roformer", "bs-roformer-live", "bs-roformer-resurrection-live",
         "mel-roformer-live", "mel-roformer-big-live", "mel-roformer-fv7b-live",
         "mel-roformer-becruily-live"]
LIVE_ITEMS = ("b_bg+0dB", "d_bg+8dB")  # live-sim on two items keeps the sweep tractable
SR_TEST = 48000


def load_items(dataset: Path) -> list[dict]:
    items = []
    for d in sorted(p for p in dataset.iterdir() if (p / "mix.wav").exists()):
        item = {"id": d.name}
        for stem in ("mix", "speech", "music", "sfx"):
            if (d / f"{stem}.wav").exists():
                item[stem], _ = audio_io.load(d / f"{stem}.wav")
        items.append(item)
    return items


def stem_scores(stems: np.ndarray, names: list[str], item: dict, sr: int) -> dict:
    """SI-SDR per stem against the references the test item has."""
    back = lambda a: audio_io.resample(np.ascontiguousarray(a), sr, SR_TEST)  # noqa: E731
    voice = back(stems[0])
    q = metrics.evaluate(voice, item["mix"], item["speech"], item["mix"] - item["speech"],
                         SR_TEST)
    out = {"voice": q["si_sdr"], "voice_kept_db": q["speech_gain_db"],
           "background_db": q["leakage_db"], "stoi": q["stoi"]}
    for name, ref in (("music", "music"), ("effects", "sfx")):
        if name in names and ref in item:
            out[name] = metrics.si_sdr(back(stems[names.index(name)]), item[ref])
    return out


def offline(sep, items) -> dict:
    sr = sep.info.sample_rate
    scores, rtf = [], []
    for item in items:
        x = audio_io.resample(item["mix"], SR_TEST, sr)
        t = time.perf_counter()
        stems = _chunked(sep, x)
        rtf.append((time.perf_counter() - t) / (len(x) / sr))
        scores.append(stem_scores(stems, sep.info.stems, item, sr))
    return {**_mean(scores), "rtf": float(np.mean(rtf))}


def _chunked(sep, x):
    """Offline stems through the model's own long-chunk path."""
    from voiceonly.chunking import process_chunked

    sr, n, ch = sep.info.sample_rate, len(x), x.shape[1]
    chunk, overlap = int(sep.info.chunk_seconds * sr), int(sep.info.overlap_seconds * sr)
    n_stems = len(sep.info.stems)
    # All stems in one pass: (stems, frames, ch) <-> (frames, stems * ch).
    flat = lambda c: sep.separate_stems(c).transpose(1, 0, 2).reshape(len(c), -1)  # noqa: E731
    y = np.concatenate(list(process_chunked(lambda s, k: x[s:s + k], n, flat, chunk, overlap)))
    return y.reshape(n, n_stems, ch).transpose(1, 0, 2)


def live(sep, items, preset_name: str) -> dict:
    import torch

    preset = presets_for(sep.info.key)[preset_name]
    sr = sep.info.sample_rate
    scores, steps = [], []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for item in items:
        x = audio_io.resample(item["mix"], SR_TEST, sr)
        proc = StreamingProcessor(sep, preset.config(), stems=True)
        hop, delay = proc.hop, proc.delay_frames
        padded = np.concatenate([x, np.zeros((delay + hop, x.shape[1]), np.float32)])
        out = []
        for start in range(0, len(x) + delay, hop):
            block = padded[start:start + hop]
            if len(block) < hop:
                block = np.pad(block, ((0, hop - len(block)), (0, 0)))
            t = time.perf_counter()
            out.append(proc.step(block))
            steps.append(time.perf_counter() - t)
        stems = np.concatenate(out, axis=1)[:, delay:delay + len(x)]
        scores.append(stem_scores(stems, sep.info.stems, item, sr))
    ms = np.array(steps[10:]) * 1000
    return {**_mean(scores), "preset": preset_name, "engine_delay_s": round(preset.delay_s, 3),
            "hop_ms": preset.hop_s * 1000, "step_ms": float(ms.mean()),
            "step_ms_p99": float(np.percentile(ms, 99)),
            "headroom": float(preset.hop_s * 1000 / np.percentile(ms, 99)),
            "vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)
            if torch.cuda.is_available() else None}


def _mean(rows: list[dict]) -> dict:
    keys = {k for r in rows for k in r}
    return {k: float(np.mean([r[k] for r in rows if r.get(k) is not None]))
            for k in keys if any(r.get(k) is not None for r in rows)}


def params_m(sep) -> float | None:
    model = getattr(sep, "model", None)
    if model is None or not hasattr(model, "parameters"):
        return None
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 2)


def realtime(sep, item: dict, preset_name: str, output: str, seconds: float = 29.0) -> dict:
    """Run the real engine in real time (file in, device out) and count glitches."""
    from voiceonly.live import LiveEngine, find_device

    preset = presets_for(sep.info.key)[preset_name]
    sr = sep.info.sample_rate
    x = audio_io.resample(item["mix"], SR_TEST, sr)[: int(seconds * sr)] * 0.5
    eng = LiveEngine(sep, preset.config(), None, find_device(output, "output"),
                     jitter_s=preset.jitter_s)
    t = time.perf_counter()
    eng.run(input_audio=x.astype(np.float32))
    s = eng.stats
    ms = sorted(s.step_ms)
    return {"preset": preset_name, "seconds": round(time.perf_counter() - t, 1),
            "steps": s.steps, "underruns": s.underrun_events,
            "short_callbacks": s.underruns, "dropped": s.dropped_blocks,
            "drift_drops": s.drift_drops,
            "glitches": s.underrun_events + s.dropped_blocks + s.drift_drops,
            "delay_s": round(eng.latency_s, 3),
            # Where the safety buffer actually settled (buffer level just before each
            # block lands, smoothed), vs the starting preset value.
            "buffer_ms": round(getattr(eng, "trough_avg", eng.jitter) / sr * 1000, 1),
            "buffer_start_ms": round(eng.jitter / sr * 1000, 1),
            "device_latency_ms": round(eng.output_latency_s * 1000, 1),
            "step_ms_max_recent": round(ms[-1], 1) if ms else None}


def run_model(key: str, items: list[dict], rt_output: str | None) -> dict:
    import torch

    sep = separators.create(key)
    t = time.perf_counter()
    sep.initialize()
    row = {"name": sep.info.name, "stems": sep.info.stems, "license": sep.info.license,
           "params_m": params_m(sep), "load_s": round(time.perf_counter() - t, 1)}
    print(f"   loaded in {row['load_s']} s, {row['params_m']} M params", flush=True)
    row["offline"] = offline(sep, items)
    print(f"   offline voice {row['offline']['voice']:.2f} dB, RTF {row['offline']['rtf']:.3f}",
          flush=True)
    if sep.info.live_window_seconds is not None:
        live_items = [i for i in items if i["id"] in LIVE_ITEMS] or items[:2]
        row["live"] = {}
        for name in presets_for(key):
            row["live"][name] = live(sep, live_items, name)
            lv = row["live"][name]
            print(f"   live {name}: voice {lv['voice']:.2f} dB, {lv['step_ms']:.1f} ms/step "
                  f"(p99 {lv['step_ms_p99']:.1f}), headroom {lv['headroom']:.1f}x", flush=True)
        if rt_output and torch.cuda.is_available():
            rt_item = next(i for i in items if i["id"] == LIVE_ITEMS[0])
            row["realtime"] = {}
            for name in ("balanced", "fast"):
                if name in presets_for(key):
                    rt = realtime(sep, rt_item, name, rt_output)
                    row["realtime"][name] = rt
                    print(f"   realtime {name}: glitches {rt['glitches']} (underruns "
                          f"{rt['underruns']}, dropped {rt['dropped']}, drift {rt['drift_drops']})"
                          f", delay {rt['delay_s']} s", flush=True)
    sep.close()
    return row


def main() -> int:
    import subprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", action="append")
    ap.add_argument("--dataset", default=str(ROOT / "data/testset"))
    ap.add_argument("--rt-output", default="Realtek",
                    help="silent output for the real-time phase ('' to skip it)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--timeout", type=float, default=2400, help="seconds per model")
    args = ap.parse_args()
    items = load_items(Path(args.dataset))
    per_model = ROOT / "runs" / "benchmark"
    per_model.mkdir(parents=True, exist_ok=True)

    if args.worker:  # child process: one model, result to its own JSON file
        row = run_model(args.worker, items, args.rt_output or None)
        (per_model / f"{args.worker}.json").write_text(json.dumps(row, indent=1))
        return 0

    out_json = ROOT / "src/voiceonly/benchmarks.json"
    previous = json.loads(out_json.read_text()) if out_json.exists() else {"models": {}}
    results, failed = previous.get("models", {}), {}
    for key in args.model or SWEEP:
        print(f"== {key}", flush=True)
        (per_model / f"{key}.json").unlink(missing_ok=True)
        cmd = [sys.executable, __file__, "--worker", key, "--dataset", args.dataset,
               "--rt-output", args.rt_output]
        try:
            proc = subprocess.run(cmd, timeout=args.timeout)
            if proc.returncode == 0 and (per_model / f"{key}.json").exists():
                results[key] = json.loads((per_model / f"{key}.json").read_text())
            else:
                failed[key] = f"exit code {proc.returncode}"
        except subprocess.TimeoutExpired:
            failed[key] = f"timed out after {args.timeout:.0f} s"
        if key in failed:
            print(f"   FAILED: {failed[key]}", flush=True)
        write_outputs(out_json, results, failed)
    return 0


def write_outputs(out_json: Path, results: dict, failed: dict) -> None:
    import torch

    meta = {"date": date.today().isoformat(), "gpu": torch.cuda.get_device_name()
            if torch.cuda.is_available() else "cpu", "os": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "items": sorted(p.name for p in (ROOT / "data/testset").iterdir())}
    out_json.write_text(json.dumps({"meta": meta, "models": results, "failed": failed},
                                   indent=1))
    write_markdown(meta, results, failed)


def write_markdown(meta: dict, results: dict, failed: dict) -> None:
    f = lambda v, d=1: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["offline"]["voice"])
    lines = [f"# Model benchmark ({meta['date']})", "",
             f"{meta['gpu']}, {meta['os']}, Python {meta['python']}, torch {meta['torch']}.",
             f"Test set: {', '.join(meta['items'])}: TTS voices over synthetic music and "
             "effects (-5 to +8 dB background). Comparable between models, not absolute; "
             "no singing in the backgrounds.",
             "SI-SDR in dB (higher is better). Voice kept: 0 dB = all speech preserved.", "",
             "## Offline quality (whole files), best voice first", "",
             "| Model | Params (M) | Voice | Music | Effects | Voice kept | STOI | RTF |",
             "|---|---|---|---|---|---|---|---|"]
    for key, r in ranked:
        o = r["offline"]
        lines.append(f"| {key} | {f(r['params_m'], 1)} | {f(o['voice'])} | {f(o.get('music'))} | "
                     f"{f(o.get('effects'))} | {f(o['voice_kept_db'], 2)} | "
                     f"{f(o.get('stoi'), 3)} | {f(o['rtf'], 3)} |")
    lines += ["", "## Live (simulated, items b and d), per preset", "",
              "| Model | Preset | Engine delay (s) | GPU ms/step (p99) | Headroom | VRAM (MB) | "
              "Voice | Music | Effects |", "|---|---|---|---|---|---|---|---|---|"]
    for key, r in ranked:
        for name, lv in r.get("live", {}).items():
            lines.append(f"| {key} | {name} | {f(lv['engine_delay_s'], 2)} | "
                         f"{f(lv['step_ms'])} ({f(lv['step_ms_p99'])}) | "
                         f"{f(lv['headroom'])}x | {lv['vram_mb']} | {f(lv['voice'])} | "
                         f"{f(lv.get('music'))} | {f(lv.get('effects'))} |")
    lines += ["", "## Real time (actual engine, 29 s clip, played to a silent output; glitch = audible gap or skip)", "",
              "| Model | Preset | Glitches | Gaps | Dropped | Drift drops | Delay (s) |",
              "|---|---|---|---|---|---|---|"]
    for key, r in ranked:
        for name, rt in r.get("realtime", {}).items():
            lines.append(f"| {key} | {name} | **{rt['glitches']}** | {rt['underruns']} | "
                         f"{rt['dropped']} | {rt['drift_drops']} | {f(rt['delay_s'], 2)} |")
    if failed:
        lines += ["", "## Failed", ""] + [f"- {k}: {v}" for k, v in failed.items()]
    lines += ["", "Delay = engine delay + device buffer; add ~0.2 s for Bluetooth headphones. "
              "Headroom = step size / p99 step time; below ~1.5x expect glitches when other "
              "apps use the GPU."]
    (ROOT / "docs/benchmarks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
