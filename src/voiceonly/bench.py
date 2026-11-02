"""Run several models over one reference set and compare them fairly."""

from __future__ import annotations

import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np

from . import audio_io, metrics, quality, separators
from .pipeline import extract

MIX_NAMES = ("mix.wav", "mixture.wav", "mix.flac", "mixture.flac")
BACKGROUND_NAMES = ("background.wav", "music.wav", "sfx.wav", "effects.wav", "noise.wav")
METRIC_KEYS = ("si_sdr", "si_sdr_i", "speech_gain_db", "leakage_db", "artifact_db", "stoi", "rtf")


def find_items(dataset: Path) -> list[dict]:
    items = []
    for d in sorted(p for p in dataset.rglob("*") if p.is_dir()) + [dataset]:
        mix = next((d / n for n in MIX_NAMES if (d / n).exists()), None)
        speech = next((d / n for n in ("speech.wav", "speech.flac") if (d / n).exists()), None)
        if mix and speech:
            bg = [d / n for n in BACKGROUND_NAMES if (d / n).exists()]
            items.append({"id": str(d.relative_to(dataset)).replace("\\", "/") or d.name,
                          "mix": mix, "speech": speech, "background": bg})
    return items


def load_item(item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mix, info = audio_io.load(item["mix"])
    speech, s_info = audio_io.load(item["speech"])
    if s_info.sample_rate != info.sample_rate:
        speech = audio_io.resample(speech, s_info.sample_rate, info.sample_rate)
    n = min(mix.shape[0], speech.shape[0])
    mix, speech = mix[:n], audio_io.remix_channels(speech[:n], mix.shape[1])
    if item["background"]:
        bg = np.zeros_like(mix)
        for path in item["background"]:
            b, b_info = audio_io.load(path)
            b = audio_io.remix_channels(audio_io.resample(b, b_info.sample_rate, info.sample_rate),
                                        mix.shape[1])[:n]
            bg[: b.shape[0]] += b
    else:
        bg = mix - speech
    return mix, speech, bg, info.sample_rate


def run(dataset: str | Path, model_keys: list[str], out_dir: str | Path,
        device: str = "auto", seed: int = 0, log=print) -> dict:
    dataset, out_dir = Path(dataset), Path(out_dir)
    items = find_items(dataset)
    if not items:
        raise SystemExit(f"No items in {dataset}: need folders with mix.wav and speech.wav")
    log(f"{len(items)} items, models: {', '.join(model_keys)}")

    rows, models_meta, failed = [], {}, {}
    for key in model_keys:
        try:
            rows += _run_model(key, items, out_dir, device, models_meta, log)
        except Exception as e:  # one broken model must not sink the whole comparison
            failed[key] = f"{type(e).__name__}: {e}"
            log(f"  {key:14s} FAILED: {failed[key]}")
        _write_results(out_dir, dataset, models_meta, failed, rows)

    if not rows:
        raise SystemExit("every model failed; see results.json")
    done = [k for k in model_keys if k in models_meta]
    export_listening(out_dir, items, done, seed)
    return {"summary": summarize(rows), "rows": rows, "failed": failed}


def _run_model(key, items, out_dir, device, models_meta, log) -> list[dict]:
    rows = []
    sep = separators.create(key, device=device)
    t0 = time.perf_counter()
    sep.initialize()
    load_s = time.perf_counter() - t0
    try:
        for item in items:
            mix, speech, bg, sr = load_item(item)
            est, timings = extract(mix, sr, sep)
            audio_io.save(out_dir / "outputs" / key / f"{item['id'].replace('/', '__')}.wav",
                          est, sr)
            row = {"model": key, "item": item["id"], **metrics.evaluate(est, mix, speech, bg, sr),
                   "rtf": timings["rtf"], "qc_ok": quality.check(est, mix.shape[0], sr, mix)["ok"]}
            rows.append(row)
            log(f"  {key:14s} {item['id']:24s} SI-SDR {row['si_sdr']:6.2f} dB  "
                f"leak {row['leakage_db']:6.1f} dB  speech {row['speech_gain_db']:5.1f} dB  "
                f"RTF {row['rtf']:.3f}")
        models_meta[key] = {**sep.info.to_dict(), "model_load_s": load_s, **sep.runtime_info()}
    finally:
        sep.close()
    return rows


def _write_results(out_dir, dataset, models_meta, failed, rows) -> None:
    """Rewritten after every model so an interrupted run keeps what finished."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(
        {"dataset": str(dataset), "models": models_meta, "failed": failed,
         "summary": summarize(rows), "rows": rows}, indent=2))
    if rows:
        with open(out_dir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for key in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == key]
        out[key] = {m: (float(np.mean(vals)) if (vals := [r[m] for r in mine if r[m] is not None])
                        else None) for m in METRIC_KEYS}
        out[key]["qc_failures"] = sum(not r["qc_ok"] for r in mine)
    return out


def export_listening(out_dir: Path, items: list[dict], model_keys: list[str], seed: int) -> None:
    """Copy outputs under shuffled letters per item; the mapping goes to key.json only."""
    rng = random.Random(seed)
    listen = out_dir / "listening"
    key: dict[str, dict] = {}
    for item in items:
        stem = item["id"].replace("/", "__")
        order = model_keys[:]
        rng.shuffle(order)
        folder = listen / stem
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(item["mix"], folder / "reference_mix.wav")
        key[stem] = {}
        for letter, model in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", order):
            shutil.copy(out_dir / "outputs" / model / f"{stem}.wav", folder / f"{letter}.wav")
            key[stem][letter] = model
    (out_dir / "key.json").write_text(json.dumps(key, indent=2))
