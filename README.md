# VoiceOnly — bake-off (v0.1)

**Live mode (filter your PC's audio to voices only): see [LIVE_SETUP.md](LIVE_SETUP.md).**

Offline speech extraction pipeline plus a benchmark to pick the model VoiceOnly
is built around. See `docs/decisions/` for the choices behind it.

## Setup (Windows)

```powershell
uv sync                         # core: numpy, scipy, soundfile
uv sync --extra demucs          # add HTDemucs (pulls PyTorch)
uv sync --extra roformer --extra metrics
```

For CUDA PyTorch on Windows, install the CUDA wheel per pytorch.org if the
default wheel reports `torch.cuda.is_available() == False`.

## Use

```powershell
uv run voiceonly models
uv run voiceonly separate movie_clip.wav -m htdemucs -o output/clip1
uv run voiceonly make-synthetic data/synthetic --items 5
uv run voiceonly bench data/synthetic -m passthrough -m htdemucs -o runs/first
```

`separate` writes `speech.wav`, `speech.flac`, `metadata.json`, `benchmark.json`.

`bench` expects one folder per item containing `mix.wav` (or `mixture.wav`) and
`speech.wav`, plus optional `music.wav`/`sfx.wav`/`background.wav` — the DnR
layout. It writes `results.json`, `results.csv`, a per-model summary and a
`listening/` folder of blinded samples with a separate `key.json`.

## Metrics

- `si_sdr` / `si_sdr_i`: separation quality vs the reference speech, and the
  improvement over the unprocessed mix.
- `speech_gain_db`: how much of the reference speech survives (0 dB = all of it;
  strongly negative = speech loss).
- `leakage_db`: background energy left in the output relative to the speech in
  it (lower is better).
- `stoi`: intelligibility estimate (needs the `metrics` extra).
- `rtf`: processing time / audio duration. Not latency.
