# Model benchmark (2026-11-02)

NVIDIA GeForce RTX 4090, Windows-11-10.0.26300-SP0, Python 3.13.18, torch 2.11.0+cu128.
Test set: a_bg-5dB, b_bg+0dB, c_bg+3dB, d_bg+8dB: TTS voices over synthetic music and effects (-5 to +8 dB background). Comparable between models, not absolute; no singing in the backgrounds.
SI-SDR in dB (higher is better). Voice kept: 0 dB = all speech preserved.

## Offline quality (whole files), best voice first

| Model | Params (M) | Voice | Music | Effects | Voice kept | STOI | RTF |
|---|---|---|---|---|---|---|---|
| bs-roformer | — | 25.3 | — | — | -0.01 | 0.993 | 0.928 |
| bs-roformer-live | 159.8 | 25.3 | — | — | -0.01 | 0.993 | 0.047 |
| mel-roformer-live | 228.2 | 23.6 | — | — | -0.09 | 0.994 | 0.024 |
| mel-roformer-fv7b-live | 228.2 | 23.3 | — | — | 0.01 | 0.996 | 0.024 |
| mel-roformer-becruily-live | 228.2 | 23.2 | — | — | -0.03 | 0.996 | 0.024 |
| mel-roformer-big-live | 369.9 | 22.8 | — | — | -0.07 | 0.994 | 0.025 |
| mdx23c | — | 18.1 | — | — | -0.34 | 0.991 | 0.463 |
| tiger | 4.2 | 15.9 | 13.3 | 8.4 | -0.34 | 0.972 | 0.302 |
| htdemucs | 42.0 | 13.3 | — | — | -0.61 | 0.974 | 0.031 |
| mdx-kim2 | — | 12.3 | — | — | -3.97 | 0.983 | 0.172 |
| mdx-vocft | — | 12.3 | — | — | -4.08 | 0.983 | 0.171 |
| htdemucs_ft | 167.9 | 11.9 | — | — | -0.51 | 0.968 | 0.089 |
| bs-roformer-resurrection-live | 51.0 | 11.9 | — | — | -1.08 | 0.968 | 0.033 |
| mrx | 30.5 | 11.0 | 11.2 | 7.5 | -0.87 | 0.953 | 0.083 |

## Live (simulated, items b and d), per preset

| Model | Preset | Engine delay (s) | GPU ms/step (p99) | Headroom | VRAM (MB) | Voice | Music | Effects |
|---|---|---|---|---|---|---|---|---|
| bs-roformer-live | clean | 0.50 | 41.8 (42.6) | 3.5x | 811 | 22.6 | — | — |
| bs-roformer-live | balanced | 0.35 | 28.6 (29.2) | 3.4x | 785 | 21.4 | — | — |
| bs-roformer-live | fast | 0.27 | 28.5 (29.2) | 3.4x | 679 | 18.5 | — | — |
| mel-roformer-live | clean | 0.50 | 32.5 (48.0) | 3.1x | 1046 | 21.8 | — | — |
| mel-roformer-live | balanced | 0.35 | 29.5 (51.4) | 1.9x | 1007 | 21.7 | — | — |
| mel-roformer-live | fast | 0.27 | 28.9 (35.5) | 2.8x | 1007 | 21.6 | — | — |
| mel-roformer-fv7b-live | clean | 0.50 | 32.4 (42.6) | 3.5x | 1046 | 21.6 | — | — |
| mel-roformer-fv7b-live | balanced | 0.35 | 30.1 (60.2) | 1.7x | 1007 | 21.5 | — | — |
| mel-roformer-fv7b-live | fast | 0.27 | 29.5 (49.0) | 2.0x | 1007 | 21.4 | — | — |
| mel-roformer-becruily-live | clean | 0.50 | 32.1 (40.6) | 3.7x | 1046 | 21.3 | — | — |
| mel-roformer-becruily-live | balanced | 0.35 | 29.4 (36.4) | 2.7x | 1007 | 20.4 | — | — |
| mel-roformer-becruily-live | fast | 0.27 | 29.4 (43.9) | 2.3x | 1007 | 20.4 | — | — |
| mel-roformer-big-live | clean | 0.50 | 35.6 (63.7) | 2.4x | 1586 | 21.1 | — | — |
| mel-roformer-big-live | balanced | 0.35 | 31.7 (54.1) | 1.8x | 1548 | 20.7 | — | — |
| mel-roformer-big-live | fast | 0.27 | 31.5 (39.2) | 2.5x | 1548 | 20.6 | — | — |
| tiger | clean | 0.55 | 106.0 (109.7) | 1.8x | 293 | 14.8 | 13.1 | 8.0 |
| tiger | balanced | 0.40 | 73.9 (76.1) | 2.0x | 250 | 14.4 | 13.1 | 7.9 |
| tiger | fast | 0.35 | 44.5 (45.7) | 2.2x | 197 | 13.7 | 11.7 | 7.3 |
| htdemucs | clean | 0.45 | 21.9 (27.5) | 3.6x | 556 | 11.9 | — | — |
| htdemucs | balanced | 0.25 | 21.3 (22.5) | 2.2x | 364 | 10.1 | — | — |
| htdemucs | fast | 0.18 | 16.7 (17.3) | 2.9x | 691 | 11.7 | — | — |
| htdemucs | ultra | 0.12 | 11.7 (12.3) | 3.2x | 825 | 11.4 | — | — |
| bs-roformer-resurrection-live | clean | 0.50 | 26.9 (27.6) | 5.4x | 370 | 5.2 | — | — |
| bs-roformer-resurrection-live | balanced | 0.35 | 18.6 (19.3) | 5.2x | 349 | 3.1 | — | — |
| bs-roformer-resurrection-live | fast | 0.27 | 18.6 (19.2) | 5.2x | 258 | 2.7 | — | — |
| mrx | clean | 0.45 | 29.7 (33.5) | 3.0x | 426 | 9.6 | 11.6 | 8.3 |
| mrx | balanced | 0.35 | 28.7 (29.9) | 3.3x | 426 | 9.5 | 11.5 | 8.1 |
| mrx | fast | 0.27 | 19.3 (20.8) | 4.8x | 454 | 9.1 | 11.0 | 8.0 |

## Real time (actual engine, 29 s clip, played to a silent output; glitch = audible gap or skip)

| Model | Preset | Glitches | Gaps | Dropped | Drift drops | Delay (s) |
|---|---|---|---|---|---|---|
| bs-roformer-live | balanced | **0** | 0 | 0 | 0 | 0.30 |
| bs-roformer-live | fast | **0** | 0 | 0 | 0 | 0.22 |
| tiger | balanced | **0** | 0 | 0 | 0 | 0.34 |
| tiger | fast | **0** | 0 | 0 | 0 | 0.29 |
| htdemucs | balanced | **0** | 0 | 0 | 0 | 0.21 |
| htdemucs | fast | **0** | 0 | 0 | 0 | 0.16 |

Delay = engine delay + device buffer; add ~0.2 s for Bluetooth headphones. Headroom = step size / p99 step time; below ~1.5x expect glitches when other apps use the GPU.
