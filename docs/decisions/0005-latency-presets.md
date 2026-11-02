# 0005 — Fast HTDemucs live path and latency presets

Status: accepted (2026-11-02)

## Context
The first live build used HTDemucs through `demucs.apply.apply_model` with a 7.5 s
window and 0.25 s steps: ~85 ms GPU time per step, ~0.9 s from app to headphones.

## Findings (RTX 4090, Windows 11)
- `apply_model` defaulted to `shifts=1`, padding past the 7.8 s training length and
  running two forward passes per step. Calling the inner model directly halved the time.
- demucs synchronises GPU and CPU several times per forward (debug asserts in
  `pad1d`, STFT windows and the 2-D position table rebuilt on the CPU, the envelope
  check inside `torch.istft`). Equivalent sync-free replacements
  (`separators/fast_demucs.py`) give identical output and allow CUDA-graph capture.
- The model is GPU-bound, not launch-bound: CUDA graphs alone saved ~2 ms.
  fp16 autocast saved ~20% with no measurable quality change (SI-SDR equal to 0.01 dB).
- Replacing GroupNorm(1, C) with LayerNorm was numerically equal but 1.8x slower; dropped.
- The window length dominates cost. With `use_train_segment=False` HTDemucs runs on
  shorter inputs; on the test clip 4 s scored better than the padded 7.8 s:

| Window | fp16 step | SI-SDR (live, 0.1 s step, 0.3 s lookahead) |
|---|---|---|
| 7.8 s | 37 ms | 13.5 dB |
| 6.0 s | 29 ms | 15.1 dB |
| 4.0 s | 20 ms | 15.0 dB |
| 3.0 s | 15 ms | 14.9 dB |
| 2.0 s | 11 ms | 14.5 dB |

- Lookahead matters more than window: 0.2 s keeps full quality; 0.05 s costs 2-4 dB.

## Decision
- Live HTDemucs runs a 4 s window in fp16 through a CUDA graph on its own model copy
  (offline processing is unchanged).
- Four presets trade delay for quality: Clean (0.45 s engine delay), Balanced (0.25 s),
  Fast (0.18 s), Ultra (0.12 s). Clean is the default. The app also offers custom values.
- Output streams request PortAudio's "low" latency and report the granted value.

## Caveats
- All quality numbers come from TTS voices over synthetic music/effects; real
  material may favour longer windows or more lookahead.
- Small steps need more GPU steps per second; with another GPU-heavy app running,
  step times rose from 20 ms to 32 ms (p99 56 ms), so Balanced and faster presets
  can glitch under load. The status line shows glitches; pick a slower preset if
  they climb.
