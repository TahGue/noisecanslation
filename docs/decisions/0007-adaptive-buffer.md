# 0007 — Adaptive safety buffer, silence compression and audio-thread priority

Status: accepted (2026-11-02)

## Context
Every preset carried a fixed safety buffer (60-100 ms) between the GPU worker and
playback. The 14-model sweep first reported 13-25 "glitches" per 29 s run for every
model; that turned out to be the measurement: underruns were counted per playback
callback (one gap = several) and the deliberate end-of-stream drain was counted too.
Counted correctly (gap events, drain excluded), the app models had 0 gaps but
occasional drift drops (skips) on small-step presets, and more buffer than needed.

## Changes (live.py)
- **Adaptive safety buffer.** The engine tracks the buffer level just before each
  output block lands. If over 40 blocks it never fell below 10 ms, half of the spare
  is removed from the target; after any audible gap the target grows 1.5x + 10 ms at
  once. Bounds: 15 ms .. max(3x preset, 300 ms).
- **Silence compression.** During near-silent blocks (RMS < 0.003, frequent between
  phrases in voice-only output) a block may be shortened or lengthened by up to half
  at once, so the buffer reaches a new target in seconds. Speech blocks keep the
  inaudible <= 1% speed change.
- **Runaway trim only on real runaway:** threshold max(3 steps, 0.25 s) instead of
  3 steps, which was 0.15 s on 0.05 s-step presets and trimmed audio needlessly.
- **Windows scheduling:** 1 ms timer resolution while running; capture threads
  registered with MMCSS as "Pro Audio".
- Glitch statistics count gap events and ignore the end-of-input drain.

## Measured (RTX 4090, scripts/realtime_check.py, 2 runs x 29 s per preset)

| Model, preset | Delay before | Delay after | Glitches before -> after |
|---|---|---|---|
| HTDemucs, balanced | 0.273 s | 0.208 s | 1 drift drop/run -> 0 |
| HTDemucs, fast | 0.203 s | 0.158 s | 0 -> 0 |
| TIGER, balanced | 0.423 s | 0.342 s | 0 -> 0 |
| TIGER, fast | 0.373 s | 0.288 s | 0 -> 0 |
| BS-RoFormer, balanced | 0.373 s | 0.296 s | 1 drift drop in 2 runs -> 0 |
| BS-RoFormer, fast | 0.293 s | 0.216 s | 0 -> 0 |

Delay = engine + device buffer (Bluetooth adds ~0.2 s). The buffer settled at 15-24 ms.

## Caveats
- Measured with a file fed in real time to a silent output; real capture adds its
  own jitter, which the adaptive buffer absorbs by growing after a gap.
- A machine with less GPU headroom will settle at a larger buffer (by design).
