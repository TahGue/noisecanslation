# 0006 — Three live models and a stem mixer

Status: accepted (2026-11-02)

## Context
Users want to choose what they hear (voice only, no music, no effects, ...) and to
trade speed for quality. Live mode previously ran one model (HTDemucs) that only
returned a voice estimate.

## Bake-off (4-item TTS test set, -5 to +8 dB background; offline unless noted)

| Model | Params | Voice SI-SDR | Notes |
|---|---|---|---|
| BS-RoFormer Viperx 1297 | 160 M | 25.3 dB | best voice; keeps 100% of the speech (-0.01 dB) |
| Mel-RoFormer Big Beta 5e | — | 22.8 dB | |
| MDX23C InstVoc HQ | — | 18.1 dB | |
| HTDemucs | 42 M | 13.6 dB | 4 stems: vocals, drums, bass, other |
| HTDemucs FT (bag of 4) | 4x42 M | 13.2 dB | 3.4x slower, no gain here |
| MRX (MERL cocktail-fork) | 30.5 M | 10.9 dB | only model with separate music and effects stems |
| MDX-Net Voc FT | — | 12.3 dB | removes 4 dB of the voice |

Mel-RoFormer (Kim) and Kim Vocal 2 results were invalid (wrong output file picked,
because their checkpoint names contain "vocal"); not re-run yet.

## Live results (live-sim, same test set)
- HTDemucs, 4 s window fp16 + CUDA graph: 20 ms/step, ~13.6 dB at 0.25-0.45 s delay.
- BS-RoFormer, 1.5 s window fp16 + CUDA-graph core (separators/fast_roformer.py):
  30 ms/step, 22.6 dB at 0.25 s engine delay; 0.38 s measured app-to-headphones.
  The graph cut CPU time per step 8x, which stopped audio-capture starvation.

## Decision
- Every separator exposes named stems (`separate_stems`); the live engine mixes them
  with gains that can change while running.
- Mix modes are defined per stem *category* (voice / music / effects / background,
  `mixer.py`). Modes that need music and effects apart are only offered for models
  that have both (MRX).
- The app offers three models: Fast (HTDemucs), Best voice (BS-RoFormer) and
  Cinematic (MRX), each with its own latency presets.

## Caveats
- Synthetic backgrounds contain no singing, so they cannot show that music-trained
  models keep sung vocals, nor how MRX does on real film mixes it was trained for.
- Community RoFormer/MDX checkpoints have unclear licenses; MRX and HTDemucs are MIT.
