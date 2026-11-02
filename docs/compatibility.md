# Compatibility

Run `uv run voiceonly doctor` (or **System check** in the app) to see what *this*
machine supports, including measured speed if the benchmark was run on the same GPU.

## Platforms

| | Windows 10/11 | Ubuntu / Linux | macOS (Apple Silicon) |
|---|---|---|---|
| App window, presets, mixer | yes | yes (CustomTkinter/Tk) | yes |
| Offline file separation (CLI, bench) | yes | yes | yes, CPU or Apple GPU (MPS) |
| Capture what the PC plays | yes: WASAPI loopback (driver-free) | not yet: needs a PipeWire/PulseAudio monitor-source backend | not yet: no system loopback; needs BlackHole or Core Audio taps (macOS 14.4+) |
| Switch default output automatically | yes (IPolicyConfig) | not yet (`pactl set-default-sink`) | not yet (Core Audio) |
| Global hotkey | yes (RegisterHotKey) | not yet (X11/Wayland differ) | not yet |
| Fast live paths (CUDA graphs) | NVIDIA + CUDA 12.8 | NVIDIA + CUDA | not available (no CUDA) |

## Models

"Live" needs a GPU fast enough that each step finishes well inside the step size
(headroom >= ~1.5x). Figures: `docs/benchmarks.md`, RTX 4090.

| Model | Size | Stems | License | NVIDIA GPU | CPU only | Apple GPU |
|---|---|---|---|---|---|---|
| TIGER-DnR (`tiger`) | 4.2 M params, 16 MB | voice, music, effects | MIT code, Apache-2.0 weights | live | offline only (slow) | offline (untested) |
| MRX (`mrx`) | 30.5 M, 122 MB | voice, music, effects | MIT | live | offline | offline (untested) |
| HTDemucs (`htdemucs`) | 42 M, 80 MB | voice, drums, bass, other | MIT | live, lightest | offline (~real time) | offline |
| BS-RoFormer (`bs-roformer-live`) | 160 M, 640 MB | voice, background | community checkpoint, unclear | live (strong GPU) | offline, very slow | offline (untested) |
| Offline-only: Mel-RoFormer Big, MDX23C, MDX-Net | 30-230 M | voice, background | community, unclear | offline | offline | offline |

Rough GPU guide for live use (scaled from the RTX 4090 step times; not measured on
other cards): HTDemucs runs on most RTX 20/30/40 cards; TIGER and MRX need roughly
RTX 3060 class or better; BS-RoFormer needs RTX 3080/4070 class or better.

## Known environment issues (Windows)
- Keep the virtual environment on a short path (`UV_PROJECT_ENVIRONMENT`, as
  `VoiceOnly.bat` does): CUDA DLLs fail to load past the 260-character path limit.
- `deepfilternet` pins `numpy<2` and cannot share the environment (decision 0004).
- `onnxruntime-gpu` is pinned to 1.23 (CUDA 12) to reuse PyTorch's CUDA 12.8 DLLs.
- Bluetooth headsets switch to a mono "hands-free" profile while a microphone is in
  use; the app handles it (mono output) but sound quality drops.
