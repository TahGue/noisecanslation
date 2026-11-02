# 0004 — Initial candidate models

Status: proposed (2026-11-02)

| Key | Model | Role | License note |
|---|---|---|---|
| `passthrough` | Identity | Pipeline sanity / lower bound | n/a |
| `htdemucs` | HTDemucs (vocals stem) | Offline baseline | MIT code + weights |
| `roformer` | BS/Mel-RoFormer via audio-separator | Offline quality candidate | Check per checkpoint; many community weights have no clear license |
| `deepfilternet` | DeepFilterNet 3 | Live-mode candidate | MIT / Apache-2.0 |

Still to verify and wrap: BandIt / BandIt v2 (DnR-trained dialogue stem),
LiteCASS, AV-CASS. A model is added to the registry only after its repository,
weights and license have been checked.

Acceptance rule: a model (or cascade) wins only if it improves speech
preservation and leakage on the benchmark set *and* wins the blind listening
test.

## Note: DeepFilterNet environment
The `deepfilternet` PyPI package pins `numpy<2`, which cannot share an
environment with Python 3.13 and the other candidates. It is not an extra of
this project. To evaluate it, use a separate Python 3.11 venv
(`uv venv -p 3.11 .venv-df` then `uv pip install -p .venv-df -e . deepfilternet torch torchaudio "numpy<2"`).
Live mode will likely use its ONNX export from Rust instead, which avoids the pin.
