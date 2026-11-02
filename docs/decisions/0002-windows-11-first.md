# 0002 — Windows 11 is the first target OS

Status: accepted (2026-11-02)

The development machine runs Windows 11 with an RTX 4090. The bake-off targets
native Windows. Live capture (later) will use WASAPI loopback. Ubuntu support is
revisited when the Rust core exists.

ffmpeg is optional: WAV/FLAC/OGG load through libsndfile; video and other
containers require `ffmpeg` on PATH (`winget install Gyan.FFmpeg`).
