# 0001 — Python-first model bake-off before any Rust

Status: accepted (2026-11-02)

## Decision
Phases 1 and 2 of the architecture plan are merged into one Python project
(`voiceonly`). It decodes media, runs candidate separators through one
interface, reconstructs the timeline with overlap-add, runs quality checks and
scores results against reference stems.

The Rust workspace starts only after a model has been selected from this
benchmark, and starts small (audio-core, decode, inference, cli).

## Why
Model choice dominates final quality. Building 13 crates before knowing the
model's sample rate, context length and runtime would mean redesigning them.
The Python pipeline also becomes the reference implementation that the Rust
port must match sample-for-sample.
