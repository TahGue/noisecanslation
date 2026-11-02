from __future__ import annotations

import argparse
import sys

from . import audio_io, separators


def cmd_models(args) -> int:
    print(f"ffmpeg: {'found' if audio_io.ffmpeg_available() else 'not found (WAV/FLAC/OGG only)'}\n")
    for key, cls in separators.REGISTRY.items():
        i = cls.info
        missing = separators.missing_requirements(key)
        status = "ready" if not missing else separators.install_hint(key)
        print(f"{key:14s} {i.name:34s} {i.sample_rate} Hz  {i.channels}ch  "
              f"license: {i.license:22s} [{status}]")
    return 0


def cmd_separate(args) -> int:
    from .pipeline import run_file

    sep = separators.create(args.model, device=args.device)

    def progress(frac: float) -> None:
        print(f"\r  separating {frac:6.1%}", end="", file=sys.stderr, flush=True)

    result = run_file(args.input, args.out, sep, on_progress=progress)
    print(file=sys.stderr)
    qc, bench = result["metadata"]["quality"], result["benchmark"]
    print(f"Wrote {args.out}/speech.wav and speech.flac")
    print(f"  audio {bench['audio_s']:.1f} s, processed in {bench['total_s']:.1f} s "
          f"(RTF {bench['rtf']:.3f})")
    print("  quality checks: " + ("passed" if qc["ok"] else "; ".join(qc["issues"])))
    return 0 if qc["ok"] else 2


def cmd_bench(args) -> int:
    from . import bench

    result = bench.run(args.dataset, args.model, args.out, device=args.device, seed=args.seed)
    print(f"\n{'model':14s} {'SI-SDR':>8s} {'SI-SDRi':>8s} {'speech':>8s} {'leak':>8s} "
          f"{'STOI':>6s} {'RTF':>7s}  QC fails")
    for key, s in result["summary"].items():
        stoi = f"{s['stoi']:.3f}" if s["stoi"] is not None else "  n/a"
        print(f"{key:14s} {s['si_sdr']:8.2f} {s['si_sdr_i']:8.2f} {s['speech_gain_db']:8.2f} "
              f"{s['leakage_db']:8.2f} {stoi:>6s} {s['rtf']:7.3f}  {s['qc_failures']}")
    for key, err in result.get("failed", {}).items():
        print(f"{key:14s} FAILED: {err}")
    print(f"\nResults in {args.out}; blinded samples in {args.out}/listening (answers: key.json)")
    return 0


def _live_config(args):
    """The chosen preset, with any explicitly given timing flag overriding it."""
    from .live import PRESETS, LiveConfig

    p = PRESETS[args.preset]
    pick = lambda flag, default: default if flag is None else flag  # noqa: E731
    return LiveConfig(hop_s=pick(args.hop, p.hop_s),
                      lookahead_s=pick(args.lookahead, p.lookahead_s),
                      window_s=pick(args.window, p.window_s),
                      crossfade_s=pick(args.crossfade, p.crossfade_s))


def _jitter(args) -> float:
    from .live import PRESETS

    return PRESETS[args.preset].jitter_s if args.jitter is None else args.jitter


def cmd_devices(args) -> int:
    import sounddevice as sd

    apis = sd.query_hostapis()
    print(f"{'#':>3s}  {'API':18s} {'in':>3s} {'out':>3s}  name")
    for i, d in enumerate(sd.query_devices()):
        api = apis[d["hostapi"]]["name"]
        if args.all or "WASAPI" in api:
            print(f"{i:3d}  {api:18s} {d['max_input_channels']:3d} {d['max_output_channels']:3d}  "
                  f"{d['name']}")
    in_d, out_d = sd.default.device
    print(f"\ndefault input #{in_d}, default output #{out_d}")
    print("\nLoopback sources (driver-free capture, use as -i loopback:<name>):")
    try:
        import soundcard as sc

        for m in sc.all_microphones(include_loopback=True):
            if m.isloopback:
                print(f"  loopback:{m.name}")
    except Exception as e:
        print(f"  unavailable: {e}")
    if not any("cable" in d["name"].lower() for d in sd.query_devices()):
        print("\nVB-CABLE not found; use a loopback source instead (see LIVE_SETUP.md).")
    return 0


def cmd_live(args) -> int:
    from .live import LiveEngine, find_device

    import sounddevice as sd

    in_dev = None if args.input_file else find_device(args.input, "input")
    out_dev = find_device(args.output, "output")
    out_name = sd.query_devices(out_dev if out_dev is not None else sd.default.device[1])["name"]
    from .live import LoopbackCapture

    if isinstance(in_dev, LoopbackCapture) and in_dev.name.split(" (")[0] in out_name:
        raise RuntimeError(f"capturing and playing on '{in_dev.name}' would loop forever. "
                           "Pick a different -o device.")
    if "cable" in out_name.lower():
        raise RuntimeError(
            f"playback device is '{out_name}', which would feed VoiceOnly back into itself. "
            "Pass your speakers/headphones with -o (see: voiceonly devices).")
    sep = separators.create(args.model, device=args.device)
    engine = LiveEngine(sep, _live_config(args), in_dev, out_dev, jitter_s=_jitter(args))
    print(f"Loading {sep.info.name}...", file=sys.stderr)
    sep.initialize()
    print(f"Live: {args.input_file or args.input or 'default input'} -> {args.output or 'default output'}, "
          f"delay ~{engine.latency_s:.2f} s. Press Enter to toggle voice-only/bypass, "
          f"Ctrl+C to stop.", file=sys.stderr)

    def status(e) -> None:
        ms = e.stats.step_ms
        mean = sum(ms) / len(ms) if ms else 0.0
        mode = "BYPASS    " if e.proc.bypass else "VOICE ONLY"
        print(f"\r  [{mode}] step {mean:6.1f} ms / hop {e.proc.hop / e.proc.sr * 1000:.0f} ms  "
              f"underruns {e.stats.underruns}  dropped {e.stats.dropped_blocks}   ",
              end="", file=sys.stderr, flush=True)

    def toggle() -> None:
        while True:
            try:
                input()
            except EOFError:
                return
            engine.proc.bypass = not engine.proc.bypass

    input_audio = None
    if args.input_file:
        audio, src = audio_io.load(args.input_file)
        input_audio = audio_io.remix_channels(
            audio_io.resample(audio, src.sample_rate, sep.info.sample_rate), sep.info.channels)
        input_audio *= 10 ** (args.gain_db / 20)

    if args.record:
        engine.recorded = []

    import threading
    threading.Thread(target=toggle, daemon=True).start()
    try:
        engine.run(on_status=status, input_audio=input_audio, duration_s=args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        sep.close()
        print(file=sys.stderr)
    if args.record and engine.recorded:
        import numpy as np

        audio_io.save(args.record, np.concatenate(engine.recorded), sep.info.sample_rate)
        print(f"recorded engine output to {args.record}")
    s = engine.stats
    ms = sorted(s.step_ms)
    print(f"steps {s.steps}, underruns {s.underruns}, dropped {s.dropped_blocks}, "
          f"drift drops {s.drift_drops}, "
          f"recent step p99 {ms[int(0.99 * (len(ms) - 1))] if ms else 0:.1f} ms")
    return 0 if s.underruns == 0 and s.dropped_blocks == 0 else 2


def cmd_live_sim(args) -> int:
    from .live import simulate

    audio, src = audio_io.load(args.input)
    sep = separators.create(args.model, device=args.device)
    info = sep.info
    x = audio_io.remix_channels(audio_io.resample(audio, src.sample_rate, info.sample_rate),
                                info.channels)
    sep.initialize()
    try:
        y, stats = simulate(x, sep, _live_config(args))
        stats.update(sep.runtime_info())
    finally:
        sep.close()
    y = audio_io.remix_channels(audio_io.resample(y, info.sample_rate, src.sample_rate),
                                min(audio.shape[1], 2))
    audio_io.save(args.out, y, src.sample_rate)
    print(f"Wrote {args.out}")
    print(f"  delay {stats['algorithmic_latency_s']:.2f} s (+ device buffers)")
    print(f"  compute per {stats['hop_s'] * 1000:.0f} ms hop: mean {stats['step_ms_mean']:.1f} ms, "
          f"p99 {stats['step_ms_p99']:.1f} ms -> {stats['realtime_headroom']:.1f}x faster than "
          "real time")
    if stats["realtime_headroom"] < 1.2:
        print("  WARNING: too slow for live use with these settings; raise --hop or use a lighter model")
    return 0


def cmd_synthetic(args) -> int:
    from .synthetic import write_set

    paths = write_set(args.out, items=args.items, seconds=args.seconds, seed=args.seed)
    print(f"Wrote {len(paths)} synthetic items to {args.out} (plumbing tests only, not real speech)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="voiceonly", description="Offline speech extraction bake-off")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="list candidate models").set_defaults(fn=cmd_models)

    s = sub.add_parser("separate", help="extract speech from one file")
    s.add_argument("input")
    s.add_argument("-m", "--model", default="htdemucs")
    s.add_argument("-o", "--out", default="output")
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=cmd_separate)

    b = sub.add_parser("bench", help="score models against reference stems")
    b.add_argument("dataset")
    b.add_argument("-m", "--model", action="append", required=True)
    b.add_argument("-o", "--out", default="runs/latest")
    b.add_argument("--device", default="auto")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_bench)

    dv = sub.add_parser("devices", help="list audio devices")
    dv.add_argument("--all", action="store_true", help="include non-WASAPI devices")
    dv.set_defaults(fn=cmd_devices)

    def live_args(sp) -> None:
        from .live import DEFAULT_PRESET, PRESETS

        sp.add_argument("-m", "--model", default="htdemucs")
        sp.add_argument("--device", default="auto")
        sp.add_argument("--preset", choices=list(PRESETS), default=DEFAULT_PRESET,
                        help="latency/quality trade-off; the flags below override it")
        sp.add_argument("--hop", type=float, default=None, help="seconds processed per step")
        sp.add_argument("--lookahead", type=float, default=None, help="future audio the model hears")
        sp.add_argument("--window", type=float, default=None, help="model window in seconds")
        sp.add_argument("--crossfade", type=float, default=None)
        sp.add_argument("--jitter", type=float, default=None, help="safety buffer in seconds")

    lv = sub.add_parser("live", help="filter live audio to voices only")
    lv.add_argument("-i", "--input", default="CABLE Output", help="capture device (name or #)")
    lv.add_argument("-o", "--output", default=None, help="playback device (name or #)")
    lv.add_argument("--input-file", help="play a file through the live engine instead of a device")
    lv.add_argument("--gain-db", type=float, default=-12.0, help="level of --input-file")
    lv.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    lv.add_argument("--record", help="also save the filtered output to this WAV file")
    live_args(lv)
    lv.set_defaults(fn=cmd_live)

    ls = sub.add_parser("live-sim", help="run the live engine over a file and time it")
    ls.add_argument("input")
    ls.add_argument("-o", "--out", default="output/live_sim.wav")
    live_args(ls)
    ls.set_defaults(fn=cmd_live_sim)

    g = sub.add_parser("make-synthetic", help="write a synthetic test set")
    g.add_argument("out")
    g.add_argument("--items", type=int, default=5)
    g.add_argument("--seconds", type=float, default=12.0)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_synthetic)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (separators.SeparationError, RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
