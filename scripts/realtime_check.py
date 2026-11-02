"""Real-time glitch and delay check for the app's models (engine only, no quality).

    uv run python scripts/realtime_check.py --label before
    uv run python scripts/realtime_check.py -m tiger --runs 3 --label after

Plays a 29 s test clip through the real engine to --output (an output nobody hears)
and records audible gaps, drift drops and delay. Appends a JSON line per run to
runs/realtime_check.jsonl so before/after changes can be compared.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_benchmark import LIVE_ITEMS, ROOT, load_items, realtime  # noqa: E402

from voiceonly import separators  # noqa: E402
from voiceonly.live import presets_for  # noqa: E402

APP_MODELS = ["htdemucs", "tiger", "bs-roformer-live"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", action="append")
    ap.add_argument("--preset", action="append")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--output", default="Realtek")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    item = next(i for i in load_items(ROOT / "data/testset") if i["id"] == LIVE_ITEMS[0])
    log = ROOT / "runs" / "realtime_check.jsonl"
    print(f"{'model':18s} {'preset':9s} {'run':>3s} {'gaps':>5s} {'drift':>5s} "
          f"{'dropped':>7s} {'delay':>6s} {'buffer':>14s}  max step")
    for key in args.model or APP_MODELS:
        sep = separators.create(key)
        sep.initialize()
        for preset in args.preset or [p for p in ("balanced", "fast") if p in presets_for(key)]:
            for run in range(args.runs):
                r = realtime(sep, item, preset, args.output)
                r.update(model=key, run=run, label=args.label)
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r) + "\n")
                print(f"{key:18s} {preset:9s} {run:3d} {r["underruns"]:5d} {r["drift_drops"]:5d} "
                      f"{r['dropped']:7d} {r['delay_s']:6.3f} {r['buffer_start_ms']:5.0f}->{r['buffer_ms']:4.0f} ms"
                      f"  {r['step_ms_max_recent']} ms",
                      flush=True)
        sep.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
