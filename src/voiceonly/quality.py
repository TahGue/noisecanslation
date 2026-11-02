"""Output integrity checks. These flag problems; they never alter audio."""

from __future__ import annotations

import numpy as np


def check(output: np.ndarray, expected_frames: int, sample_rate: int,
          reference: np.ndarray | None = None) -> dict:
    issues: list[str] = []
    frames = int(output.shape[0])
    if frames != expected_frames:
        issues.append(f"duration mismatch: {frames} frames vs {expected_frames} expected")

    finite = bool(np.isfinite(output).all())
    if not finite:
        issues.append("output contains NaN or Inf")
    out = np.nan_to_num(output)

    peak = float(np.abs(out).max()) if out.size else 0.0
    clipped = float(np.mean(np.abs(out) >= 0.999)) if out.size else 0.0
    if clipped > 1e-4:
        issues.append(f"{clipped:.4%} of samples at or above full scale")

    # Discontinuity: sample-to-sample jumps far beyond what the input contains.
    jumps = 0
    if reference is not None and frames > 1 and reference.shape[0] > 1:
        ref_d = np.abs(np.diff(reference.mean(axis=1)))
        out_d = np.abs(np.diff(out.mean(axis=1)))
        limit = max(4 * float(np.percentile(ref_d, 99.99)), 1e-3)
        jumps = int(np.sum(out_d > limit))
        if jumps:
            issues.append(f"{jumps} sample jumps larger than anything in the input")

    # Long dropouts: >= 2 s of digital silence where the input had signal.
    dropouts = 0
    if reference is not None and frames:
        win = sample_rate // 10
        n = min(frames, reference.shape[0]) // win
        if n:
            o = np.abs(out[: n * win]).reshape(n, win, -1).max(axis=(1, 2))
            r = np.abs(reference[: n * win]).reshape(n, win, -1).max(axis=(1, 2))
            silent = (o < 1e-6) & (r > 1e-3)
            run = best = 0
            for s in silent:
                run = run + 1 if s else 0
                best = max(best, run)
            dropouts = best
            if best >= 20:
                issues.append(f"{best / 10:.1f} s of output silence where input has signal")

    return {
        "ok": not issues,
        "issues": issues,
        "frames": frames,
        "peak": round(peak, 6),
        "clipped_fraction": clipped,
        "discontinuities": jumps,
        "longest_dropout_s": dropouts / 10,
    }
