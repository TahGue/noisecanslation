"""Reference-based speech extraction metrics.

All metrics work on mono (channel-averaged) signals at a common sample rate.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-10


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


def _align(*signals: np.ndarray) -> list[np.ndarray]:
    n = min(s.shape[0] for s in signals)
    return [_mono(s[:n]) for s in signals]


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    est, ref = _align(estimate, reference)
    est, ref = est - est.mean(), ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.dot(target, target) + EPS) / (np.dot(noise, noise) + EPS)))


def decompose(estimate: np.ndarray, speech: np.ndarray, background: np.ndarray) -> dict:
    """Least-squares fit estimate ~= a*speech + c*background + residual.

    speech_gain_db: 20*log10(a). 0 dB means the reference speech fully survives;
        negative means speech was removed.
    leakage_db: background energy in the output relative to speech energy in it.
    artifact_db: energy explained by neither, relative to speech energy.
    """
    est, s, b = _align(estimate, speech, background)
    basis = np.stack([s, b], axis=1)
    (a, c), *_ = np.linalg.lstsq(basis, est, rcond=None)
    speech_part, bg_part = a * s, c * b
    resid = est - speech_part - bg_part
    e_s = np.dot(speech_part, speech_part) + EPS
    return {
        "speech_gain_db": float(20 * np.log10(abs(a) + EPS)),
        "leakage_db": float(10 * np.log10((np.dot(bg_part, bg_part) + EPS) / e_s)),
        "artifact_db": float(10 * np.log10((np.dot(resid, resid) + EPS) / e_s)),
    }


def stoi(estimate: np.ndarray, reference: np.ndarray, sample_rate: int) -> float | None:
    try:
        from pystoi import stoi as _stoi
    except ImportError:
        return None
    est, ref = _align(estimate, reference)
    return float(_stoi(ref, est, sample_rate, extended=False))


def evaluate(estimate: np.ndarray, mixture: np.ndarray, speech: np.ndarray,
             background: np.ndarray, sample_rate: int) -> dict:
    sdr = si_sdr(estimate, speech)
    sdr_mix = si_sdr(mixture, speech)
    result = {"si_sdr": sdr, "si_sdr_mix": sdr_mix, "si_sdr_i": sdr - sdr_mix}
    result.update(decompose(estimate, speech, background))
    result["stoi"] = stoi(estimate, speech, sample_rate)
    return result
