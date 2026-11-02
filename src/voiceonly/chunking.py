"""Overlap-add chunk processing.

Chunks of `chunk` frames advance by `chunk - overlap`. Neighbouring chunks are
crossfaded with complementary linear ramps over the overlap, so an identity
processor reconstructs the input exactly. Output is emitted as soon as a region
can no longer be touched by a later chunk, so memory is bounded by one chunk
plus one overlap regardless of input length.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

Processor = Callable[[np.ndarray], np.ndarray]


def chunk_starts(total: int, chunk: int, overlap: int) -> list[int]:
    if chunk <= 0 or not 0 <= overlap < chunk:
        raise ValueError("need chunk > 0 and 0 <= overlap < chunk")
    if total <= 0:
        return []
    hop = chunk - overlap
    starts = [0]
    while starts[-1] + chunk < total:
        starts.append(starts[-1] + hop)
    return starts


def crossfade_window(n: int, overlap: int, first: bool, last: bool) -> np.ndarray:
    w = np.ones(n, dtype=np.float64)
    if overlap == 0:
        return w
    # Half-sample offset keeps every weight > 0 and makes up+down ramps sum to 1.
    ramp = (np.arange(overlap) + 0.5) / overlap
    if not first:
        k = min(overlap, n)
        w[:k] = ramp[:k]
    if not last:
        w[n - overlap:] *= ramp[::-1]
    return w


def process_chunked(
    read: Callable[[int, int], np.ndarray],
    total: int,
    process: Processor,
    chunk: int,
    overlap: int,
) -> Iterator[np.ndarray]:
    """Yield reconstructed output blocks in order.

    `read(start, n)` returns up to n frames shaped (frames, channels).
    `process` receives exactly `chunk` frames (zero-padded at the end of input)
    and must return the same number of frames.
    """
    starts = chunk_starts(total, chunk, overlap)
    acc: np.ndarray | None = None
    weight = np.zeros(0)
    acc_start = 0

    for i, start in enumerate(starts):
        x = read(start, chunk)
        n = x.shape[0]
        if n < chunk:
            x = np.pad(x, ((0, chunk - n), (0, 0)))
        y = np.asarray(process(x), dtype=np.float64)
        if y.shape[0] != chunk:
            raise RuntimeError(f"processor returned {y.shape[0]} frames, expected {chunk}")
        y = y[:n]
        last = i == len(starts) - 1
        w = crossfade_window(n, overlap, first=i == 0, last=last)

        if acc is None:
            acc = np.zeros((0, y.shape[1]))
        end = start - acc_start + n
        if end > acc.shape[0]:
            acc = np.pad(acc, ((0, end - acc.shape[0]), (0, 0)))
            weight = np.pad(weight, (0, end - weight.shape[0]))
        off = start - acc_start
        acc[off:end] += y * w[:, None]
        weight[off:end] += w

        final = total if last else starts[i + 1]
        k = final - acc_start
        if k > 0:
            yield (acc[:k] / weight[:k, None]).astype(np.float32)
            acc, weight, acc_start = acc[k:], weight[k:], final
