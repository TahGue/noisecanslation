import numpy as np
import pytest

from voiceonly.live import LiveConfig, LiveEngine, StreamingProcessor, simulate
from voiceonly.separators import create
from voiceonly.separators.base import ModelInfo, SpeechSeparator


class Gain(SpeechSeparator):
    """Window-dependent processor: output depends on the whole window, like a real model."""

    info = ModelInfo(key="gain", name="gain", version="1", sample_rate=1000, channels=1,
                     chunk_seconds=1, overlap_seconds=0, live_window_seconds=1.0)

    def separate(self, chunk):
        return chunk * 0.5


def test_live_passthrough_is_exact_after_delay():
    x = np.random.default_rng(0).uniform(-1, 1, (48_000 * 3 + 123, 2)).astype(np.float32)
    y, stats = simulate(x, create("passthrough"), LiveConfig(hop_s=0.1, lookahead_s=0.2))
    assert y.shape == x.shape
    np.testing.assert_allclose(y, x, atol=1e-6)
    assert stats["algorithmic_latency_s"] == pytest.approx(0.1 + 0.2 + 0.05)


@pytest.mark.parametrize("hop,look,fade", [(0.1, 0.0, 0.0), (0.05, 0.3, 0.02), (0.25, 0.25, 0.1)])
def test_live_gain_has_no_seams(hop, look, fade):
    x = np.random.default_rng(1).uniform(-1, 1, (5000, 1)).astype(np.float32)
    y, _ = simulate(x, Gain(), LiveConfig(hop_s=hop, lookahead_s=look, crossfade_s=fade))
    np.testing.assert_allclose(y, 0.5 * x, atol=1e-6)


def test_bypass_switch_keeps_timing():
    proc = StreamingProcessor(Gain(), LiveConfig(hop_s=0.1, lookahead_s=0.1, crossfade_s=0.0))
    ones = np.ones((100, 1), np.float32)
    for _ in range(5):
        proc.step(ones)
    assert proc.step(ones)[0, 0] == pytest.approx(0.5)
    proc.bypass = True
    assert proc.step(ones)[0, 0] == pytest.approx(1.0)


def test_window_must_fit_hop_and_lookahead():
    with pytest.raises(ValueError):
        StreamingProcessor(Gain(), LiveConfig(hop_s=0.6, lookahead_s=0.6))


def test_engine_queues_without_devices():
    eng = LiveEngine(Gain(), LiveConfig(hop_s=0.1, lookahead_s=0.1, crossfade_s=0.0), jitter_s=0.0)
    out = np.zeros((50, 1), np.float32)
    eng._on_output(out, 50, None, None)  # nothing processed yet: silence, not an underrun
    assert eng.stats.underruns == 0

    eng.running = True
    import threading
    t = threading.Thread(target=eng._worker, daemon=True)
    t.start()
    for _ in range(10):
        eng._on_input(np.ones((50, 1), np.float32), 50, None, None)
    import time
    deadline = time.time() + 2
    while eng.stats.steps < 5 and time.time() < deadline:
        time.sleep(0.01)
    eng.running = False
    t.join()
    assert eng.stats.steps == 5
    eng._on_output(out, 50, None, None)
    assert eng.stats.underruns == 0
