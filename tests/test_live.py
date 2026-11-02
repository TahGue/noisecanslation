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


def test_engine_downmixes_for_a_mono_output_device():
    import threading
    import time

    eng = LiveEngine(create("passthrough"), LiveConfig(hop_s=0.05, lookahead_s=0.05,
                                                       crossfade_s=0.0), jitter_s=0.01)
    eng.out_channels = 1  # what run() picks for a hands-free Bluetooth headset
    eng.running = True
    t = threading.Thread(target=eng._worker, daemon=True)
    t.start()
    for _ in range(20):
        eng._on_input(np.full((480, 2), 0.5, np.float32), 480, None, None)
    deadline = time.time() + 2
    while eng.stats.steps < 3 and time.time() < deadline:
        time.sleep(0.01)
    eng.running = False
    t.join()
    out = np.zeros((480, 1), np.float32)
    eng._on_output(out, 480, None, None)
    assert eng.stats.underruns == 0


def _engine(**kw):
    return LiveEngine(Gain(), LiveConfig(hop_s=0.1, lookahead_s=0.1, crossfade_s=0.0),
                      jitter_s=0.1, **kw)


def test_adaptive_buffer_shrinks_unused_slack():
    eng = _engine()
    start = eng.jitter_target
    for _ in range(40):
        eng._adapt_jitter(int(0.08 * 1000))  # buffer never below 80 ms before a block
    # slack = 80 - 10 ms margin -> target drops by half of it
    assert eng.jitter_target == pytest.approx(start - 0.5 * (80 - 10))


def test_adaptive_buffer_grows_after_a_gap_and_respects_limits():
    eng = _engine()
    eng.jitter_target = 50.0
    eng.gap_seen = True
    eng._adapt_jitter(0)
    assert eng.jitter_target == pytest.approx(50 * 1.5 + 10)
    for _ in range(400):
        eng._adapt_jitter(10_000)  # huge slack every time
    assert eng.jitter_target == eng.min_jitter


def test_fixed_buffer_when_adaptive_is_off():
    eng = _engine(adaptive=False)
    for _ in range(80):
        eng._adapt_jitter(500)
    assert eng.jitter_target == eng.jitter


def test_silent_blocks_absorb_buffer_excess_quickly():
    eng = _engine()
    eng.jitter_target = 20.0           # frames at 1 kHz: want 20 ms of buffer
    eng.trough_avg = 80.0              # currently 80 ms
    quiet = np.zeros((100, 1), np.float32)
    out = eng._rate_adjust(quiet, 80)
    assert out.shape[0] == 100 - 50    # up to half the block removed at once
    loud = np.full((100, 1), 0.5, np.float32)
    eng.trough_avg = 80.0
    assert eng._rate_adjust(loud, 80).shape[0] == 99  # speech: at most 1% per block
