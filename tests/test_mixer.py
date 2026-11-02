import numpy as np
import pytest

from voiceonly.live import LiveConfig, LiveEngine, StreamingProcessor
from voiceonly.mixer import MODES, available, gains_for
from voiceonly.separators.base import ModelInfo, SpeechSeparator

HTDEMUCS = ["voice", "drums", "bass", "other"]
CINEMATIC = ["voice", "music", "effects"]
ROFORMER = ["voice", "background"]


def test_voice_only_keeps_just_the_voice():
    assert gains_for("voice", CINEMATIC) == {"voice": 1.0, "music": 0.0, "effects": 0.0}
    assert gains_for("voice", HTDEMUCS) == {"voice": 1.0, "drums": 0.0, "bass": 0.0, "other": 0.0}


def test_music_and_effects_modes_need_a_model_that_splits_them():
    assert available("no_music", CINEMATIC) and available("no_effects", CINEMATIC)
    for stems in (HTDEMUCS, ROFORMER):
        assert not available("no_music", stems) and not available("no_effects", stems)
    assert gains_for("no_music", CINEMATIC) == {"voice": 1.0, "music": 0.0, "effects": 1.0}


@pytest.mark.parametrize("mode", list(MODES))
def test_every_mode_works_where_available(mode):
    for stems in (HTDEMUCS, CINEMATIC, ROFORMER):
        if available(mode, stems):
            assert set(gains_for(mode, stems)) == set(stems)


def test_karaoke_and_boost():
    assert gains_for("karaoke", ROFORMER) == {"voice": 0.0, "background": 1.0}
    assert gains_for("boost", CINEMATIC) == {"voice": 1.0, "music": 0.25, "effects": 0.25}


class ThreeStems(SpeechSeparator):
    """Splits a known mix back into its parts: stem i = part i of the input."""

    info = ModelInfo(key="three", name="t", version="1", sample_rate=1000, channels=1,
                     chunk_seconds=1, overlap_seconds=0, live_window_seconds=1.0,
                     stems=CINEMATIC)

    def separate(self, chunk):
        return chunk * 0.5

    def separate_stems(self, chunk):
        return np.stack([chunk * 0.5, chunk * 0.3, chunk * 0.2])


def test_processor_carries_every_stem_through_the_crossfade():
    proc = StreamingProcessor(ThreeStems(), LiveConfig(hop_s=0.1, lookahead_s=0.1,
                                                       crossfade_s=0.05), stems=True)
    ones = np.ones((100, 1), np.float32)
    for _ in range(12):
        out = proc.step(ones)
    assert out.shape == (3, 100, 1)
    np.testing.assert_allclose(out[:, 50, 0], [0.5, 0.3, 0.2], atol=1e-6)


def test_engine_gains_mix_stems_and_can_change_live():
    eng = LiveEngine(ThreeStems(), LiveConfig(hop_s=0.1, lookahead_s=0.1, crossfade_s=0.0),
                     jitter_s=0.0, gains=gains_for("no_music", CINEMATIC))
    np.testing.assert_allclose(eng.gains, [1.0, 0.0, 1.0])
    eng.set_gains({"voice": 0.0, "music": 1.0})
    np.testing.assert_allclose(eng.gains, [0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        eng.set_gains([1.0, 0.0])
