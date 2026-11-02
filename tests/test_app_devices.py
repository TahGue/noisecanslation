import pytest

tk = pytest.importorskip("tkinter")

from voiceonly import app as appmod  # noqa: E402
from voiceonly.winaudio import Speaker  # noqa: E402

HEADPHONES = "Headphones (WH-1000XM4)"
WIRED = "Headphones (Realtek USB Audio)"
SINK = "Realtek Digital Output (Realtek USB Audio)"


@pytest.fixture
def make_app(monkeypatch, tmp_path):
    state = {"speakers": [HEADPHONES, WIRED, SINK], "default": HEADPHONES, "set": []}
    monkeypatch.setattr(appmod, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(appmod.winaudio, "speakers",
                        lambda: [Speaker(n, n) for n in state["speakers"]])
    monkeypatch.setattr(appmod.winaudio, "default_speaker",
                        lambda: Speaker(state["default"], state["default"]))
    monkeypatch.setattr(appmod.winaudio, "set_default_speaker",
                        lambda i: state["set"].append(i))
    monkeypatch.setattr(appmod.App, "_load_model", lambda self, key: None)
    monkeypatch.setattr(appmod.App, "_hotkey_loop", lambda self: None)
    created = []

    def factory():
        # Creating several Tk roots in one process occasionally fails Tcl init
        # ("invalid command name tcl_findLibrary"); the real app makes only one.
        for attempt in range(5):
            try:
                a = appmod.App()
                break
            except tk.TclError:
                if attempt == 4:
                    raise
        a.root.withdraw()
        a.cfg.update(listen="WH-1000XM4", capture="Realtek Digital")
        a.listen_var.set(HEADPHONES)
        a.capture_var.set(SINK)
        created.append(a)
        return a, state

    yield factory
    for a in created:
        try:
            a.root.destroy()
        except tk.TclError:
            pass  # the test closed it already


def test_lost_headphones_fall_back_to_a_real_output(make_app):
    a, state = make_app()
    state["speakers"] = [WIRED, SINK]
    state["default"] = WIRED
    a._check_devices()
    assert a.listen_var.get() == WIRED
    assert "disconnected" in a.status.cget("text")


def test_fallback_never_picks_the_capture_device(make_app):
    a, state = make_app()
    state["speakers"] = [SINK, WIRED]
    state["default"] = SINK  # Windows still points at the sink, e.g. mid-switch
    a._check_devices()
    assert a.listen_var.get() == WIRED


def test_filter_mode_drops_to_normal_when_headphones_vanish(make_app, monkeypatch):
    a, state = make_app()
    a.mode = "filter"
    modes = []
    monkeypatch.setattr(a, "set_mode", lambda m: (modes.append(m), setattr(a, "mode", m)))
    state["speakers"] = [WIRED, SINK]
    a._check_devices()
    assert modes == ["normal"]


def test_preferred_headphones_are_picked_again_when_back(make_app):
    a, state = make_app()
    state["speakers"] = [WIRED, SINK]
    state["default"] = WIRED
    a._check_devices()
    state["speakers"] = [HEADPHONES, WIRED, SINK]
    a._check_devices()
    assert a.listen_var.get() == HEADPHONES


def test_mix_modes_follow_the_model(make_app):
    a, _ = make_app()
    a.cfg["model"] = "htdemucs"
    a._refresh_keep()
    assert str(a.mix_buttons["no_music"].cget("state")) == "disabled"
    assert set(a.mixer_vars) == {"voice", "drums", "bass", "other"}
    a.cfg["model"] = "tiger"
    a._refresh_keep()
    assert str(a.mix_buttons["no_music"].cget("state")) == "normal"
    assert set(a.mixer_vars) == {"voice", "music", "effects"}


def test_custom_mixer_levels_are_kept_per_model(make_app):
    a, _ = make_app()
    a.cfg["model"] = "tiger"
    a._refresh_keep()
    a.mixer_vars["effects"].set(40)
    a._mixer_changed()
    assert a.cfg["mix"] == "custom"
    assert a._gains() == {"voice": 1.0, "music": 0.0, "effects": 0.4}
    assert a.btn_filter.cget("text") == "Custom mix"


def test_switching_model_resets_modes_it_cannot_do(make_app, monkeypatch):
    a, _ = make_app()
    loaded = []
    monkeypatch.setattr(a, "_load_model", loaded.append)
    a.cfg["model"], a.cfg["mix"] = "tiger", "no_music"
    a.set_model("htdemucs")
    assert loaded == ["htdemucs"] and a.cfg["mix"] == "voice"


def test_closing_restores_normal_mode_and_frees_the_model(make_app, monkeypatch):
    a, _ = make_app()

    class FakeModel:
        closed = False

        def close(self):
            FakeModel.closed = True

    a.separator = FakeModel()
    a.mode = "filter"
    modes = []
    monkeypatch.setattr(a, "set_mode", lambda m: (modes.append(m), setattr(a, "mode", m)))
    a.quit()
    assert modes == ["normal"] and FakeModel.closed and a.separator is None
    with pytest.raises(tk.TclError):
        a.root.winfo_exists()  # window destroyed
    a.quit()  # a second close (e.g. hotkey during shutdown) is harmless
