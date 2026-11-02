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
    monkeypatch.setattr(appmod.App, "_load_model", lambda self: None)
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
        a.root.destroy()


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


def test_voice_mode_drops_to_normal_when_headphones_vanish(make_app, monkeypatch):
    a, state = make_app()
    a.mode = "voice"
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
