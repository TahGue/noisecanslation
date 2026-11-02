"""What to keep: mix modes expressed as gains per stem category.

Models name their stems differently (HTDemucs: drums/bass/other, MRX: music/effects,
RoFormer: background). Each stem maps to a category, and a mode sets a gain per
category. A mode that needs to tell music from effects is only offered when the
model has separate music and effects stems.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CATEGORY = {
    "voice": "voice", "speech": "voice",
    "music": "music", "drums": "music", "bass": "music",
    "effects": "effects", "sfx": "effects",
    # Stems that mix music and effects together.
    "other": "background", "background": "background",
}


@dataclass(frozen=True)
class Mode:
    label: str
    gains: dict[str, float]  # category -> gain; "*" = every category not listed
    needs: frozenset = field(default_factory=frozenset)  # categories the model must have
    hint: str = ""


MODES: dict[str, Mode] = {
    "voice": Mode("Voice only", {"voice": 1.0}, hint="Only dialogue and speech"),
    "boost": Mode("Voice boost", {"voice": 1.0, "*": 0.25},
                  hint="Speech up front, background kept quietly"),
    "no_music": Mode("No music", {"voice": 1.0, "effects": 1.0},
                     frozenset({"music", "effects"}), "Keep voices and sound effects"),
    "no_effects": Mode("No effects", {"voice": 1.0, "music": 1.0},
                       frozenset({"music", "effects"}), "Keep voices and music"),
    "karaoke": Mode("No voice", {"voice": 0.0, "*": 1.0}, hint="Everything except voices"),
}
DEFAULT_MODE = "voice"


def categories(stems: list[str]) -> set[str]:
    return {CATEGORY.get(s, "background") for s in stems}


def available(mode: str, stems: list[str]) -> bool:
    return MODES[mode].needs <= categories(stems)


def gains_for(mode: str, stems: list[str]) -> dict[str, float]:
    """Gain per stem name for a mode."""
    m = MODES[mode]
    default = m.gains.get("*", 0.0)
    return {s: m.gains.get(CATEGORY.get(s, "background"), default) for s in stems}
