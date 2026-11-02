"""Read and change Windows' default playback device.

Uses IPolicyConfig, the COM interface the Windows sound panel itself uses. It is
undocumented but has kept the same layout since Windows 7.
"""

from __future__ import annotations

import ctypes
import uuid
from ctypes import POINTER, WINFUNCTYPE, byref, c_int, c_ulong, c_void_p, c_wchar_p
from dataclasses import dataclass

_CLSID_POLICY_CONFIG = "870af99c-171d-4f9e-af0d-e63df40c2bc9"
_IID_IPOLICY_CONFIG = "f8679f50-850a-41cf-9c72-430f290290c8"
_CLSCTX_ALL = 0x17
# vtable: 3 IUnknown slots, then GetMixFormat .. SetPropertyValue (10), then SetDefaultEndpoint.
_SET_DEFAULT_ENDPOINT = 13
_RELEASE = 2
_ROLES = (0, 1, 2)  # console, multimedia, communications


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", c_ulong), ("Data2", ctypes.c_ushort), ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, s: str) -> "_GUID":
        return cls.from_buffer_copy(uuid.UUID(s).bytes_le)


@dataclass
class Speaker:
    name: str
    id: str


def speakers() -> list[Speaker]:
    import soundcard as sc

    return [Speaker(s.name, s.id) for s in sc.all_speakers()]


def default_speaker() -> Speaker:
    import soundcard as sc

    s = sc.default_speaker()
    return Speaker(s.name, s.id)


def find_speaker(fragment: str) -> Speaker:
    for s in speakers():
        if fragment.lower() in s.name.lower():
            return s
    raise RuntimeError(f"no playback device matching '{fragment}'")


def set_default_speaker(device_id: str) -> None:
    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx(None, 0)  # already-initialised results are fine to ignore
    ptr = c_void_p()
    hr = ole32.CoCreateInstance(byref(_GUID.parse(_CLSID_POLICY_CONFIG)), None, _CLSCTX_ALL,
                                byref(_GUID.parse(_IID_IPOLICY_CONFIG)), byref(ptr))
    if hr != 0 or not ptr:
        raise OSError(f"PolicyConfig unavailable (HRESULT {hr & 0xFFFFFFFF:#010x})")
    vtbl = ctypes.cast(ptr, POINTER(POINTER(c_void_p))).contents
    try:
        set_default = WINFUNCTYPE(ctypes.HRESULT, c_void_p, c_wchar_p, c_int)(
            vtbl[_SET_DEFAULT_ENDPOINT])
        for role in _ROLES:
            set_default(ptr, device_id, role)
    finally:
        WINFUNCTYPE(c_ulong, c_void_p)(vtbl[_RELEASE])(ptr)
