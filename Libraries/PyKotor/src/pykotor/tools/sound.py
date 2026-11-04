from __future__ import annotations

import struct


def deobfuscate_audio(
    data: bytes,
) -> bytes:  # sourcery skip: remove-redundant-slice-index
    """Remove KotOR's SFX, voice-over and MP3-in-WAV wrappers.

    Standard WAV and raw MP3 data remain unchanged. SFX wrappers contain a
    RIFF payload after 470 bytes; voice-over wrappers contain one after 20.
    The short RIFF wrapper around MP3 data occupies 58 bytes.
    """
    b0x4 = struct.unpack("<I", data[0:4])[0]
    b4x8 = struct.unpack("<I", data[4:8])[0]
    b16x20 = struct.unpack("<I", data[16:20])[0]
    if b0x4 == 0xC460F3FF and data[470:474] == b"RIFF":
        data = data[470:]
    elif b0x4 == 0x46464952 and data[20:24] == b"RIFF":
        data = data[20:]
    elif b0x4 == 0x46464952 and b4x8 == 50 and b16x20 == 18:
        data = data[58:]
    return data
