"""Ordered TXI documents and the settings consumed by the game.

TXI is a byte-oriented format, including counted binary lists. Latin-1 is used
as a reversible byte-to-character mapping, not as an encoding guess. Unknown
commands and original formatting remain in the document, not in a regenerated
settings dictionary.
"""
from __future__ import annotations

import re
import difflib
import struct

from dataclasses import dataclass, field
from typing import Iterable

from pykotor.common.misc import Game

_INT = re.compile(rb"[ \t\r\v\f]*([+-]?[0-9]+)")
_FLOAT = re.compile(rb"[ \t\r\v\f]*([+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|inf(?:inity)?|nan))", re.I)
_LINE_END = re.compile(rb"\r\n|\r|\n")
_FIELD = re.compile(rb"[ \t\r\v\f]*([^ \t\r\v\f]+)(?:[ \t\r\v\f]+(.*))?", re.S)
_LISTS = {"upperleftcoords": 3, "lowerrightcoords": 3, "channelscale": 1, "channeltranslate": 1}
_SHORTS = {"filerange", "defaultwidth", "defaultheight", "downsamplemax", "downsamplemin", "clamp", "numx", "numy"}
_BOOLS = {"mipmap", "filter", "maptexelstopixels", "isdiffusebumpmap", "isspecularbumpmap", "cube", "temporary", "useglobalalpha", "isenvironmentmapped"}
_FLOATS = {"gamma", "alphamean", "bumpmapscaling", "bumpintensity", "envmapalpha", "diffusebumpintensity", "specularbumpintensity"}
_FONT_FLOATS = {"fontheight", "baselineheight", "texturewidth", "spacingr", "spacingb"}
_CONTROLLERS = {"water", "life", "perlin", "arturo", "wave", "cycle", "random", "ringtexdistort", "dirty", "dirty2", "dirty3"}


def _integer(value: bytes) -> int:
    token = value.split()[0].lower() if value.split() else b""
    if token in (b"true", b"false"):
        return int(token == b"true")
    match = _INT.match(value)
    if match is None:
        raise ValueError(f"Invalid TXI integer: {value!r}")
    return int(match[1])


def _floats(value: bytes, count: int) -> tuple[float, ...]:
    values = []
    position = 0
    for _ in range(count):
        match = _FLOAT.match(value, position)
        if match is None:
            raise ValueError(f"Invalid TXI float/vector: {value!r}")
        # Native consumers store single-precision values.
        values.append(struct.unpack("<f", struct.pack("<f", float(match[1])))[0])
        position = match.end()
    return tuple(values)


def _line(data: bytes, start: int) -> tuple[bytes, int]:
    ending = _LINE_END.search(data, start)
    end = ending.end() if ending else len(data)
    return data[start:end], end


@dataclass(frozen=True)
class TXIRecord:
    """A source occurrence, including its complete list block when applicable."""

    start: int
    end: int
    line_end: int
    keyword: str
    operand: bytes
    values: tuple = ()
    binary: bool = False
    counted: bool = False


@dataclass
class TXISettings:
    """Effective state after executing the document in source order."""

    texture: dict = field(default_factory=lambda: {
        "bumpmapscaling": 1.0, "bumpintensity": 1.0, "alphamean": -1.0,
        "gamma": 1.0, "envmapalpha": 1.0, "diffusebumpintensity": 1.0,
        "specularbumpintensity": 1.0, "downsamplemax": 15, "downsamplemin": 0,
        "clamp": 0, "numx": 1, "numy": 1, "filerange": 0,
        "defaultwidth": 2, "defaultheight": 2, "isbumpmap": 0,
        "isdiffusebumpmap": True, "isspecularbumpmap": True,
        "isenvironmentmapped": False, "temporary": False, "useglobalalpha": False,
        "cube": False, "mipmap": True, "filter": True, "maptexelstopixels": False,
        "texture3d": False,
        "specularcolor": (1.0, 1.0, 1.0),
    })
    material: dict = field(default_factory=lambda: {
        "blending": None, "decal": False, "wateralpha": 1.0,
        "renderbmlmtype": 1, "bumpmaptexture": None,
        "bumpyshinytexture": None, "envmaptexture": None,
    })
    font: dict | None = None
    controller: str | None = None
    controller_settings: dict = field(default_factory=dict)


class TXI:
    """Lossless TXI source with occurrence-aware, targeted editing.

    Reading/writing a document does not evaluate its settings. This permits raw
    preservation independently of whether a consumer supports every directive.
    """

    def __init__(self, data: bytes | str = b""):
        self._data = data.encode("latin-1") if isinstance(data, str) else bytes(data)

    def __bytes__(self) -> bytes:
        return self._data

    def __str__(self) -> str:
        return self.text

    @property
    def text(self) -> str:
        return self._data.decode("latin-1")

    @property
    def records(self) -> tuple[TXIRecord, ...]:
        records = []
        position = 0
        while position < len(self._data):
            start = position
            line, position = _line(self._data, position)
            line_end = position
            visible = line.split(b"\0", 1)[0].rstrip(b"\r\n")
            match = _FIELD.fullmatch(visible)
            keyword = match[1].lower().decode("latin-1") if match else ""
            operand = (match[2] or b"") if match else b""
            values: tuple = ()
            binary = counted = False
            if keyword in _LISTS:
                components = _LISTS[keyword]
                count_match = _INT.match(operand)
                counted = count_match is not None
                count = int(count_match[1]) if counted else None
                if count is not None and count < 0:
                    raise ValueError("A TXI list cannot have a negative element count.")
                binary = counted and bool(operand[count_match.end():].strip())
                if binary:
                    length = count * components * 4
                    if length > len(self._data) - position:
                        raise ValueError(f"Truncated binary TXI list {keyword!r}.")
                    array = struct.unpack_from(f"<{count * components}f", self._data, position)
                    values = tuple(tuple(array[i:i + components]) if components > 1 else array[i]
                                   for i in range(0, len(array), components))
                    position += length
                else:
                    items = []
                    terminated = False
                    while position < len(self._data) and (count is None or len(items) < count):
                        item, position = _line(self._data, position)
                        visible_item = item.split(b"\0", 1)[0].strip()
                        if count is None and visible_item.startswith(b"endlist"):
                            terminated = True
                            break
                        parsed = _floats(visible_item, components)
                        items.append(parsed if components > 1 else parsed[0])
                    if (count is not None and len(items) != count) or (count is None and not terminated):
                        raise ValueError(f"Truncated text TXI list {keyword!r}.")
                    values = tuple(items)
            records.append(TXIRecord(start, position, line_end, keyword, operand, values, binary, counted))
        return tuple(records)

    @property
    def has_binary_blocks(self) -> bool:
        return any(record.binary for record in self.records)

    def settings(self, *, game: int = 2) -> TXISettings:
        state = TXISettings()
        is_k2 = Game(game).is_k2()
        for record in self.records:
            key, value = record.keyword, record.operand
            token = value.split()[0] if value.split() else b""
            if key in _SHORTS:
                state.texture[key] = (_integer(value) + 0x8000) % 0x10000 - 0x8000
            elif key in _BOOLS or key == "decal":
                lowered = token.lower()
                if lowered in (b"true", b"false", b"1", b"0"):
                    (state.material if key == "decal" else state.texture)[key] = lowered in (b"true", b"1")
            elif key in _FLOATS:
                state.texture[key] = _floats(value, 1)[0]
            elif key == "isbumpmap":
                state.texture[key] = _integer(value)
            elif key == "specularcolor":
                state.texture[key] = _floats(value, 3)
            elif key == "proceduretype" and token:
                name = token.decode("latin-1")
                state.controller = None
                state.controller_settings = {}
                if name in _CONTROLLERS and (is_k2 or name not in {"dirty", "dirty2", "dirty3"}):
                    state.controller = name
                    if name in {"dirty", "dirty2", "dirty3"}:
                        state.texture["texture3d"] = True
                    state.controller_settings = {
                        "distort": 0, "distortangle": 1, "distortionamplitude": 5.0,
                        "speed": 1.0, "channelscale": [], "channeltranslate": [],
                    }
                    if name == "cycle":
                        state.controller_settings["fps"] = 1.0
                    elif name == "water":
                        state.controller_settings.update(waterwidth=31, waterheight=31,
                                                         forcecyclespeed=57.2957763671875,
                                                         anglecyclespeed=2.864788770675659)
                    elif name == "arturo":
                        state.controller_settings.update(arturowidth=15, arturoheight=15)
            elif key in {"bumpmaptexture", "bumpyshinytexture", "envmaptexture"} and token:
                state.material[key] = token.decode("latin-1")
            elif key == "blending":
                if token in (b"additive", b"punchthrough"):
                    state.material[key] = token.decode("ascii")
            elif key == "renderbmlmtype":
                state.material[key] = _integer(value)
            elif key == "wateralpha":
                state.material[key] = _floats(value, 1)[0]
            elif key in _FONT_FLOATS or key in {"numchars", "upperleftcoords", "lowerrightcoords"}:
                if state.font is None:
                    state.font = dict(numchars=0, fontheight=0.0, baselineheight=0.0,
                                      texturewidth=0.0, spacingr=0.0, spacingb=0.0,
                                      upperleftcoords=(), lowerrightcoords=())
                if key in _FONT_FLOATS:
                    state.font[key] = _floats(value, 1)[0]
                elif key == "numchars":
                    state.font[key] = _integer(value)
                else:
                    state.font[key] = (record.values if record.counted
                                       else record.values + state.font[key][len(record.values):])
            elif state.controller is not None:
                fields = state.controller_settings
                if key in {"distort", "distortangle"}:
                    fields[key] = _integer(value)
                elif key in {"distortionamplitude", "speed"}:
                    fields[key] = _floats(value, 1)[0]
                elif key in {"channelscale", "channeltranslate"}:
                    fields[key] = (list(record.values) if record.counted
                                   else list(record.values) + fields[key][len(record.values):])
                elif re.fullmatch(r"(?:channelscale|channeltranslate)[0-3]", key):
                    name, index = key[:-1], int(key[-1])
                    channel_value = _floats(value, 1)[0]
                    if channel_value >= 0.0:
                        if not fields[name]:
                            fields[name] = [1.0] * 4
                        fields[name][index] = channel_value
                elif state.controller == "cycle" and key == "fps":
                    fields[key] = _floats(value, 1)[0]
                elif state.controller == "water" and key in {"waterwidth", "waterheight"}:
                    fields[key] = _integer(value)
                elif state.controller == "water" and key in {"forcecyclespeed", "anglecyclespeed"}:
                    fields[key] = _floats(value, 1)[0]
                elif state.controller == "arturo" and key in {"arturowidth", "arturoheight"}:
                    fields[key] = _integer(value)
        return state

    def _insertion_offset(self) -> int:
        records = self.records
        if records and records[-1].binary:
            return len(self._data)
        return len(self._data.rstrip(b"\0"))

    def update_text(self, value: str) -> None:
        """Apply a text-editor change without normalizing intact source lines.

        Binary lists must be edited through set_list(), not a plain-text widget.
        """
        if self.has_binary_blocks:
            raise ValueError("Binary TXI lists require structured editing.")
        encoded = value.encode("latin-1")
        end = self._insertion_offset()
        old = self._data[:end].splitlines(keepends=True)
        new = encoded.rstrip(b"\0").splitlines(keepends=True)
        normalize = lambda line: re.sub(rb"\r\n|\r", b"\n", line)
        matcher = difflib.SequenceMatcher(a=[normalize(line) for line in old],
                                         b=[normalize(line) for line in new], autojunk=False)
        result = []
        newline = self._newline()
        for operation, i, j, k, n in matcher.get_opcodes():
            if operation == "equal":
                result.extend(old[i:j])
            elif operation in ("replace", "insert"):
                result.extend(line.rstrip(b"\r\n") + newline if line.endswith((b"\r", b"\n")) else line for line in new[k:n])
        self._data = b"".join(result) + self._data[end:]

    def _newline(self) -> bytes:
        ending = _LINE_END.search(self._data)
        return ending[0] if ending else b"\n"

    def set(self, keyword: str, value: str, *, occurrence: int = -1) -> None:
        """Edit one scalar occurrence, or append an absent command without sorting.

        Existing keyword spelling, indentation, trailing whitespace, inline
        comments and line endings are preserved. List editing uses set_list().
        """
        if not keyword or re.search(r"\s|\x00", keyword):
            raise ValueError("A TXI keyword must be a single nonempty token.")
        if keyword.lower() in _LISTS:
            raise ValueError("Use set_list() for a TXI list.")
        encoded = value.encode("latin-1")
        if b"\n" in encoded or b"\r" in encoded or b"\0" in encoded:
            raise ValueError("A scalar TXI operand cannot contain a line ending or NUL.")
        matches = [r for r in self.records if r.keyword == keyword.lower()]
        if not matches:
            end = self._insertion_offset()
            separator = b"" if end == 0 or self._data[end - 1:end] in (b"\r", b"\n") else self._newline()
            self._data = (self._data[:end] + separator + keyword.encode("ascii") + b" " + encoded
                          + self._newline() + self._data[end:])
            return
        record = matches[occurrence]
        line = self._data[record.start:record.line_end]
        match = re.match(rb"([ \t]*[^ \t\r\n\0]+)([ \t]*)(.*)", line, re.S)
        body = match[3]
        ending = b"\r\n" if body.endswith(b"\r\n") else body[-1:] if body.endswith((b"\r", b"\n")) else b""
        body = body[:-len(ending)] if ending else body
        suffix_match = re.search(rb"[ \t]+(?:#|//).*$|\0.*$|[ \t\r]+$", body, re.S)
        suffix = suffix_match[0] if suffix_match else b""
        replacement = match[1] + (match[2] or b" ") + encoded + suffix + ending
        self._data = self._data[:record.start] + replacement + self._data[record.end:]

    def set_list(self, keyword: str, values: Iterable, *, occurrence: int = -1, binary: bool | None = None) -> None:
        """Replace only the chosen list block, retaining its binary/text form."""
        key = keyword.lower()
        if key not in _LISTS:
            raise ValueError(f"Not a known TXI list: {keyword!r}.")
        components = _LISTS[key]
        items = list(values)
        flattened = []
        for item in items:
            element = tuple(item) if components > 1 else (item,)
            if len(element) != components:
                raise ValueError("Incorrect TXI list element width.")
            flattened.extend(float(value) for value in element)
        matches = [r for r in self.records if r.keyword == key]
        record = matches[occurrence] if matches else None
        binary = record.binary if binary is None and record is not None else bool(binary)
        newline = self._newline()
        if record is None:
            prefix = keyword.encode("ascii")
            space = b" "
            start = end = self._insertion_offset()
            before = b"" if start == 0 or self._data[start - 1:start] in (b"\r", b"\n") else newline
        else:
            line = self._data[record.start:record.line_end]
            match = re.match(rb"([ \t]*[^ \t\r\n\0]+)([ \t]*)", line)
            prefix, space = match[1], match[2] or b" "
            start, end, before = record.start, record.end, b""
            newline = b"\r\n" if line.endswith(b"\r\n") else line[-1:] if line.endswith((b"\r", b"\n")) else newline
        if binary:
            replacement = prefix + space + str(len(items)).encode("ascii") + b" b" + newline
            replacement += struct.pack(f"<{len(flattened)}f", *flattened)
        else:
            counted = record is None or record.counted
            replacement = prefix + (space + str(len(items)).encode("ascii") if counted else b"") + newline
            for index in range(len(items)):
                replacement += b" ".join(format(x, ".9g").encode("ascii") for x in flattened[index * components:(index + 1) * components]) + newline
            if not counted:
                replacement += b"endlist" + newline
        self._data = self._data[:start] + before + replacement + self._data[end:]
