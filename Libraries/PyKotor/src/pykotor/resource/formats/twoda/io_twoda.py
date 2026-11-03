from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pykotor.common.stream import BinaryReader
from pykotor.resource.formats.twoda.io_twoda_text import _token, _tokens
from pykotor.resource.formats.twoda.twoda_data import TwoDA
from pykotor.resource.type import ResourceReader, ResourceWriter, autoclose

if TYPE_CHECKING:
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES

_DEFAULT_LINE = re.compile(rb"[ \t\r\n]*(?i:DEFAULT:)(?=[ \t\r\n])([^\r\n]*)[\r\n]+")


class TwoDABinaryReader(ResourceReader):
    def __init__(self, source: SOURCE_TYPES, offset: int = 0, size: int = 0):
        super().__init__(source, offset, size)

    @autoclose
    def load(self, auto_close: bool = True) -> TwoDA:
        # Bound terminated strings and offsets to this resource, not its container.
        data = self._reader.read_bytes(self._size)
        with BinaryReader.from_bytes(data) as reader:
            if reader.read_bytes(8) != b"2DA V2.b":
                raise ValueError("Expected a binary 2DA V2.b header.")
            newline = reader.read_bytes(1)
            if newline == b"\r":
                if reader.peek() == b"\n":
                    reader.skip(1)
            elif newline != b"\n":
                raise ValueError("Missing line ending after the binary 2DA header.")

            twoda = TwoDA()
            default_line = _DEFAULT_LINE.match(data, reader.position())
            if default_line is not None:
                twoda.default = next(_tokens(default_line[1].decode("latin-1")), "")
                reader.seek(default_line.end())
            while reader.peek() != b"\0":
                twoda.add_column(reader.read_terminated_string("\t", encoding="latin-1"))
            reader.skip(1)

            row_count = reader.read_uint32()
            for _ in range(row_count):
                twoda.add_row(reader.read_terminated_string("\t", encoding="latin-1"))

            headers = twoda.get_headers()
            offsets = [reader.read_uint16() for _ in range(row_count * len(headers))]
            reader.skip(2)  # Legacy string-pool size; not an addressing bound.
            data_start = reader.position()
            strings: dict[int, str] = {}
            for index, offset in enumerate(offsets):
                if offset not in strings:
                    reader.seek(data_start + offset)
                    strings[offset] = reader.read_terminated_string("\0", encoding="latin-1")
                twoda.set_cell(index // len(headers), headers[index % len(headers)], strings[offset])
            return twoda


class TwoDABinaryWriter(ResourceWriter):
    def __init__(self, twoda: TwoDA, target: TARGET_TYPES):
        super().__init__(target)
        self._twoda = twoda

    @autoclose
    def write(self, auto_close: bool = True):
        headers = self._twoda.get_headers()
        self._writer.write_bytes(b"2DA V2.b\n")
        if self._twoda.default:
            self._writer.write_bytes(("DEFAULT: " + _token(self._twoda.default) + "\n").encode("latin-1"))
        for header in headers:
            self._writer.write_bytes((header + "\t").encode("latin-1"))
        self._writer.write_bytes(b"\0")

        self._writer.write_uint32(self._twoda.get_height())
        for label in self._twoda.get_labels():
            self._writer.write_bytes((str(label) + "\t").encode("latin-1"))

        offsets: dict[bytes, int] = {}
        data = bytearray()
        for index in range(self._twoda.get_height()):
            for header in headers:
                # Serialization uses physical storage, not case-insensitive lookup.
                value = self._twoda.get_cell(index, header).encode("latin-1")
                if b"\0" in value:
                    raise ValueError("A binary 2DA cell cannot contain a null byte.")
                if value not in offsets:
                    offsets[value] = len(data)
                    data.extend(value + b"\0")
                self._writer.write_uint16(offsets[value])

        # Only cell starts are addressed by the on-disk offsets. The final string
        # may extend beyond 64 KiB; the engine does not use this word as a length.
        self._writer.write_uint16(len(data) & 0xFFFF)
        self._writer.write_bytes(data)
