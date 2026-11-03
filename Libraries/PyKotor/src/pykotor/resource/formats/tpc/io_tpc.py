from __future__ import annotations

import struct

from typing import TYPE_CHECKING

from pykotor.resource.formats.tpc.tpc_data import TPC, TPCHeader
from pykotor.resource.formats.txi import TXI
from pykotor.resource.type import ResourceReader, ResourceWriter, autoclose

if TYPE_CHECKING:
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


class TPCBinaryReader(ResourceReader):
    def __init__(self, source: SOURCE_TYPES, offset: int = 0, size: int = 0, *, game: int = 1):
        super().__init__(source, offset, size)
        self._game = game

    @autoclose
    def load(self, auto_close: bool = True) -> TPC:
        raw = self._reader.read_bytes(self._size)
        if len(raw) < 128:
            raise ValueError("TPC header is shorter than 128 bytes.")
        header = TPCHeader(*struct.unpack_from("<IIHHBB", raw), raw[14:128], self._game)
        end = 128 + header.image_size()
        if end > len(raw):
            raise ValueError("TPC image data extends beyond the resource.")
        tpc = TPC()
        tpc.header = header
        tpc._texture_format = header.texture_format()
        tpc._image_data = raw[128:end]
        tpc.embedded_txi = TXI(raw[end:])
        return tpc


class TPCBinaryWriter(ResourceWriter):
    def __init__(self, tpc: TPC, target: TARGET_TYPES):
        # Validate before ResourceWriter can open a file or truncate a bytearray.
        header = tpc.header.to_bytes()
        if tpc.header.image_size() != len(tpc.image_data):
            raise ValueError("TPC header and stored image extent disagree.")
        self._data = header + tpc.image_data + bytes(tpc.embedded_txi)
        super().__init__(target)

    @autoclose
    def write(self, auto_close: bool = True):
        self._writer.write_bytes(self._data)
