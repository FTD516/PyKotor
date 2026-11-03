from __future__ import annotations

import os
import shutil
import struct
import tempfile

from pathlib import Path
from typing import TYPE_CHECKING

from pykotor.common.misc import Game
from pykotor.common.stream import BinaryReader
from pykotor.resource.formats.tpc.io_bmp import TPCBMPWriter
from pykotor.resource.formats.tpc.io_tga import TPCTGAReader, TPCTGAWriter
from pykotor.resource.formats.tpc.io_tpc import TPCBinaryReader, TPCBinaryWriter
from pykotor.resource.formats.tpc.tpc_data import TPCHeader
from pykotor.resource.formats.txi import read_txi
from pykotor.resource.type import ResourceType
from pykotor.tools.path import CaseAwarePath

if TYPE_CHECKING:
    from pykotor.resource.formats.tpc.tpc_data import TPC
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


def _source_bytes(source: SOURCE_TYPES, offset: int, size: int | None) -> bytes:
    if offset < 0 or size is not None and size < 0:
        raise ValueError("A texture slice cannot have a negative offset or size.")
    if isinstance(source, BinaryReader):
        position = source.position()
        try:
            source.skip(offset)
            return source.read_bytes(source.remaining() if size is None else size)
        finally:
            source.seek(position)
    with BinaryReader.from_auto(source, offset) as reader:
        return reader.read_bytes(reader.remaining() if size is None else size)


def detect_tpc(source: SOURCE_TYPES, offset: int = 0, *, game: int = 1) -> ResourceType:
    """Classify a header structurally, never by assumed zero padding.

    TPC has no magic. An ambiguous byte sequence returns INVALID; callers with
    a resource identifier should pass that type to read_tpc() explicitly.
    """
    raw = _source_bytes(source, offset, None)
    tpc = False
    if len(raw) >= 128:
        values = struct.unpack_from("<IIHHBB", raw)
        size, _, width, height, encoding, _ = values
        packed = Game(game).is_k1() and size and encoding & 0x10
        channel = next((bit for bit in (1, 2, 4) if encoding & bit), 0)
        if width and height and (packed or channel) and not (size and channel == 1 and not packed):
            header = TPCHeader(*values, raw[14:128], game)
            tpc = 128 + header.image_size() <= len(raw)
    tga = False
    if len(raw) >= 18:
        id_length, color_map, image_type = raw[:3]
        width, height = struct.unpack_from("<HH", raw, 12)
        depth = raw[16]
        tga = (color_map in (0, 1) and image_type in (1, 2, 3, 9, 10, 11)
               and width > 0 and height > 0 and depth in (8, 16, 24, 32)
               and 18 + id_length <= len(raw))
    if tpc == tga:
        return ResourceType.INVALID
    return ResourceType.TPC if tpc else ResourceType.TGA


def read_tpc(source: SOURCE_TYPES, offset: int = 0, size: int | None = None,
             txi_source: SOURCE_TYPES | None = None, *, file_format: ResourceType | None = None, game: int = 1) -> TPC:
    """Read storage separately from the optional external TXI selection.

    An explicit format (or .tpc/.tga filename) is authoritative. A missing
    explicitly supplied TXI raises; automatic sidecar discovery is optional.
    game=2 selects K2's flag interpretation; standalone reading defaults to
    K1, whose format set also includes PVRTC packed chains.
    """
    if file_format is None and isinstance(source, (str, os.PathLike)):
        suffix = Path(source).suffix.lower()
        if suffix in (".tpc", ".tga"):
            file_format = ResourceType.TPC if suffix == ".tpc" else ResourceType.TGA
    raw = _source_bytes(source, offset, size)
    if file_format is None:
        file_format = detect_tpc(raw, game=game)
    if file_format == ResourceType.TPC:
        texture = TPCBinaryReader(raw, game=game).load()
    elif file_format == ResourceType.TGA:
        texture = TPCTGAReader(raw).load()
    else:
        raise ValueError("Cannot identify this texture unambiguously; supply TPC or TGA explicitly.")
    texture.header.game = game
    if txi_source is None and isinstance(source, (str, os.PathLike)):
        candidate = CaseAwarePath.get_case_sensitive_path(Path(source).with_suffix(".txi"))
        if candidate.is_file():
            txi_source = candidate
    if txi_source is not None:
        texture.external_txi = read_txi(txi_source)
        if isinstance(txi_source, (str, os.PathLike)):
            texture.external_txi_source = os.fspath(txi_source)
    return texture


def write_tpc(tpc: TPC, target: TARGET_TYPES, file_format: ResourceType = ResourceType.TPC,
              *, image: int | None = None, mipmap: int = 0) -> None:
    """Write all TPC storage, or explicitly export one image to TGA/BMP.

    External TXI is never implicitly embedded. Use write_txi() to save that
    document. A multi-image raster export requires an explicit image index.
    """
    writers = {ResourceType.TPC: TPCBinaryWriter, ResourceType.TGA: TPCTGAWriter, ResourceType.BMP: TPCBMPWriter}
    if file_format not in writers:
        raise ValueError("Unsupported format specified; use TPC, TGA or BMP.")
    if file_format == ResourceType.TPC:
        if image is not None or mipmap:
            raise ValueError("TPC writing preserves the whole texture; use an explicit raster export for an image.")
        output = tpc
    else:
        if image is None and tpc.image_count() != 1:
            raise ValueError("Select a face or frame explicitly when exporting a compound texture.")
        from pykotor.resource.formats.tpc.tpc_data import TPC, TPCTextureFormat
        width, height, pixels = tpc.convert(TPCTextureFormat.RGBA, mipmap, image=0 if image is None else image)
        output = TPC()
        output.set_single(width, height, bytes(pixels), TPCTextureFormat.RGBA)
    writer = writers[file_format]
    if not isinstance(target, (str, os.PathLike)):
        writer(output, target).write()
        return
    path = Path(target)
    if path.is_dir():
        raise IsADirectoryError(str(path))
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name.lower()}.", suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        writer(output, temporary_path).write()
        if path.exists():
            shutil.copymode(path, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def bytes_tpc(tpc: TPC, file_format: ResourceType = ResourceType.TPC, *, image: int | None = None, mipmap: int = 0) -> bytes:
    data = bytearray()
    write_tpc(tpc, data, file_format, image=image, mipmap=mipmap)
    return bytes(data)
