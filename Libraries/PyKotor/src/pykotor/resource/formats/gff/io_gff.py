from __future__ import annotations

import hashlib
import struct
import uuid
from typing import TYPE_CHECKING, Any

from pykotor.common.geometry import Vector4
from pykotor.common.misc import ResRef
from pykotor.common.stream import BinaryWriter
from pykotor.resource.formats.gff.gff_data import (
    GFF, GFFContent, GFFFieldType, GFFList, GFFStruct,
    _GFFFieldBinary, _GFFStructBinary,
)
from pykotor.resource.type import ResourceReader, ResourceWriter, autoclose

if TYPE_CHECKING:
    from pykotor.resource.formats.gff.gff_data import _GFFField
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES

_COMPLEX_FIELD: set[GFFFieldType] = {
    GFFFieldType.UInt64,
    GFFFieldType.Int64,
    GFFFieldType.Double,
    GFFFieldType.String,
    GFFFieldType.ResRef,
    GFFFieldType.LocalizedString,
    GFFFieldType.Binary,
    GFFFieldType.Vector3,
    GFFFieldType.Vector4,
}


def _value_bytes(kind: GFFFieldType, value: Any, raw_data: bytes | None = None) -> bytes:
    writer = BinaryWriter.to_bytearray()
    if kind in (GFFFieldType.UInt8, GFFFieldType.UInt16, GFFFieldType.UInt32):
        writer.write_uint32(value, max_neg1=True)
    elif kind in (GFFFieldType.Int8, GFFFieldType.Int16, GFFFieldType.Int32):
        writer.write_int32(value)
    elif kind is GFFFieldType.Single:
        writer.write_single(value)
    elif kind is GFFFieldType.UInt64:
        writer.write_uint64(value)
    elif kind is GFFFieldType.Int64:
        writer.write_int64(value)
    elif kind is GFFFieldType.Double:
        writer.write_double(value)
    elif kind is GFFFieldType.String:
        data = raw_data if raw_data is not None else value.encode("cp1252", "surrogateescape")
        writer.write_uint32(len(data))
        writer.write_bytes(data)
    elif kind is GFFFieldType.ResRef:
        data = value.to_bytes(fixed_width=False)
        writer.write_uint8(len(data))
        writer.write_bytes(data)
    elif kind is GFFFieldType.LocalizedString:
        writer.write_locstring(value)
    elif kind is GFFFieldType.Binary:
        writer.write_uint32(len(value))
        writer.write_bytes(value)
    elif kind is GFFFieldType.Vector3:
        writer.write_vector3(value)
    elif kind is GFFFieldType.Vector4:
        # GFF quaternions store the scalar component before X, Y and Z.
        writer.write_bytes(struct.pack("<4f", value.w, value.x, value.y, value.z))
    else:
        raise ValueError(f"Unknown field type '{kind}'")
    return bytes(writer.data())


def _payload_size(kind: GFFFieldType, data: bytes | bytearray, offset: int) -> int:
    if kind in (GFFFieldType.UInt64, GFFFieldType.Int64, GFFFieldType.Double):
        return 8
    if kind is GFFFieldType.Vector3:
        return 12
    if kind is GFFFieldType.Vector4:
        return 16
    if kind is GFFFieldType.ResRef:
        return 1 + data[offset]
    return 4 + struct.unpack_from("<I", data, offset)[0]


def _struct_signatures(root: GFFStruct) -> dict[int, bytes]:
    """Identify equal retained records when a shared struct has multiple editor views."""
    signatures: dict[int, bytes] = {}

    def visit(node: GFFStruct) -> bytes:
        if id(node) in signatures:
            return signatures[id(node)]
        digest = hashlib.sha256(struct.pack("<I", node.struct_id & 0xFFFFFFFF))
        digest.update((node._source_key or "").encode("ascii"))
        for label, field in node._field_order:
            name = label.encode("ascii", "surrogateescape")
            kind, value = field.field_type(), field.value()
            if kind is GFFFieldType.Struct:
                data = visit(value)
            elif kind is GFFFieldType.List:
                data = struct.pack("<I", len(value)) + b"".join(visit(row) for row in value)
            else:
                data = _value_bytes(kind, value, field.raw_data)
            digest.update(struct.pack("<III", len(name), kind.value, len(data)))
            digest.update(name)
            digest.update(data)
        signatures[id(node)] = digest.digest()
        return signatures[id(node)]

    visit(root)
    return signatures


class GFFBinaryReader(ResourceReader):
    def __init__(
        self,
        source: SOURCE_TYPES,
        offset: int = 0,
        size: int = 0,
    ):
        super().__init__(source, offset, size)
        self._gff: GFF | None = None
        self._structs: dict[int, GFFStruct] = {}

        self._labels: list[str] = []
        self._field_data_offset: int = 0
        self._field_indices_offset: int = 0
        self._list_indices_offset: int = 0
        self._struct_offset: int = 0
        self._field_offset: int = 0

    @autoclose
    def load(
        self,
        auto_close: bool = True,
    ) -> GFF:
        self._gff = GFF()
        self._structs.clear()
        self._reader.seek(0)
        self._gff._binary_data = self._reader.read_bytes(self._size)
        self._reader.seek(0)

        file_type = self._reader.read_string(4)
        file_version = self._reader.read_string(4)

        if not any(x for x in GFFContent if x.value == file_type):
            msg = "Not a valid binary GFF file."
            raise ValueError(msg)

        if file_version != "V3.2":
            msg = "The GFF version of the file is unsupported."
            raise ValueError(msg)

        self._gff.content = GFFContent(file_type)

        self._struct_offset = self._reader.read_uint32()
        self._reader.read_uint32()  # struct count
        self._field_offset = self._reader.read_uint32()
        self._reader.read_uint32()  # field count
        label_offset = self._reader.read_uint32()
        label_count = self._reader.read_uint32()
        self._field_data_offset = self._reader.read_uint32()
        self._reader.read_uint32()  # field data count
        self._field_indices_offset = self._reader.read_uint32()
        self._reader.read_uint32()  # field indices count
        self._list_indices_offset = self._reader.read_uint32()
        self._reader.read_uint32()  # list indices count

        self._labels = []
        self._reader.seek(label_offset)
        self._labels.extend(self._reader.read_bytes(16).split(b"\0", 1)[0].decode("ascii", "surrogateescape") for _ in range(label_count))
        self._gff.root = self._load_struct(0)
        signatures = _struct_signatures(self._gff.root)
        for index, node in self._structs.items():
            node._binary = _GFFStructBinary(self._gff._binary_data, index, node._source_key, signatures[id(node)])

        return self._gff

    def _load_struct(
        self,
        struct_index: int,
    ) -> GFFStruct:
        if struct_index in self._structs:
            return self._structs[struct_index]
        gff_struct = GFFStruct()
        gff_struct._source_key = uuid.uuid4().hex
        self._structs[struct_index] = gff_struct
        self._reader.seek(self._struct_offset + struct_index * 12)
        struct_id, data, field_count = (
            self._reader.read_uint32(max_neg1=True),
            self._reader.read_uint32(),
            self._reader.read_uint32(),
        )

        gff_struct.struct_id = struct_id

        if field_count == 1:
            self._load_field(gff_struct, data)
        elif field_count > 1:
            self._reader.seek(self._field_indices_offset + data)
            indices: list[int] = [self._reader.read_uint32() for _ in range(field_count)]
            for index in indices:
                self._load_field(gff_struct, index)
        return gff_struct

    def _load_field(
        self,
        gff_struct: GFFStruct,
        field_index: int,
    ):
        self._reader.seek(self._field_offset + field_index * 12)
        field_type_id = self._reader.read_uint32()
        label_id = self._reader.read_uint32()
        data_ref = self._reader.read_uint32()
        self._reader.seek(self._field_offset + field_index * 12 + 8)

        field_type = GFFFieldType(field_type_id)
        label = self._labels[label_id]

        if field_type in _COMPLEX_FIELD:
            offset = self._reader.read_uint32()  # relative to field data
            self._reader.seek(self._field_data_offset + offset)
            if field_type is GFFFieldType.UInt64:
                gff_struct.add_field(label, GFFFieldType.UInt64, self._reader.read_uint64())
            elif field_type is GFFFieldType.Int64:
                gff_struct.add_field(label, GFFFieldType.Int64, self._reader.read_int64())
            elif field_type is GFFFieldType.Double:
                gff_struct.add_field(label, GFFFieldType.Double, self._reader.read_double())
            elif field_type is GFFFieldType.String:
                length = self._reader.read_uint32()
                raw_text = self._reader.read_bytes(length)
                gff_struct.add_field(label, GFFFieldType.String, raw_text.decode("cp1252", "surrogateescape"), raw_data=raw_text)
            elif field_type is GFFFieldType.ResRef:
                length = self._reader.read_uint8()
                resref = ResRef.from_bytes(self._reader.read_bytes(length), fixed_width=False)
                gff_struct.add_field(label, GFFFieldType.ResRef, resref)
            elif field_type is GFFFieldType.LocalizedString:
                gff_struct.add_field(label, GFFFieldType.LocalizedString, self._reader.read_locstring())
            elif field_type is GFFFieldType.Binary:
                length = self._reader.read_uint32()
                gff_struct.add_field(label, GFFFieldType.Binary, self._reader.read_bytes(length))
            elif field_type is GFFFieldType.Vector3:
                gff_struct.add_field(label, GFFFieldType.Vector3, self._reader.read_vector3())
            elif field_type is GFFFieldType.Vector4:
                w, x, y, z = struct.unpack("<4f", self._reader.read_bytes(16))
                gff_struct.add_field(label, GFFFieldType.Vector4, Vector4(x, y, z, w))
        elif field_type is GFFFieldType.Struct:
            struct_index = self._reader.read_uint32()
            new_struct = self._load_struct(struct_index)
            gff_struct.add_field(label, GFFFieldType.Struct, new_struct)
        elif field_type is GFFFieldType.List:
            self._load_list(gff_struct, label)
        elif field_type is GFFFieldType.UInt8:
            gff_struct.add_field(label, GFFFieldType.UInt8, self._reader.read_uint8())
        elif field_type is GFFFieldType.Int8:
            gff_struct.add_field(label, GFFFieldType.Int8, self._reader.read_int8())
        elif field_type is GFFFieldType.UInt16:
            gff_struct.add_field(label, GFFFieldType.UInt16, self._reader.read_uint16())
        elif field_type is GFFFieldType.Int16:
            gff_struct.add_field(label, GFFFieldType.Int16, self._reader.read_int16())
        elif field_type is GFFFieldType.UInt32:
            gff_struct.add_field(label, GFFFieldType.UInt32, self._reader.read_uint32())
        elif field_type is GFFFieldType.Int32:
            gff_struct.add_field(label, GFFFieldType.Int32, self._reader.read_int32())
        elif field_type is GFFFieldType.Single:
            gff_struct.add_field(label, GFFFieldType.Single, self._reader.read_single())

        field = gff_struct._field_order[-1][1]
        source = self._gff._binary_data
        if field_type in _COMPLEX_FIELD:
            start = self._field_data_offset + data_ref
            data = source[start:start + _payload_size(field_type, source, start)]
        elif field_type is GFFFieldType.List:
            start = self._list_indices_offset + data_ref
            size = 4 + 4 * len(field.value())
            data = source[start:start + size]
        else:
            data = struct.pack("<I", data_ref)
        value = b"" if field_type in (GFFFieldType.Struct, GFFFieldType.List) else _value_bytes(field_type, field.value(), field.raw_data)
        field._binary = _GFFFieldBinary(source, field_index, field_type, label_id, data_ref, value, data)

    def _load_list(self, gff_struct: GFFStruct, label: str):
        offset = self._reader.read_uint32()  # relative to list indices
        self._reader.seek(self._list_indices_offset + offset)
        value = GFFList()
        count = self._reader.read_uint32()
        list_indices: list[int] = [self._reader.read_uint32() for _ in range(count)]
        for struct_index in list_indices:
            value._structs.append(self._load_struct(struct_index))
        gff_struct.add_field(label, GFFFieldType.List, value)


class GFFBinaryWriter(ResourceWriter):
    """Retain binary table slots and append new records without implicit compaction.

    Unedited files are written byte-for-byte. Growing a section moves later
    sections while retaining original record slots and untouched allocations.
    """

    def __init__(self, gff: GFF, target: TARGET_TYPES):
        super().__init__(target)
        self._gff = gff
        self._source = gff._binary_data
        header = struct.unpack_from("<12I", self._source, 8) if self._source else (56, 0) * 6
        self._offsets = list(header[::2])
        self._sizes = [count * width for count, width in zip(header[1::2], (12, 12, 16, 1, 1, 1))]
        self._buffers = [
            bytearray(self._source[offset:offset + size])
            for offset, size in zip(self._offsets, self._sizes)
        ]
        # Field data, field-index arrays and list-index arrays can be shared.
        # Keep every original allocation, including currently unreferenced records.
        self._blocks: dict[int, list[tuple[int, int, int]]] = {3: [], 4: [], 5: []}
        for index in range(header[1]):
            _, offset, count = struct.unpack_from("<III", self._buffers[0], index * 12)
            if count > 1:
                self._blocks[4].append((offset, count * 4, index))
        for index in range(header[3]):
            kind, _, offset = struct.unpack_from("<III", self._buffers[1], index * 12)
            if kind in _COMPLEX_FIELD:
                size = _payload_size(GFFFieldType(kind), self._buffers[3], offset)
                self._blocks[3].append((offset, size, index))
            elif kind == GFFFieldType.List:
                size = 4 + 4 * struct.unpack_from("<I", self._buffers[5], offset)[0]
                self._blocks[5].append((offset, size, index))

        self._labels = [
            bytes(self._buffers[2][index:index + 16]).split(b"\0", 1)[0].decode("ascii", "surrogateescape")
            for index in range(0, len(self._buffers[2]), 16)
        ]
        self._label_indices: dict[str, int] = {}
        for index, label in enumerate(self._labels):
            self._label_indices.setdefault(label, index)
        self._nodes: dict[int, tuple[GFFStruct, bool]] = {}
        self._struct_indices: dict[int, int] = {}
        self._struct_nodes: dict[int, tuple[GFFStruct, bool]] = {}
        self._field_indices: dict[tuple[int, int], int] = {}
        self._fields: dict[int, tuple[int, _GFFField, bytes]] = {}

    @autoclose
    def write(self, auto_close: bool = True):
        self._collect_structs(self._gff.root, False)
        self._assign_structs(_struct_signatures(self._gff.root))
        self._assign_fields()
        self._reserve_blocks()
        self._write_fields()
        self._write_structs()

        data = bytearray(self._source) if self._source else bytearray(56)
        offsets = self._offsets.copy()
        # Keep the source's section order, gaps and trailer. Previously empty
        # sections acquire storage at the original end of the file.
        sections = sorted(
            (self._offsets[index] if self._sizes[index] else len(data), index)
            for index, buffer in enumerate(self._buffers) if buffer
        )
        growth = 0
        for original_offset, index in sections:
            offset = original_offset + growth
            buffer = self._buffers[index]
            data[offset:offset + self._sizes[index]] = buffer
            offsets[index] = offset
            growth += len(buffer) - self._sizes[index]
        header = [
            number
            for offset, buffer, width in zip(offsets, self._buffers, (12, 12, 16, 1, 1, 1))
            for number in (offset, len(buffer) // width)
        ]
        data[:8] = self._gff.content.value.encode("ascii") + b"V3.2"
        struct.pack_into("<12I", data, 8, *header)
        self._writer.write_bytes(data)

    def _collect_structs(self, node: GFFStruct, detached: bool):
        if id(node) in self._nodes:
            return
        binary = node._binary
        detached = detached or (
            binary is not None and binary.source is self._source and binary.key != node._source_key
        )
        self._nodes[id(node)] = node, detached
        for _, field in node._field_order:
            if field.field_type() is GFFFieldType.Struct:
                self._collect_structs(field.value(), detached)
            elif field.field_type() is GFFFieldType.List:
                for child in field.value():
                    self._collect_structs(child, detached)

    def _assign_structs(self, signatures: dict[int, bytes]):
        root = self._gff.root
        self._struct_indices[id(root)] = 0
        self._struct_nodes[0] = self._nodes[id(root)]
        if not self._buffers[0]:
            self._buffers[0].extend(bytes(12))
        claimed = {0: signatures[id(root)]}

        def unchanged(entry):
            node, detached = entry
            binary = node._binary
            return (
                not detached and binary is not None and binary.source is self._source
                and signatures[id(node)] == binary.signature
            )

        # An unchanged view owns the original slot when another view edits a shared record.
        for node, detached in sorted(self._nodes.values(), key=unchanged, reverse=True):
            if id(node) in self._struct_indices:
                continue
            binary = node._binary
            index = binary.index if not detached and binary is not None and binary.source is self._source else None
            signature = signatures[id(node)]
            if index in claimed and claimed[index] == signature:
                self._struct_indices[id(node)] = index
                continue
            if index is None or index in claimed:
                index = len(self._buffers[0]) // 12
                self._buffers[0].extend(bytes(12))
            claimed[index] = signature
            self._struct_indices[id(node)] = index
            self._struct_nodes[index] = node, detached

    def _label_index(self, label: str, binary: _GFFFieldBinary | None) -> int:
        if binary is not None and binary.source is self._source and self._labels[binary.label_index] == label:
            return binary.label_index
        index = self._label_indices.get(label)
        if index is None:
            data = label.encode("ascii", "surrogateescape")
            if len(data) > 16:
                raise ValueError(f"GFF field label exceeds 16 bytes: {label!r}")
            index = len(self._labels)
            self._labels.append(label)
            self._label_indices[label] = index
            self._buffers[2].extend(data.ljust(16, b"\0"))
        return index

    def _field_data(self, field: _GFFField) -> bytes:
        kind, value = field.field_type(), field.value()
        if kind is GFFFieldType.Struct:
            return struct.pack("<I", self._struct_indices[id(value)])
        if kind is GFFFieldType.List:
            return struct.pack("<I", len(value)) + b"".join(
                struct.pack("<I", self._struct_indices[id(node)]) for node in value
            )
        data = _value_bytes(kind, value, field.raw_data)
        binary = field._binary
        if binary is not None and binary.kind is kind:
            if data == binary.value:
                return binary.data
            widths = {
                GFFFieldType.UInt8: 1, GFFFieldType.Int8: 1,
                GFFFieldType.UInt16: 2, GFFFieldType.Int16: 2,
            }
            if kind in widths:
                width = widths[kind]
                return data[:width] + binary.data[width:]
        return data

    def _assign_fields(self):
        entries = []
        for struct_index, (node, detached) in self._struct_nodes.items():
            for ordinal, (label, field) in enumerate(node._field_order):
                label_index = self._label_index(label, field._binary)
                data = self._field_data(field)
                binary = field._binary
                unchanged = (
                    binary is not None and binary.kind is field.field_type()
                    and binary.label_index == label_index and binary.data == data
                )
                entries.append((unchanged, struct_index, ordinal, detached, label_index, field, data))

        claimed = {}
        for _, struct_index, ordinal, detached, label_index, field, data in sorted(
            entries, key=lambda entry: entry[0], reverse=True,
        ):
            binary = field._binary
            index = binary.index if not detached and binary is not None and binary.source is self._source else None
            signature = field.field_type(), label_index, data
            if index in claimed and claimed[index] == signature:
                self._field_indices[struct_index, ordinal] = index
                continue
            if index is None or index in claimed:
                index = len(self._buffers[1]) // 12
                self._buffers[1].extend(bytes(12))
            claimed[index] = signature
            self._field_indices[struct_index, ordinal] = index
            self._fields[index] = label_index, field, data

    def _reserve_blocks(self):
        # New copies may still reference an unchanged payload. Reserve those uses
        # before writing so an edit to the original cannot overwrite its copy.
        for index, (_, field, _) in self._fields.items():
            binary = field._binary
            if binary is not None and binary.source is self._source:
                if binary.kind in _COMPLEX_FIELD:
                    self._blocks[3].append((binary.offset, len(binary.data), index))
                elif binary.kind is GFFFieldType.List:
                    self._blocks[5].append((binary.offset, len(binary.data), index))
        for index, (node, _) in self._struct_nodes.items():
            binary = node._binary
            if binary is not None and binary.source is self._source:
                _, offset, count = struct.unpack_from("<III", self._source, self._offsets[0] + binary.index * 12)
                if count > 1:
                    self._blocks[4].append((offset, count * 4, index))

    def _place_block(self, section: int, data: bytes, owner: int, old: tuple[int, int] | None) -> int:
        buffer = self._buffers[section]
        if old is not None:
            offset, size = old
            if len(data) <= size and buffer[offset:offset + len(data)] == data:
                return offset
            end = offset + len(data)
            exclusive = all(
                other == owner or start >= end or start + length <= offset
                for start, length, other in self._blocks[section]
            )
            if len(data) <= size and exclusive:
                buffer[offset:offset + len(data)] = data
                return offset
        offset = len(buffer)
        buffer.extend(data)
        return offset

    def _write_fields(self):
        for index, (label_index, field, data) in self._fields.items():
            kind = field.field_type()
            binary = field._binary
            if kind in _COMPLEX_FIELD or kind is GFFFieldType.List:
                section = 5 if kind is GFFFieldType.List else 3
                old = None
                if binary is not None and binary.source is self._source:
                    if (kind in _COMPLEX_FIELD and binary.kind in _COMPLEX_FIELD) or kind is binary.kind:
                        old = binary.offset, len(binary.data)
                offset = self._place_block(section, data, index, old)
                data = struct.pack("<I", offset)
            self._buffers[1][index * 12:(index + 1) * 12] = struct.pack("<II", kind.value, label_index) + data

    def _write_structs(self):
        for index, (node, _) in self._struct_nodes.items():
            binary = node._binary
            old_offset, old_count = 0xFFFFFFFF, 0
            if binary is not None and binary.source is self._source:
                _, old_offset, old_count = struct.unpack_from(
                    "<III", self._source, self._offsets[0] + binary.index * 12,
                )
            fields = [self._field_indices[index, ordinal] for ordinal in range(len(node))]
            count = len(fields)
            if count == 0:
                offset = old_offset if old_count == 0 else 0xFFFFFFFF
            elif count == 1:
                offset = fields[0]
            else:
                data = b"".join(struct.pack("<I", field) for field in fields)
                old = (old_offset, old_count * 4) if old_count > 1 else None
                offset = self._place_block(4, data, index, old)
            struct.pack_into("<III", self._buffers[0], index * 12, node.struct_id & 0xFFFFFFFF, offset, count)
