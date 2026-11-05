from __future__ import annotations
import math
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from pykotor.common.geometry import Vector2, Vector3, Vector4
from pykotor.common.misc import Color, Game
from pykotor.common.stream import BinaryReader, BinaryWriter, BinaryWriterBytearray, BinaryWriterFile
from pykotor.resource.formats.mdl.mdl_data import MDL, MDLAABB, MDLAnimation, MDLBoneVertex, MDLController, MDLControllerRow, MDLDangly, MDLEmitter, MDLEvent, MDLFace, MDLLight, MDLMesh, MDLNode, MDLReference, MDLSaber, MDLSkin, MDLWalkmesh
if TYPE_CHECKING:
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES
_GEOMETRY_POINTERS = {Game.K1: (4273776, 4216096), Game.K2: (4285200, 4216320)}
_ANIMATION_POINTERS = {Game.K1: (4273392, 4451552), Game.K2: (4284816, 4522928)}
_MESH_POINTERS = {Game.K1: {32: (4216656, 4216672), 64: (4216592, 4216608), 256: (4216640, 4216624)}, Game.K2: {32: (4216880, 4216896), 64: (4216816, 4216832), 256: (4216864, 4216848)}}
_PART_SIZES = {2: 92, 4: 224, 16: 36, 64: 100, 256: 28, 512: 4, 2048: 20}

def _read_source(source: SOURCE_TYPES, offset: int=0, size: int=0) -> bytes:
    """Read one bounded resource, leaving a borrowed stream at its original position."""
    if offset < 0 or size < 0:
        raise ValueError('Model offsets and sizes must be nonnegative')
    if isinstance(source, (bytes, bytearray, memoryview)):
        end = offset + size if size else len(source)
        if offset > len(source) or end > len(source):
            raise ValueError('Model resource exceeds its input bounds')
        return bytes(source[offset:end])
    if isinstance(source, BinaryReader):
        position = source.position()
        try:
            start = position + offset
            count = size or source.size() - start
            if count < 0 or start + count > source.size():
                raise ValueError('Model resource exceeds its reader bounds')
            source.seek(start)
            return source.read_bytes(count)
        finally:
            source.seek(position)
    if isinstance(source, (str, os.PathLike)):
        with open(source, 'rb') as stream:
            stream.seek(0, 2)
            length = stream.tell()
            end = offset + size if size else length
            if offset > length or end > length:
                raise ValueError('Model resource exceeds its file bounds')
            stream.seek(offset)
            return stream.read(end - offset)
    position = source.tell()
    try:
        source.seek(0, 2)
        length = source.tell()
        start = position + offset
        end = start + size if size else length
        if start > length or end > length:
            raise ValueError('Model resource exceeds its stream bounds')
        source.seek(start)
        return source.read(end - start)
    finally:
        source.seek(position)

class _Buffer:
    """A bounded on-disk address space. MDL addresses exclude the 12-byte preamble."""

    def __init__(self, data=b''):
        self.data = bytearray(data)

    def read(self, offset, size):
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise ValueError(f'Model range {offset}+{size} exceeds {len(self.data)} bytes')
        return bytes(self.data[offset:offset + size])

    def values(self, offset, fmt):
        return struct.unpack('<' + fmt, self.read(offset, struct.calcsize('<' + fmt)))

    def u32(self, offset):
        return self.values(offset, 'I')[0]

    def f32(self, offset):
        return self.values(offset, 'f')[0]

    def array(self, offset, count, fmt):
        step = struct.calcsize('<' + fmt)
        raw = self.read(offset, count * step) if count else b''
        return list(struct.iter_unpack('<' + fmt, raw))

    def text(self, offset, width=None):
        if width is None:
            end = self.data.find(0, offset)
            if end < offset:
                raise ValueError('Unterminated model string')
            width = end - offset + 1
        return self.read(offset, width).split(b'\x00', 1)[0].decode('latin1')

    def put(self, offset, fmt, *values):
        size = struct.calcsize('<' + fmt)
        old = self.values(offset, fmt)
        # Do not quiet NaNs or normalize negative zero during unrelated edits.
        if all((a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) for a, b in zip(old, values))):
            return
        self.data[offset:offset + size] = struct.pack('<' + fmt, *values)

    def text_at(self, offset, width, value):
        raw = value.encode('latin1')
        if len(raw) > width or b'\x00' in raw:
            raise ValueError(f'Model string must fit its {width}-byte field without embedded NULs')
        if self.text(offset, width) != value:
            self.data[offset:offset + width] = raw.ljust(width, b'\x00')

    def allocate(self, payload):
        self.data.extend(bytes(-len(self.data) % 4))
        offset = len(self.data)
        self.data.extend(payload)
        return offset

    def store(self, payload, original=0, original_size=0):
        """Reuse intact source storage; changed/shared arrays get their own allocation."""
        if len(payload) == original_size and 0 <= original <= len(self.data) - len(payload):
            if self.read(original, original_size) == payload:
                return original
        return self.allocate(payload) if payload else 0

def _state(value):
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, (str, bytes, int, type(None), bool)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple((_state(v) for v in value))
    return tuple(((k, _state(v)) for k, v in vars(value).items() if not k.startswith('_')))

def _flags(node):
    flags = 1 | (8 if node.camera else 0)
    for bit, attr in ((2, 'light'), (4, 'emitter'), (16, 'reference'), (32, 'mesh'), (64, 'skin'), (256, 'dangly'), (512, 'aabb'), (2048, 'saber')):
        if getattr(node, attr) is not None:
            flags |= bit
    if flags & (64 | 256 | 512 | 2048) and (not flags & 32):
        raise ValueError('A specialized mesh node requires a mesh header')
    return flags

def _node_size(flags, game):
    size = 80 + (340 if game == Game.K2 else 332) * bool(flags & 32)
    return size + sum((n for bit, n in _PART_SIZES.items() if flags & bit))

def _controller_width(kind, columns):
    if kind & 16777215 == 20 and columns == 2:
        return 1
    return (columns & 15) * (3 if columns & 16 else 1)

class MDLBinaryReader:

    def __init__(self, source: SOURCE_TYPES, offset=0, size=0, source_ext: SOURCE_TYPES | None=None, offset_ext=0, size_ext=0, game: Game | None=None):
        self._raw = _read_source(source, offset, size or 0)
        self._xraw = None if source_ext is None else _read_source(source_ext, offset_ext, size_ext or 0)
        self._game = game

    def load(self, auto_close=True) -> MDL:
        if len(self._raw) < 208 or self._raw[:4] != bytes(4):
            raise ValueError('Not a complete binary MDL resource')
        _, body_size, mdx_size = struct.unpack_from('<3I', self._raw)
        if body_size < 196 or body_size + 12 > len(self._raw):
            raise ValueError('MDL body size exceeds the supplied resource')
        if self._xraw is not None and mdx_size > len(self._xraw):
            raise ValueError('MDX size exceeds the supplied companion resource')
        self.b = _Buffer(self._raw[12:12 + body_size])
        self.x = None if self._xraw is None else _Buffer(self._xraw)
        fp = self.b.u32(0)
        games = [g for g, pair in _GEOMETRY_POINTERS.items() if fp == pair[0]]
        if self._game is None and (not games):
            raise ValueError('Unknown model layout; specify its game explicitly')
        self.game = games[0] if self._game is None else self._game
        m = MDL()
        m.game = m._source_game = self.game
        m._source_mdl, m._source_mdx = (self._raw, self._xraw)
        m.name, m.supermodel = (self.b.text(8, 32), self.b.text(136, 32))
        m.model_type, m.fog = (self.b.values(80, 'B')[0], bool(self.b.data[83]))
        m.bb_min = Vector3(*self.b.values(104, '3f'))
        m.bb_max = Vector3(*self.b.values(116, '3f'))
        m.radius, m.anim_scale = self.b.values(128, '2f')
        offsets = [r[0] for r in self.b.array(self.b.u32(184), self.b.u32(188), 'I')]
        self.names = [self.b.text(o) for o in offsets]
        m._source_names, m._source_name_offsets = (self.names[:], offsets)
        self.nodes = {}
        m.root = self._tree(self.b.u32(40))
        for offset, in self.b.array(self.b.u32(88), self.b.u32(92), 'I'):
            a = MDLAnimation()
            a._source = dict(offset=offset, header=self.b.read(offset, 136), origin=self._raw)
            a.name = self.b.text(offset + 8, 32)
            a.root_model = self.b.text(offset + 88, 32)
            a.anim_length, a.transition_length = self.b.values(offset + 80, '2f')
            for i in range(self.b.u32(offset + 124)):
                at = self.b.u32(offset + 120) + i * 36
                e = MDLEvent()
                e._source = self.b.read(at, 36)
                e.activation_time, e.name = (self.b.f32(at), self.b.text(at + 4, 32))
                a.events.append(e)
            a.root = self._tree(self.b.u32(offset + 40))
            a._source['actual_nodes'] = len(a.all_nodes())
            m.anims.append(a)
        m._source_node_count = len(m.all_nodes())
        return m

    def _tree(self, root):
        pending, seen = ([(root, None)], set())
        result = None
        while pending:
            offset, parent = pending.pop()
            if offset in seen:
                raise ValueError('Cycle or shared-parent node in MDL tree')
            seen.add(offset)
            node = self._node(offset)
            if parent is None:
                result = node
            else:
                parent.children.append(node)
            children = [a[0] for a in self.b.array(self.b.u32(offset + 44), self.b.u32(offset + 48), 'I')]
            pending.extend(((o, node) for o in reversed(children)))
        return result

    def _node(self, o):
        b = self.b
        flags, node_id, name_id = b.values(o, 'HHI')
        if flags & ~(1 | 2 | 4 | 8 | 16 | 32 | 64 | 256 | 512 | 2048):
            raise ValueError(f'Unsupported model node flags 0x{flags:04X}')
        if name_id >= len(self.names):
            raise ValueError('Node name index is outside the name table')
        n = MDLNode()
        n.node_id, n.name, n.camera = (node_id, self.names[name_id], bool(flags & 8))
        n.position = Vector3(*b.values(o + 16, '3f'))
        w, x, y, z = b.values(o + 28, '4f')
        n.orientation = Vector4(x, y, z, w)
        n._source = dict(offset=o, flags=flags, game=self.game, header=b.read(o, _node_size(flags, self.game)), origin=self._raw)
        p = o + 80
        if flags & 32:
            n.mesh = self._mesh(p)
            p += 340 if self.game == Game.K2 else 332
        for bit, cls, attr in ((2, MDLLight, 'light'), (4, MDLEmitter, 'emitter'), (16, MDLReference, 'reference'), (64, MDLSkin, 'skin'), (256, MDLDangly, 'dangly'), (512, MDLWalkmesh, 'aabb'), (2048, MDLSaber, 'saber')):
            if not flags & bit:
                continue
            part = cls()
            part._header = b.read(p, _PART_SIZES[bit])
            setattr(n, attr, part)
            if bit == 2:
                part.flare_radius = b.f32(p)
                part._unknown = b.read(b.u32(p + 4), b.u32(p + 8) * 4) if b.u32(p + 8) else b''
                part.flare_sizes = [r[0] for r in b.array(b.u32(p + 16), b.u32(p + 20), 'f')]
                part.flare_positions = [r[0] for r in b.array(b.u32(p + 28), b.u32(p + 32), 'f')]
                part.flare_color_shifts = [Vector3(*r) for r in b.array(b.u32(p + 40), b.u32(p + 44), '3f')]
                part.flare_textures = [b.text(r[0]) for r in b.array(b.u32(p + 52), b.u32(p + 56), 'I')]
                for i, a in enumerate(('light_priority', 'ambient_only', 'dynamic_type', 'affect_dynamic', 'shadow', 'flare', 'fading_light')):
                    setattr(part, a, b.u32(p + 64 + 4 * i))
            elif bit == 4:
                # Fixed disk emitter layout; unknown flags/bytes remain in the header.
                part.dead_space, part.blast_radius, part.blast_length = b.values(p, '3f')
                part.branch_count = b.u32(p + 12)
                part.control_point_smoothing = b.f32(p + 16)
                part.x_grid, part.y_grid = b.values(p + 20, '2I')
                for off, width, a in ((28, 32, 'update'), (60, 32, 'render'), (92, 32, 'blend'), (124, 32, 'texture'), (156, 16, 'chunk_name')):
                    setattr(part, a, b.text(p + off, width))
                for off, a in ((172, 'two_sided_texture'), (176, 'loop'), (180, 'render_order'), (184, 'frame_blender')):
                    setattr(part, a, b.u32(p + off))
                part.depth_texture = b.text(p + 188, 32)
                part.flags = b.u32(p + 220)
            elif bit == 16:
                part.model, part.reattachable = (b.text(p, 32), bool(b.u32(p + 32)))
            elif bit == 64:
                part.bonemap = [r[0] for r in b.array(b.u32(p + 20), b.u32(p + 24), 'f')]
                part.qbones = [Vector4(*r) for r in b.array(b.u32(p + 28), b.u32(p + 32), '4f')]
                part.tbones = [Vector3(*r) for r in b.array(b.u32(p + 40), b.u32(p + 44), '3f')]
                part._unknown = b.read(b.u32(p + 52), b.u32(p + 56) * 4) if b.u32(p + 56) else b''
                part.bone_indices = b.values(p + 64, '16H')
                if self.x is not None:
                    weights, indices = b.values(p + 12, '2I')
                    if n.mesh._source['vertex_count'] and (weights + 16 > n.mesh.mdx_stride or indices + 16 > n.mesh.mdx_stride):
                        raise ValueError('Skin attributes exceed the MDX vertex stride')
                    for i in range(n.mesh._source['vertex_count']):
                        at = n.mesh._source['mdx_offset'] + i * n.mesh.mdx_stride
                        v = MDLBoneVertex()
                        v.vertex_weights = self.x.values(at + weights, '4f')
                        v.vertex_indices = self.x.values(at + indices, '4f')
                        part.vertex_bones.append(v)
                part._vertex_snapshot = _state(part.vertex_bones)
            elif bit == 256:
                part.constraints = [r[0] for r in b.array(b.u32(p), b.u32(p + 4), 'f')]
                part.displacement, part.tightness, part.period = b.values(p + 12, '3f')
                part.vertices = [Vector3(*r) for r in b.array(b.u32(p + 24), n.mesh._source['vertex_count'], '3f')] if b.u32(p + 24) else []
            elif bit == 512:
                part.root = self._aabb(b.u32(p)) if b.u32(p) else None
            elif bit == 2048:
                for off, attr, fmt, cls in ((0, 'vertices', '3f', Vector3), (4, 'texcoords', '2f', Vector2), (8, 'normals', '3f', Vector3)):
                    setattr(part, attr, [cls(*r) for r in b.array(b.u32(p + off), 176, fmt)] if b.u32(p + off) else [])
            part._source_offset = p
            part._snapshot = _state(part)
            p += _PART_SIZES[bit]
        cp, count, _, dp, dc, _ = b.values(o + 56, '6I')
        raw_data = b.read(dp, dc * 4) if dc else b''
        n._source['controller_data'] = raw_data
        for i in range(count):
            h = b.read(cp + i * 16, 16)
            kind, part, rows, key, value, columns, padding = struct.unpack('<IHHHHB3s', h)
            width = _controller_width(kind, columns)
            if rows and (key + rows > dc or value + rows * width > dc):
                raise ValueError('Controller keys or values exceed its declared data array')
            keys = [r[0] for r in b.array(dp + key * 4, rows, 'f')]
            vr = b.read(dp + value * 4, rows * width * 4) if rows * width else b''
            c = MDLController()
            c.controller_type, c.columns, c.part_offset, c.padding = (kind, columns, part, padding)
            if kind & 16777215 == 20 and columns == 2:
                values = [list(Vector4.from_compressed(r[0])) for r in struct.iter_unpack('<I', vr)]
            else:
                values = [list(r) for r in struct.iter_unpack('<' + 'f' * width, vr)] if width else [[] for _ in keys]
            c.rows = [MDLControllerRow(t, v) for t, v in zip(keys, values)]
            c._source = dict(header=h, values=vr, keys=b.read(dp + key * 4, rows * 4), snapshot=_state(c), row_values=_state([row.data for row in c.rows]))
            n.controllers.append(c)
        return n

    def _aabb(self, offset):
        records, pending = ({}, [offset])
        while pending:
            o = pending.pop()
            if o in records:
                raise ValueError('Cycle or shared child in model AABB tree')
            a = MDLAABB()
            a._offset, a._header = (o, self.b.read(o, 40))
            a.bb_min = Vector3(*self.b.values(o, '3f'))
            a.bb_max = Vector3(*self.b.values(o + 12, '3f'))
            a.face, a.plane = self.b.values(o + 32, '2I')
            records[o] = a
            pending.extend((x for x in self.b.values(o + 24, '2I') if x))
        for o, a in records.items():
            l, r = self.b.values(o + 24, '2I')
            a.left, a.right = (records.get(l), records.get(r))
        return records[offset]

    def _mesh(self, p):
        b, x = (self.b, self.x)
        m = MDLMesh()
        size = 340 if self.game == Game.K2 else 332
        h = b.read(p, size)
        m._source = dict(offset=p, header=h, vertex_count=b.values(p + 304, 'H')[0], mdx_offset=b.u32(p + size - 8), vertices_offset=b.u32(p + size - 4))
        for off, attr in ((20, 'bb_min'), (32, 'bb_max'), (48, 'average')):
            setattr(m, attr, Vector3(*b.values(p + off, '3f')))
        m.radius, m.area = (b.f32(p + 44), b.f32(p + (324 if self.game == Game.K2 else 316)))
        m.diffuse, m.ambient = (Color.from_bgr_vector3(Vector3(*b.values(p + o, '3f'))) for o in (60, 72))
        m.transparency_hint = b.u32(p + 84)
        m.texture_1, m.texture_2 = (b.text(p + 88, 32), b.text(p + 120, 32))
        m.saber_unknowns = b.read(p + 224, 8)
        m.animate_uv = bool(b.u32(p + 232))
        m.uv_direction_x, m.uv_direction_y, m.uv_jitter, m.uv_jitter_speed = b.values(p + 236, '4f')
        m.mdx_stride, m.mdx_bitmap = b.values(p + 252, '2I')
        m.mdx_offsets = list(b.values(p + 260, '11I'))
        m.texture_count = b.values(p + 306, 'H')[0]
        for off, attr in enumerate(('has_lightmap', 'rotate_texture', 'background_geometry', 'shadow', 'beaming', 'render')):
            setattr(m, attr, bool(b.data[p + 308 + off]))
        if self.game == Game.K2:
            m.dirt_enabled = bool(b.data[p + 314])
            m.dirt_texture = b.values(p + 316, 'H')[0]
            m.dirt_coordinate_space = b.values(p + 318, 'H')[0]
            m.hide_in_hologram = bool(b.data[p + 320])
        counts = [r[0] for r in b.array(b.u32(p + 176), b.u32(p + 180), 'I')]
        pointers = [r[0] for r in b.array(b.u32(p + 188), b.u32(p + 192), 'I')]
        if len(counts) != len(pointers):
            raise ValueError('Mesh index counts and pointers disagree')
        m.indices = [[r[0] for r in b.array(o, n, 'H')] for o, n in zip(pointers, counts)]
        m.inverted_counters = [r[0] for r in b.array(b.u32(p + 200), b.u32(p + 204), 'I')]
        for r in b.array(b.u32(p + 8), b.u32(p + 12), '4fI6H'):
            f = MDLFace()
            f.normal = Vector3(*r[:3])
            f.coefficient = r[3]
            f.material = r[4]
            f.a1, f.a2, f.a3, f.v1, f.v2, f.v3 = r[5:]
            m.faces.append(f)
        count = m._source['vertex_count']
        m.vertex_positions = [Vector3(*r) for r in b.array(m._source['vertices_offset'], count, '3f')] if m._source['vertices_offset'] else []
        if x is not None:
            m.mdx_data = x.read(m._source['mdx_offset'], count * m.mdx_stride) if count * m.mdx_stride else b''
            for bit, slot, width, cls, attr in ((1, 0, 3, Vector3, 'vertex_positions'), (32, 1, 3, Vector3, 'vertex_normals'), (2, 3, 2, Vector2, 'vertex_uv1'), (4, 4, 2, Vector2, 'vertex_uv2')):
                if m.mdx_bitmap & bit and m.mdx_offsets[slot] != 4294967295:
                    if count and m.mdx_offsets[slot] + width * 4 > m.mdx_stride:
                        raise ValueError('Mesh attribute exceeds the MDX vertex stride')
                    values = [cls(*x.values(m._source['mdx_offset'] + i * m.mdx_stride + m.mdx_offsets[slot], str(width) + 'f')) for i in range(count)]
                    if attr != 'vertex_positions' or not m.vertex_positions:
                        setattr(m, attr, values)
        m._source['snapshot'] = _state(m)
        m._source['positions'] = _state(m.vertex_positions)
        m._source['normals'] = _state(m.vertex_normals)
        m._source['uv1'] = _state(m.vertex_uv1)
        m._source['uv2'] = _state(m.vertex_uv2)
        m._source['faces'] = _state(m.faces)
        m._source['face_vertices'] = tuple(((f.v1, f.v2, f.v3) for f in m.faces))
        m._source['indices'] = _state(m.indices)
        return m

class MDLBinaryWriter:
    """Serialize full model records; source arrays remain intact until explicitly edited."""

    def __init__(self, mdl: MDL, target: TARGET_TYPES, target_ext: TARGET_TYPES | None=None):
        self._mdl, self._target, self._target_ext = (mdl, target, target_ext)
        self.game = mdl.game

    def encode(self):
        m = self._mdl
        original = m._source_mdl
        body_size = struct.unpack_from('<I', original, 4)[0] if original else 196
        self.b = _Buffer(original[12:12 + body_size] if original else bytes(196))
        self.x = None if m._source_mdx is None else _Buffer(m._source_mdx)
        self.same_game = not original or self.game == m._source_game
        self.original = original
        self._sources = {}
        self._array_source = self.b
        self.graphs = [m] + m.anims
        self.nodes, self.parents, self.roots, self.node_offsets, self.name_ids = ([], {}, {}, {}, {})
        occupied = set()
        # Name-table identity is independent of node identity. Existing indices are
        # retained; equal display names do not combine nodes or child references.
        self.names = m._source_names[:]
        name_offsets = m._source_name_offsets[:]
        for graph in self.graphs:
            nodes = graph.all_nodes()
            for node in nodes:
                if id(node) in self.node_offsets:
                    raise ValueError('Base and animation graphs must contain independent node occurrences')
                flags = _flags(node)
                src = node._source
                own = src.get('origin') == original and bool(original)
                old = src.get('offset', 0)
                reuse = own and self.same_game and (flags == src.get('flags')) and (old not in occupied)
                at = old if reuse else self.b.allocate(bytes(_node_size(flags, self.game)))
                occupied.add(at)
                self.node_offsets[id(node)] = at
                self.nodes.append(node)
                self.roots[id(node)] = graph
                for child in node.children:
                    self.parents[id(child)] = node
                ni = struct.unpack_from('<I', src['header'], 4)[0] if own else -1
                if not (0 <= ni < len(self.names) and self.names[ni] == node.name):
                    ni = len(self.names)
                    self.names.append(node.name)
                    raw = node.name.encode('latin1')
                    if b'\x00' in raw:
                        raise ValueError('Embedded NUL in node name')
                    name_offsets.append(self.b.allocate(raw + b'\x00'))
                self.name_ids[id(node)] = ni
        # Preserve animation-header addresses where possible; all offsets are body-relative.
        self.anim_offsets = {}
        for a in m.anims:
            old = a._source.get('offset', 0) if a._source.get('origin') == original else 0
            at = old if old and old not in occupied else self.b.allocate(bytes(136))
            self.anim_offsets[id(a)] = at
            occupied.add(at)
        for n in self.nodes:
            self._write_node(n)
        for a in m.anims:
            self._write_animation(a)
        b = self.b
        self._array_source = b
        if not original or not self.same_game:
            b.put(0, '2I', *_GEOMETRY_POINTERS[self.game])
        b.text_at(8, 32, m.name)
        b.put(40, 'II', self.node_offsets[id(m.root)], b.u32(44) + len(m.all_nodes()) - m._source_node_count if original else len(m.all_nodes()))
        b.put(76, 'B', 2)
        b.put(80, 'B', m.model_type)
        self._boolean(b, 83, m.fog)
        self._array_header(b, 88, [self.anim_offsets[id(a)] for a in m.anims], 'I')
        b.put(104, '3f', *m.bb_min)
        b.put(116, '3f', *m.bb_max)
        b.put(128, '2f', m.radius, m.anim_scale)
        b.text_at(136, 32, m.supermodel)
        b.put(168, 'I', self.node_offsets[id(m.root)])
        xraw = None if self.x is None else bytes(self.x.data)
        stream_changed = not original or xraw is not None and xraw != m._source_mdx
        xsize = len(xraw or b'') if stream_changed else struct.unpack_from('<I', original, 8)[0]
        if stream_changed:
            b.put(176, 'I', xsize)
        self._array_header(b, 184, name_offsets, 'I')
        tail = original[12 + body_size:] if original else b''
        return (struct.pack('<3I', 0, len(b.data), xsize) + bytes(b.data) + tail, xraw)

    @staticmethod
    def _boolean(b, offset, value):
        if bool(b.data[offset]) != bool(value):
            b.put(offset, 'B', int(bool(value)))

    def _source_buffer(self, origin):
        if not origin or origin == self.original:
            return self.b
        key = id(origin)
        if key not in self._sources:
            size = struct.unpack_from('<I', origin, 4)[0]
            self._sources[key] = _Buffer(origin[12:12 + size])
        return self._sources[key]

    def _array_header(self, header, at, values, fmt):
        old, count, allocated = header.values(at, '3I')
        rows = [v if isinstance(v, (tuple, list)) else (v,) for v in values]
        if not rows and (not count):
            return old
        raw = b''.join((struct.pack('<' + fmt, *r) for r in rows))
        size = count * struct.calcsize('<' + fmt)
        # Same float values must not canonicalize NaN payloads in unchanged arrays.
        source = self._array_source
        if size == len(raw) and old + size <= len(source.data):
            prev = source.array(old, count, fmt)
            if _state(prev) == _state(rows):
                if source is self.b:
                    return old
                raw = source.read(old, size)
        dest = self.b.store(raw, old, size)
        header.put(at, 'I', dest)
        if count != len(rows):
            header.put(at + 4, 'II', len(rows), len(rows))
        return dest

    def _blob(self, header, at, data, count=None, step=1, triple=True):
        old = header.u32(at)
        n = header.u32(at + 4) if count is None else count
        if not data and (not n):
            return old
        dest = self.b.store(data, old, n * step)
        header.put(at, 'I', dest)
        if count is None:
            new = len(data) // step
            if new != n:
                header.put(at + 4, 'II' if triple else 'I', *[new] * (2 if triple else 1))
        return dest

    def _write_node(self, n):
        src = n._source
        self._array_source = self._source_buffer(src.get('origin'))
        flags = _flags(n)
        size = _node_size(flags, self.game)
        h = _Buffer(bytes(size))
        if src:
            h.data[:80] = src['header'][:80]
        h.put(0, 'HHI', flags, n.node_id if n.node_id >= 0 else self.nodes.index(n), self.name_ids[id(n)])
        graph = self.roots[id(n)]
        root_ref = 0 if graph is self._mdl else self.anim_offsets[id(graph)]
        h.put(8, 'I', root_ref)
        parent = self.parents.get(id(n))
        h.put(12, 'I', self.node_offsets[id(parent)] if parent is not None else 0)
        h.put(16, '3f', *n.position)
        h.put(28, '4f', n.orientation.w, n.orientation.x, n.orientation.y, n.orientation.z)
        self._array_header(h, 44, [self.node_offsets[id(c)] for c in n.children], 'I')
        self._controllers(n, h)
        p = 80
        if n.mesh is not None:
            self._write_mesh(n, h, p)
            p += 340 if self.game == Game.K2 else 332
        for bit, attr in ((2, 'light'), (4, 'emitter'), (16, 'reference'), (64, 'skin'), (256, 'dangly'), (512, 'aabb'), (2048, 'saber')):
            part = getattr(n, attr)
            if part is None:
                continue
            self._write_part(n, part, h, p, bit)
            p += _PART_SIZES[bit]
        at = self.node_offsets[id(n)]
        self.b.data[at:at + len(h.data)] = h.data

    def _controllers(self, n, h):
        data = bytearray(n._source.get('controller_data', b''))
        records = []
        for c in n.controllers:
            cs = c._source
            if cs and cs['snapshot'] == _state(c):
                # Preserve packed quaternion bits, Bézier tangents, padding,
                # shared data and key offsets exactly when this record is intact.
                records.append(cs['header'])
                continue
            cols = c.columns
            if cols is None:
                cols = len(c.rows[0].data) if c.rows else 0
            values_unchanged = bool(cs) and cols == cs['header'][12] and _state([row.data for row in c.rows]) == cs['row_values']
            if c.controller_type & 16777215 == 20 and cols == 2 and not values_unchanged:
                # Explicit quaternion-value edits use four floats; key-only edits
                # retain the original packed values and column mode.
                cols = 4
            width = _controller_width(c.controller_type, cols)
            if not values_unchanged and any((len(r.data) != width for r in c.rows)):
                raise ValueError('Controller row width does not match its mode')
            key = len(data) // 4
            data.extend(b''.join((struct.pack('<f', r.time) for r in c.rows)))
            val = len(data) // 4
            data.extend(cs['values'] if values_unchanged else b''.join((struct.pack('<' + 'f' * width, *r.data) for r in c.rows)))
            records.append(struct.pack('<IHHHHB3s', c.controller_type, c.part_offset, len(c.rows), key, val, cols, c.padding))
        self._blob(h, 56, b''.join(records), step=16)
        self._blob(h, 68, bytes(data), step=4)

    def _write_mesh(self, n, h, p):
        m = n.mesh
        s = m._source
        src = _Buffer(s.get('header', bytes(340 if self.game == Game.K2 else 332)))
        base = bytes(src.data)
        if len(base) == 332 and self.game == Game.K2:
            base = base[:316] + bytes(8) + base[316:]
        elif len(base) == 340 and self.game == Game.K1:
            base = base[:316] + base[324:]
        h.data[p:p + len(base)] = base
        if not s or n._source.get('game') != self.game or (_flags(n) ^ n._source.get('flags', 0)) & (32 | 64 | 256):
            kind = 64 if n.skin else 256 if n.dangly else 32
            h.put(p, '2I', *_MESH_POINTERS[self.game][kind])
        for off, attr in ((20, 'bb_min'), (32, 'bb_max'), (48, 'average')):
            h.put(p + off, '3f', *getattr(m, attr))
        h.put(p + 44, 'f', m.radius)
        h.put(p + (324 if self.game == Game.K2 else 316), 'f', m.area)
        h.put(p + 60, '3f', m.diffuse.b, m.diffuse.g, m.diffuse.r)
        h.put(p + 72, '3f', m.ambient.b, m.ambient.g, m.ambient.r)
        h.put(p + 84, 'I', m.transparency_hint)
        h.text_at(p + 88, 32, m.texture_1)
        h.text_at(p + 120, 32, m.texture_2)
        h.data[p + 224:p + 232] = bytes(m.saber_unknowns)
        if bool(h.u32(p + 232)) != m.animate_uv:
            h.put(p + 232, 'I', int(m.animate_uv))
        h.put(p + 236, '4f', m.uv_direction_x, m.uv_direction_y, m.uv_jitter, m.uv_jitter_speed)
        for i, a in enumerate(('has_lightmap', 'rotate_texture', 'background_geometry', 'shadow', 'beaming', 'render')):
            self._boolean(h, p + 308 + i, getattr(m, a))
        h.put(p + 306, 'H', m.texture_count)
        if self.game == Game.K2:
            self._boolean(h, p + 314, m.dirt_enabled)
            h.put(p + 316, 'HH', m.dirt_texture, m.dirt_coordinate_space)
            self._boolean(h, p + 320, m.hide_in_hologram)
        count = len(m.vertex_positions)
        if s and (not m.vertex_positions) and (not s['vertices_offset']) and (self.x is None):
            count = s['vertex_count']
        if any((v >= count for f in m.faces for v in (f.v1, f.v2, f.v3))):
            raise ValueError('Mesh face references a vertex outside its array')
        rows = [(*f.normal, f.coefficient, int(f.material), f.a1, f.a2, f.a3, f.v1, f.v2, f.v3) for f in m.faces]
        self._array_header(h, p + 8, rows, '4fI6H')
        groups = m.indices
        faces_changed = not s or s['face_vertices'] != tuple(((f.v1, f.v2, f.v3) for f in m.faces))
        if not groups and m.faces or (faces_changed and (not s or s['indices'] == _state(groups))):
            groups = [[v for f in m.faces for v in (f.v1, f.v2, f.v3)]]
        old_ptrs = [a[0] for a in self._array_source.array(h.u32(p + 188), h.u32(p + 192), 'I')]
        old_counts = [a[0] for a in self._array_source.array(h.u32(p + 176), h.u32(p + 180), 'I')]
        ptrs = []
        for i, g in enumerate(groups):
            if any((v < 0 or v >= count for v in g)):
                raise ValueError('Mesh draw index is outside its vertex array')
            raw = struct.pack('<' + 'H' * len(g), *g)
            ptrs.append(self.b.store(raw, old_ptrs[i] if i < len(old_ptrs) else 0, old_counts[i] * 2 if i < len(old_counts) else 0))
        self._array_header(h, p + 176, [len(g) for g in groups], 'I')
        self._array_header(h, p + 188, ptrs, 'I')
        self._array_header(h, p + 200, m.inverted_counters, 'I')
        size = 340 if self.game == Game.K2 else 332
        vp = p + size - 4
        xp = p + size - 8
        if m.vertex_positions and (not s or s['vertices_offset'] or _state(m.vertex_positions) != s['positions']):
            raw = b''.join((struct.pack('<3f', *v) for v in m.vertex_positions))
            h.put(vp, 'I', self.b.store(raw, h.u32(vp), s.get('vertex_count', 0) * 12))
        h.put(p + 304, 'H', count)
        self._mesh_stream(n, h, p, xp, count)

    def _mesh_stream(self, n, h, p, xp, count):
        m = n.mesh
        s = m._source
        positions_changed = not s or _state(m.vertex_positions) != s['positions']
        normals_changed = not s or _state(m.vertex_normals) != s['normals']
        uv1_changed = not s or _state(m.vertex_uv1) != s['uv1']
        uv2_changed = not s or _state(m.vertex_uv2) != s['uv2']
        skin_changed = n.skin is not None and getattr(n.skin, '_vertex_snapshot', None) != _state(n.skin.vertex_bones)
        if self.x is None:
            if positions_changed or normals_changed or uv1_changed or uv2_changed or skin_changed:
                raise ValueError('Editing model vertex streams requires the MDX companion')
            return
        raw = bytearray(m.mdx_data)
        stride = m.mdx_stride
        bitmap = m.mdx_bitmap
        offsets = m.mdx_offsets[:]
        arrays = [(1, 0, 3, m.vertex_positions, positions_changed), (32, 1, 3, m.vertex_normals, normals_changed), (2, 3, 2, m.vertex_uv1, uv1_changed), (4, 4, 2, m.vertex_uv2, uv2_changed)]
        if not s:
            raw = bytearray()
            stride = 0
            bitmap = 0
            offsets = [4294967295] * 11
        for bit, slot, width, values, changed in arrays:
            if not changed:
                continue
            if values is None:
                bitmap &= ~bit
                offsets[slot] = 4294967295
                continue
            if len(values) != count:
                raise ValueError('Mesh vertex streams have different row counts')
            if offsets[slot] == 4294967295 or not bitmap & bit:
                old_stride = stride
                offsets[slot] = stride
                stride += width * 4
                bitmap |= bit
                new = bytearray(stride * count)
                for i in range(min(count, len(raw) // old_stride if old_stride else 0)):
                    new[i * stride:i * stride + old_stride] = raw[i * old_stride:(i + 1) * old_stride]
                raw = new
            elif offsets[slot] + width * 4 > stride:
                raise ValueError('Invalid vertex attribute offset')
            needed = count * stride
            if len(raw) < needed:
                raw.extend(bytes(needed - len(raw)))
            for i, v in enumerate(values):
                struct.pack_into('<' + str(width) + 'f', raw, i * stride + offsets[slot], *v)
        if n.skin is not None:
            sh = _Buffer(n.skin._header)
            wo, bo = sh.values(12, '2I')
            if not s:
                old = stride
                wo, bo = (stride, stride + 16)
                stride += 32
                new = bytearray(stride * count)
                for i in range(count):
                    new[i * stride:i * stride + old] = raw[i * old:(i + 1) * old]
                raw = new
            if skin_changed or not s:
                if len(n.skin.vertex_bones) != count:
                    raise ValueError('Skin weight rows do not match its vertices')
                if wo + 16 > stride or bo + 16 > stride:
                    raise ValueError('Skin attribute exceeds the MDX stride')
                for i, v in enumerate(n.skin.vertex_bones):
                    struct.pack_into('<4f', raw, i * stride + wo, *v.vertex_weights)
                    struct.pack_into('<4f', raw, i * stride + bo, *v.vertex_indices)
            n_offsets = (wo, bo)
            self._skin_offsets = getattr(self, '_skin_offsets', {})
            self._skin_offsets[id(n)] = n_offsets
        if s and len(raw) != count * stride:
            raw = raw[:count * stride]
        if not s and count:
            raw.extend(bytes(stride))
        at = self.x.store(bytes(raw), s.get('mdx_offset', 0), len(m.mdx_data))
        h.put(xp, 'I', at)
        h.put(p + 252, '2I', stride, bitmap)
        h.put(p + 260, '11I', *offsets)

    def _write_part(self, n, part, h, p, bit):
        original = getattr(part, '_header', bytes(_PART_SIZES[bit]))
        h.data[p:p + _PART_SIZES[bit]] = original
        if bit == 2:
            h.put(p, 'f', part.flare_radius)
            self._blob(h, p + 4, part._unknown, step=4)
            self._array_header(h, p + 16, part.flare_sizes, 'f')
            self._array_header(h, p + 28, part.flare_positions, 'f')
            self._array_header(h, p + 40, [tuple(v) for v in part.flare_color_shifts], '3f')
            old = [r[0] for r in self._array_source.array(h.u32(p + 52), h.u32(p + 56), 'I')]
            strings = []
            for i, t in enumerate(part.flare_textures):
                raw = t.encode('latin1') + b'\x00'
                at = old[i] if i < len(old) else 0
                length = len(self._array_source.text(at).encode('latin1')) + 1 if at else 0
                strings.append(self.b.store(raw, at, length))
            self._array_header(h, p + 52, strings, 'I')
            for i, a in enumerate(('light_priority', 'ambient_only', 'dynamic_type', 'affect_dynamic', 'shadow', 'flare', 'fading_light')):
                h.put(p + 64 + 4 * i, 'I', getattr(part, a))
        elif bit == 4:
            h.put(p, '3f', part.dead_space, part.blast_radius, part.blast_length)
            h.put(p + 12, 'I', part.branch_count)
            h.put(p + 16, 'f', part.control_point_smoothing)
            h.put(p + 20, '2I', part.x_grid, part.y_grid)
            for off, width, a in ((28, 32, 'update'), (60, 32, 'render'), (92, 32, 'blend'), (124, 32, 'texture'), (156, 16, 'chunk_name'), (188, 32, 'depth_texture')):
                h.text_at(p + off, width, getattr(part, a))
            for off, a in ((172, 'two_sided_texture'), (176, 'loop'), (180, 'render_order'), (184, 'frame_blender'), (220, 'flags')):
                h.put(p + off, 'I', getattr(part, a))
        elif bit == 16:
            h.text_at(p, 32, part.model)
            if bool(h.u32(p + 32)) != part.reattachable:
                h.put(p + 32, 'I', int(part.reattachable))
        elif bit == 64:
            weights, bones = getattr(self, '_skin_offsets', {}).get(id(n), h.values(p + 12, '2I'))
            h.put(p + 12, '2I', weights, bones)
            self._blob(h, p + 20, b''.join((struct.pack('<f', v) for v in part.bonemap)), step=4, triple=False)
            self._array_header(h, p + 28, [tuple(v) for v in part.qbones], '4f')
            self._array_header(h, p + 40, [tuple(v) for v in part.tbones], '3f')
            self._blob(h, p + 52, part._unknown, step=4)
            h.put(p + 64, '16H', *part.bone_indices)
        elif bit == 256:
            self._array_header(h, p, part.constraints, 'f')
            h.put(p + 12, '3f', part.displacement, part.tightness, part.period)
            raw = b''.join((struct.pack('<3f', *v) for v in part.vertices))
            old = h.u32(p + 24)
            h.put(p + 24, 'I', self.b.store(raw, old, n.mesh._source.get('vertex_count', 0) * 12 if old else 0))
        elif bit == 512:
            h.put(p, 'I', self._write_aabb(part.root))
        elif bit == 2048:
            for off, a, width in ((0, 'vertices', 3), (4, 'texcoords', 2), (8, 'normals', 3)):
                values = getattr(part, a)
                if values and len(values) != 176:
                    raise ValueError('Saber arrays have 176 fixed records in this format')
                raw = b''.join((struct.pack('<' + str(width) + 'f', *v) for v in values))
                old = h.u32(p + off)
                h.put(p + off, 'I', self.b.store(raw, old, 176 * width * 4 if old else 0))

    def _write_aabb(self, root):
        if root is None:
            return 0
        offsets, seen = ({}, set())
        pending = [(root, False)]
        while pending:
            a, visited = pending.pop()
            if not visited:
                if id(a) in seen:
                    raise ValueError('Cycle or shared record in AABB tree')
                seen.add(id(a))
                pending.append((a, True))
                pending.extend(((c, False) for c in (a.right, a.left) if c is not None))
                continue
            h = _Buffer(a._header)
            h.put(0, '3f', *a.bb_min)
            h.put(12, '3f', *a.bb_max)
            h.put(24, '4I', offsets[id(a.left)] if a.left else 0, offsets[id(a.right)] if a.right else 0, a.face, a.plane)
            # Modified or copied trees must not overwrite a shared source tree.
            offsets[id(a)] = self.b.store(bytes(h.data), a._offset, 40 if a._offset else 0)
        return offsets[id(root)]

    def _write_animation(self, a):
        self._array_source = self._source_buffer(a._source.get('origin'))
        h = _Buffer(a._source.get('header', bytes(136)))
        if not a._source or self._array_source.u32(a._source['offset']) != _ANIMATION_POINTERS[self.game][0]:
            h.put(0, '2I', *_ANIMATION_POINTERS[self.game])
        h.text_at(8, 32, a.name)
        h.put(40, '2I', self.node_offsets[id(a.root)], h.u32(44) + len(a.all_nodes()) - a._source['actual_nodes'] if a._source else len(a.all_nodes()))
        h.put(76, 'B', 5)
        h.put(80, '2f', a.anim_length, a.transition_length)
        h.text_at(88, 32, a.root_model)
        events = []
        for e in a.events:
            eh = _Buffer(e._source or bytes(36))
            eh.put(0, 'f', e.activation_time)
            eh.text_at(4, 32, e.name)
            events.append(bytes(eh.data))
        self._blob(h, 120, b''.join(events), step=36)
        at = self.anim_offsets[id(a)]
        self.b.data[at:at + 136] = h.data

    def write(self, auto_close=True):
        mdl, mdx = self.encode()
        if self._target_ext is not None and mdx is None:
            raise ValueError('The MDX companion was not supplied when this model was loaded')
        _write_pair(self._target, mdl, self._target_ext, mdx, auto_close=auto_close)

def _write_pair(target, mdl, target_ext=None, mdx=None, *, auto_close=True):
    """Stage path outputs before either commit; roll back handled commit failures.

    Two files cannot be made power-loss atomic by filesystem rename. No output
    is opened until serialization succeeds, and a handled second commit error
    restores the first. In-memory/borrowed-stream targets have their own lifetime.
    """
    outputs = [(target, mdl)]
    if target_ext is not None:

        def storage(t):
            if isinstance(t, BinaryWriterBytearray):
                return t._ba
            if isinstance(t, BinaryWriterFile):
                return t._stream
            return t
        if storage(target_ext) is storage(target):
            raise ValueError('MDL and MDX require distinct targets')
        outputs.append((target_ext, mdx))
    paths = [Path(p) for p, _ in outputs if isinstance(p, (str, os.PathLike))]
    if paths and len(paths) != len(outputs):
        raise ValueError('Use two path targets or two memory/stream targets')
    if len(paths) == 2:
        a, b = paths
        if os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b)) or (a.exists() and b.exists() and os.path.samefile(a, b)):
            raise ValueError('MDL and MDX targets refer to the same file')
    if not paths:
        for p, data in outputs:
            if data is None:
                raise ValueError('MDX companion is unavailable')
            if isinstance(p, bytearray):
                p[:] = data
            elif isinstance(p, BinaryWriter):
                p.write_bytes(data)
            else:
                with BinaryWriter.to_auto(p) as writer:
                    writer.write_bytes(data)
        return
    staged = []
    backups = []
    committed = []
    recovery = []
    try:
        for p, (_, data) in zip(paths, outputs):
            if data is None:
                raise ValueError('MDX companion is unavailable')
            with tempfile.NamedTemporaryFile(dir=p.parent, prefix='.' + p.name.lower() + '.', suffix='.tmp', delete=False) as f:
                staged.append(Path(f.name))
                f.write(data)
                f.flush()
            if p.exists():
                shutil.copymode(p, staged[-1])
                with tempfile.NamedTemporaryFile(dir=p.parent, prefix='.' + p.name.lower() + '.', suffix='.bak', delete=False) as f:
                    backups.append(Path(f.name))
                shutil.copy2(p, backups[-1])
            else:
                backups.append(None)
        for i, p in enumerate(paths):
            os.replace(staged[i], p)
            committed.append(i)
    except BaseException:
        for i in reversed(committed):
            try:
                if backups[i] is not None:
                    os.replace(backups[i], paths[i])
                else:
                    paths[i].unlink()
            except OSError:
                # Keep the original backup available when restoration itself fails.
                recovery.append(backups[i])
        if recovery:
            raise OSError(f'Model pair restoration failed; originals retained at {recovery}')
        raise
    finally:
        for p in staged + backups:
            if p is not None and p not in recovery and p.exists():
                p.unlink()
