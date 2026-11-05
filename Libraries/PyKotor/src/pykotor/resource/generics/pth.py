from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

from pykotor.resource.formats.gff.gff_data import GFFFieldType
from pykotor.resource.generics._gff import (remember_gff, preserve_gff, remember_gff_struct, bind_gff_struct)
from pykotor.common.geometry import Vector2
from pykotor.common.misc import Game
from pykotor.resource.formats.gff import GFF, GFFContent, GFFList, GFFStruct, read_gff, write_gff
from pykotor.resource.formats.gff.gff_auto import bytes_gff
from pykotor.resource.type import ResourceType
from utility.error_handling import format_exception_with_variables

if TYPE_CHECKING:
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


class PTH:
    """Stores the path data for a module."""

    BINARY_TYPE = ResourceType.PTH

    def __init__(
        self,
    ):
        self._points: list[Vector2] = []
        self._connections: list[PTHEdge] = []
        self._connection_rows: list[PTHEdge] = []

    def __iter__(
        self,
    ):
        yield from self._points

    def __len__(
        self,
    ):
        return len(self._points)

    def __getitem__(
        self,
        item: int,
    ) -> Vector2:
        return self._points[item]

    def add(
        self,
        x: float,
        y: float,
    ) -> int:
        self._points.append(Vector2(x, y))
        return len(self._points) - 1

    def remove(
        self,
        index: int,
    ):
        self._points.pop(index)

        self._connections = [x for x in self._connections if index not in {x.source, x.target}]

        for connection in self._connections:
            connection.source = connection.source - 1 if connection.source > index else connection.source
            connection.target = connection.target - 1 if connection.target > index else connection.target

    def get(
        self,
        index: int,
    ) -> Vector2 | None:
        try:
            return self._points[index]
        except Exception as e:
            print(format_exception_with_variables(e, message="This exception has been suppressed."))
        return None

    def find(
        self,
        point: Vector2,
    ) -> int | None:
        return self._points.index(point)

    def connect(
        self,
        source: int,
        target: int,
    ):
        self._connections.append(PTHEdge(source, target))

    def disconnect(
        self,
        source: int,
        target: int,
    ):
        for edge in copy(self._connections):
            tuple_check: tuple[int, int] = (source, target)
            has_source: bool = edge.source in tuple_check
            has_target: bool = edge.target in tuple_check
            if has_source and has_target:
                self._connections.remove(edge)

    def is_connected(
        self,
        source: int,
        target: int,
    ) -> bool:
        return any(x for x in self._connections if x == PTHEdge(source, target))

    def outgoing(
        self,
        source: int,
    ) -> list[PTHEdge]:
        return [connection for connection in self._connections if connection.source == source]

    def incoming(
        self,
        target: int,
    ) -> list[PTHEdge]:
        return [connection for connection in self._connections if connection.target == target]


class PTHEdge:
    def __init__(
        self,
        source: int,
        target: int,
    ):
        self.source: int = source
        self.target: int = target

    def __repr__(
        self,
    ):
        return f"{self.__class__.__name__}(source={self.source}, target={self.target})"

    def __eq__(
        self,
        other: PTHEdge | object,
    ):
        if self is other:
            return True
        if isinstance(other, PTHEdge):
            return self.source == other.source and self.target == other.target
        return NotImplemented


def construct_pth(
    gff: GFF,
) -> PTH:
    pth = PTH()

    connections_list: GFFList = gff.root.acquire("Path_Conections", GFFList(), field_type=GFFFieldType.List)
    for connection_struct in connections_list:
        connection = PTHEdge(-1, connection_struct.acquire("Destination", 0, field_type=GFFFieldType.UInt32))
        remember_gff_struct(connection, connection_struct)
        pth._connection_rows.append(connection)

    for point_struct in gff.root.acquire("Path_Points", GFFList(), field_type=GFFFieldType.List):
        connections: int = point_struct.acquire("Conections", 0, field_type=GFFFieldType.UInt32)
        first_connection: int = point_struct.acquire("First_Conection", 0, field_type=GFFFieldType.UInt32)
        x: float = point_struct.acquire("X", 0.0, field_type=GFFFieldType.Single)
        y: float = point_struct.acquire("Y", 0.0, field_type=GFFFieldType.Single)

        source: int = pth.add(x, y)
        remember_gff_struct(pth[source], point_struct)

        for i in range(first_connection, first_connection + connections):
            connection = copy(pth._connection_rows[i])
            connection.source = source
            pth._connections.append(connection)

    remember_gff(pth, gff)
    return pth


def dismantle_pth(
    pth: PTH,
    game: Game = Game.K2,
    *,
    use_deprecated: bool = True,
) -> GFF:
    gff = GFF(GFFContent.PTH)

    connections_list: GFFList = gff.root.set_list("Path_Conections", GFFList())
    points_list: GFFList = gff.root.set_list("Path_Points", GFFList())

    def connection_struct(edge: PTHEdge) -> GFFStruct:
        row = GFFStruct(3)
        bind_gff_struct(row, edge)
        row.set_uint32("Destination", edge.target)
        return row

    # Keep the stored row order, including rows not referenced by any point.
    connections_list._structs = [connection_struct(edge) for edge in pth._connection_rows]
    blocks: list[tuple[int | None, int, list[PTHEdge]]] = []
    for i, point in enumerate(pth):
        record = getattr(point, "_gff_record", None)
        first = None if record is None else record.acquire("First_Conection", 0, field_type=GFFFieldType.UInt32)
        count = 0 if record is None else record.acquire("Conections", 0, field_type=GFFFieldType.UInt32)
        blocks.append((first, count, pth.outgoing(i)))

    # Grow the original tail before allocating other new blocks after it.
    allocation_order = sorted(range(len(blocks)), key=lambda i: not (
        blocks[i][0] is not None
        and blocks[i][0] + blocks[i][1] == len(pth._connection_rows)
        and len(blocks[i][2]) > blocks[i][1]
    ))
    placements = [0] * len(blocks)
    for i in allocation_order:
        first, count, outgoings = blocks[i]
        rows = [connection_struct(edge) for edge in outgoings]
        if first is None:
            first = len(connections_list)
            connections_list._structs.extend(rows)
        else:
            exclusive = all(
                i == j or first + count <= other_first or other_first + other_count <= first
                for j, (other_first, other_count, _) in enumerate(blocks)
                if other_first is not None and other_count
            )
            same_prefix = all(
                getattr(edge, "_gff_key", None) == pth._connection_rows[first + n]._gff_key
                and edge.target == pth._connection_rows[first + n].target
                for n, edge in enumerate(outgoings[:count])
            )
            fits = len(rows) <= count or first + count == len(connections_list)
            if fits and (exclusive or same_prefix):
                connections_list._structs[first:first + len(rows)] = rows
            else:
                # Connections for a point must be contiguous. Relocate only
                # that block when it cannot grow without moving another one.
                first = len(connections_list)
                connections_list._structs.extend(rows)
        placements[i] = first

    for i, point in enumerate(pth):
        point_struct = points_list.add(2)
        bind_gff_struct(point_struct, point)
        point_struct.set_uint32("Conections", len(blocks[i][2]))
        point_struct.set_uint32("First_Conection", placements[i])
        point_struct.set_single("X", point.x)
        point_struct.set_single("Y", point.y)

    return preserve_gff(pth, gff, lambda original: dismantle_pth(original, game=game, use_deprecated=use_deprecated))


def read_pth(
    source: SOURCE_TYPES,
    offset: int = 0,
    size: int | None = None,
) -> PTH:
    gff: GFF = read_gff(source, offset, size)
    return construct_pth(gff)


def write_pth(
    pth: PTH,
    target: TARGET_TYPES,
    game: Game = Game.K2,
    file_format: ResourceType = ResourceType.GFF,
    *,
    use_deprecated: bool = True,
):
    gff: GFF = dismantle_pth(pth, game, use_deprecated=use_deprecated)
    write_gff(gff, target, file_format)


def bytes_pth(
    pth: PTH,
    game: Game = Game.K2,
    file_format: ResourceType = ResourceType.GFF,
    *,
    use_deprecated: bool = True,
) -> bytes:
    gff: GFF = dismantle_pth(pth, game, use_deprecated=use_deprecated)
    return bytes_gff(gff, file_format)
