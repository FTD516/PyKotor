"""Retained GFF documents for editing specialized resource models.

A builder describes modeled values, not the complete input document. Only its
edits are applied to the retained document. Unmodeled fields, original types,
absent defaults and physical duplicate records remain untouched.
"""
from __future__ import annotations

import struct
import uuid

from collections import deque
from copy import deepcopy
from typing import Any, Callable, Hashable, Iterable, TypeVar

from pykotor.common.language import LocalizedString
from pykotor.common.misc import ResRef
from pykotor.resource.formats.gff import GFF, GFFFieldType, GFFList, GFFStruct


T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


def update_gff_membership(
    groups: Iterable[Iterable[T]],
    selected: Iterable[K],
    *,
    key: Callable[[T], K],
    create: Callable[[K], T],
) -> list[list[T]]:
    """Apply a membership selection without reordering retained occurrences.

    Each selected key retains every existing occurrence, including duplicate
    records, in its original group. New keys fill removed slots in group/list
    order; excess additions go to the final group. Selection order affects only
    new keys, and repeated selections do not create extra records.

    Retained objects are returned unchanged. New members are built by ``create``
    and do not inherit the removed record's source identity or hidden fields.
    Unfilled vacancies are removed, so deletions without enough replacements
    necessarily shift later indices. Inputs are not modified.

    This is an explicit membership-editor operation, not a serialization rule.
    Ordered editors should insert, delete, or reorder their lists directly.
    """
    original = [list(group) for group in groups]
    selected_keys = dict.fromkeys(selected)
    if not original:
        if selected_keys:
            raise ValueError("Cannot add members without a destination list.")
        return []

    present = {key(item) for group in original for item in group}
    additions = deque(create(value) for value in selected_keys if value not in present)
    result: list[list[T]] = []
    for group in original:
        updated: list[T] = []
        for item in group:
            if key(item) in selected_keys:
                updated.append(item)
            elif additions:
                updated.append(additions.popleft())
        result.append(updated)
    result[-1].extend(additions)
    return result


class GFFInteger(int):
    """An integer list element retaining its source record identity."""


def remember_gff_struct(model: Any, source: GFFStruct) -> None:
    if source._source_key is None:
        source._source_key = uuid.uuid4().hex
    model._gff_key = source._source_key
    model._gff_record = deepcopy(source)


def bind_gff_struct(target: GFFStruct, model: Any, *, preserve_id: bool = True) -> None:
    """Bind a row to its source occurrence, retaining its struct ID after a move.

    Equipment slots use the ID as an editable key and pass ``preserve_id=False``.
    """
    target._source_key = getattr(model, "_gff_key", None)
    target._source_record = getattr(model, "_gff_record", None)
    if preserve_id and target._source_record is not None:
        target.struct_id = target._source_record.struct_id


def remember_gff(model: Any, source: GFF) -> None:
    # Snapshot before attaching the source pair, so snapshots cannot recurse.
    snapshot = deepcopy(model)
    model._gff_source = (deepcopy(source), snapshot)


def _same_value(kind: GFFFieldType, left: Any, right: Any) -> bool:
    if kind is GFFFieldType.Struct:
        return _same_struct(left, right)
    if kind is GFFFieldType.List:
        return len(left) == len(right) and all(_same_struct(a, b) for a, b in zip(left, right))
    if kind is GFFFieldType.ResRef:
        return left.to_bytes(fixed_width=False) == right.to_bytes(fixed_width=False)
    if kind is GFFFieldType.LocalizedString:
        return left.stringref == right.stringref and left._substrings == right._substrings
    if kind in (GFFFieldType.Single, GFFFieldType.Double):
        format_string = "<f" if kind is GFFFieldType.Single else "<d"
        return struct.pack(format_string, left) == struct.pack(format_string, right)
    return left == right


def _same_struct(left: GFFStruct, right: GFFStruct) -> bool:
    return (
        left._source_key == right._source_key
        and left.struct_id == right.struct_id
        and len(left) == len(right)
        and all(a == b and ta is tb and _same_value(ta, va, vb)
                for (a, ta, va), (b, tb, vb) in zip(left, right))
    )


def _index_structs(root: GFFStruct) -> dict[str, GFFStruct]:
    result: dict[str, GFFStruct] = {}
    pending = [root]
    while pending:
        node = pending.pop()
        if node._source_key is not None:
            result[node._source_key] = node
        for _, kind, value in node:
            if kind is GFFFieldType.Struct:
                pending.append(value)
            elif kind is GFFFieldType.List:
                pending.extend(value)
    return result


def preserve_gff(model: Any, generated: GFF, build: Callable[[Any], GFF]) -> GFF:
    source_pair = getattr(model, "_gff_source", None)
    if source_pair is None:
        return generated
    original, snapshot = source_pair
    baseline = build(snapshot)
    original_rows = _index_structs(original.root)
    baseline_rows = _index_structs(baseline.root)

    def merge_list(source: GFFList, before: GFFList | None, after: GFFList) -> GFFList:
        result = GFFList()
        if before is not None and len(before) == len(after) and all(row._source_key is None for row in (*before, *after)):
            # Fixed positional lists can model only part of a source list, or
            # contain implicit default rows that were never stored in the file.
            last_edit = max((i for i, (a, b) in enumerate(zip(before, after)) if not _same_struct(a, b)), default=-1)
            for index in range(max(len(source), last_edit + 1)):
                if index >= len(after):
                    result._structs.append(deepcopy(source.at(index)))
                elif index >= len(source):
                    result._structs.append(deepcopy(after.at(index)))
                else:
                    result._structs.append(merge_struct(source.at(index), before.at(index), after.at(index)))
            return result
        for index, row in enumerate(after):
            key = row._source_key
            if key is not None:
                source_row = original_rows.get(key)
                if source_row is None:
                    source_row = getattr(row, "_source_record", None)
                old_row = baseline_rows.get(source_row._source_key if source_row is not None else key)
            else:
                old_row = before.at(index) if before is not None and len(before) == len(after) else None
                if old_row is not None and old_row._source_key is not None:
                    old_row = None
                source_row = source.at(index) if old_row is not None else None
            if source_row is None:
                result._structs.append(deepcopy(row))
            else:
                result._structs.append(merge_struct(source_row, old_row, row))
        return result

    def merge_struct(source: GFFStruct, before: GFFStruct | None, after: GFFStruct) -> GFFStruct:
        result = deepcopy(source)
        if after._source_key is not None:
            result._source_key = after._source_key
        if before is not None and after.struct_id != before.struct_id:
            result.struct_id = after.struct_id
        if before is not None:
            for label in before._fields:
                if not after.exists(label):
                    result.remove(label)
        for label, field in after._fields.items():
            kind, value = field.field_type(), field.value()
            previous = before._fields.get(label) if before is not None else None
            original_field = source._fields.get(label)
            matching = original_field is not None and original_field.field_type() is kind
            old_value = None
            if previous is not None and previous.field_type() is kind:
                old_value = previous.value()
                if _same_value(kind, old_value, value):
                    continue
                if kind is GFFFieldType.LocalizedString and matching:
                    edited = deepcopy(original_field.value())
                    edited.stringref = value.stringref
                    for string_id in old_value._substrings.keys() - value._substrings.keys():
                        edited._substrings.pop(string_id, None)
                        edited._substring_bytes.pop(string_id, None)
                    for string_id, text in value._substrings.items():
                        if old_value._substrings.get(string_id) != text:
                            edited._substrings[string_id] = text
                            edited._substring_bytes.pop(string_id, None)
                    value = edited
            if kind is GFFFieldType.Struct:
                source_value = original_field.value() if matching else GFFStruct(value.struct_id)
                value = merge_struct(source_value, old_value, value)
            elif kind is GFFFieldType.List:
                source_value = original_field.value() if matching else GFFList()
                value = merge_list(source_value, old_value, value)
            # New/imported rows keep unmodeled source fields as well.
            result._set_field(label, kind, deepcopy(value))
        return result

    result = GFF(original.content)
    result._binary_data = original._binary_data
    result.root = merge_struct(original.root, baseline.root, generated.root)
    return result
