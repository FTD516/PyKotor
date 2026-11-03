from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from pykotor.common.language import Language
from pykotor.common.misc import ResRef
from pykotor.common.stream import BinaryReader
from pykotor.resource.formats.tlk.io_tlk import TLKHeader
from pykotor.resource.formats.tlk.tlk_data import TLK
from utility.system.path import Path

if TYPE_CHECKING:
    import os


class StringResult(NamedTuple):
    text: str
    sound: ResRef


class TLKData(NamedTuple):
    flags: int
    sound_resref: ResRef
    volume_variance: int
    pitch_variance: int
    text_offset: int
    text_length: int
    sound_length: float


class TalkTable:
    """Talktables are for read-only loading of stringrefs stored in a dialog.tlk file.

    Files are only opened when accessing a stored string, this means that strings are always up to date at
    the time of access as opposed to TLK objects which may be out of date with its source file.
    """

    def __init__(
        self,
        path: os.PathLike | str,
    ):
        self._path: Path = Path.pathify(path)

    def path(self) -> Path:
        return self._path

    def string(
        self,
        stringref: int,
    ) -> str:
        """Access a string from the tlk file.

        Args:
        ----
            stringref: The entry id.

        Returns:
        -------
            A string.
        """
        return self.batch([stringref])[stringref].text

    def sound(
        self,
        stringref: int,
    ) -> ResRef:
        """Access the sound ResRef from the tlk file.

        Args:
        ----
            stringref: The entry id.

        Returns:
        -------
            A ResRef.
        """
        return self.batch([stringref])[stringref].sound

    def _extract_common_tlk_data(
        self,
        reader: BinaryReader,
        stringref: int,
        header: TLKHeader,
    ) -> TLKData:
        reader.seek(20 + header.entry_size * stringref)

        return TLKData(
            flags=reader.read_uint32(),
            sound_resref=ResRef.from_bytes(reader.read_bytes(16)),
            volume_variance=reader.read_uint32(),
            pitch_variance=reader.read_uint32(),
            text_offset=reader.read_uint32(),
            text_length=reader.read_uint32(),
            sound_length=reader.read_single() if header.version == "V3.0" else 0.0,
        )

    def batch(
        self,
        stringrefs: list[int],
        *,
        include_missing: bool = True,
    ) -> dict[int, StringResult]:
        """Loads a list of strings and sound ResRefs from the specified list.

        This is all performed using a single file handle and should be used if loading multiple strings from the tlk file.

        Args:
        ----
            stringrefs: A list of stringref ints.
            include_missing: Include empty results for missing/excluded entries.
                False keeps those distinct from existing entries with empty text.

        Returns:
        -------
            Dictionary with stringref keys and Tuples (string, sound) values.
        """
        with BinaryReader.from_file(self._path) as reader:
            header = TLKHeader.read(reader)
            encoding = TLK.encoding_for_language(header.language_id)
            results: dict[int, StringResult] = {}
            for stringref in stringrefs:
                reference = stringref & 0xFFFFFFFF
                index = reference & 0xFFFFFF
                text = ""
                sound = ResRef.from_blank()
                if include_missing:
                    results[stringref] = StringResult(text, sound)
                if reference != 0xFFFFFFFF and index < header.string_count:
                    entry = self._extract_common_tlk_data(reader, index, header)
                    if not entry.flags & 0x8000:
                        if entry.flags & 1 and entry.text_length:
                            reader.seek(header.texts_offset + entry.text_offset)
                            data = reader.read_bytes(entry.text_length).split(b"\0", 1)[0]
                            text = data.decode(encoding, "surrogateescape")
                        if entry.flags & 2:
                            sound = entry.sound_resref
                        results[stringref] = StringResult(text, sound)
            return results

    def size(
        self,
    ) -> int:
        """Returns the number of entries in the talk table.

        Returns:
        -------
            The number of entries in the talk table.
        """
        with BinaryReader.from_file(self._path) as reader:
            return TLKHeader.read(reader).string_count

    def language(
        self,
    ) -> Language:
        """Returns the matching Language of the TLK file.

        Returns:
        -------
            The language of the TLK file.
        """
        language_id = self.language_id()
        return Language(language_id) if language_id in Language._value2member_map_ else Language.UNKNOWN

    def language_id(self) -> int:
        """Return the stored language ID, including IDs outside the enum."""
        with BinaryReader.from_file(self._path) as reader:
            return TLKHeader.read(reader).language_id
