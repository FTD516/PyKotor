"""This module handles classes relating to editing TPC files."""

from __future__ import annotations

import itertools as tpc_itertools
import struct

from dataclasses import dataclass

from enum import IntEnum
from typing import NamedTuple

from pykotor.common.stream import BinaryReader
from pykotor.resource.type import ResourceType
from pykotor.resource.formats.txi import TXI


class TPCGetResult(NamedTuple):
    width: int
    height: int
    texture_format: TPCTextureFormat
    data: bytes


class TPCConvertResult(NamedTuple):
    width: int
    height: int
    data: bytearray


@dataclass
class TPCHeader:
    """Stored header fields; these are independent of interpreted frame mips."""

    compressed_size: int = 0
    alpha_mean_bits: int = 0
    width: int = 4
    height: int = 4
    encoding: int = 2
    mipmap_count: int = 1
    reserved: bytes = bytes(114)

    @property
    def alpha_mean(self) -> float:
        return struct.unpack("<f", struct.pack("<I", self.alpha_mean_bits))[0]

    @alpha_mean.setter
    def alpha_mean(self, value: float) -> None:
        self.alpha_mean_bits = struct.unpack("<I", struct.pack("<f", value))[0]

    def image_size(self) -> int:
        """Native resource extent, calculated before interpreting embedded TXI."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("TPC dimensions must be positive.")
        if self.encoding not in (1, 2, 4) or self.compressed_size and self.encoding == 1:
            raise ValueError("Unsupported TPC encoding.")
        if not self.compressed_size:
            bpp = {1: 1, 2: 3, 4: 4}[self.encoding]
            return sum((self.width >> level) * (self.height >> level) * bpp
                       for level in range(self.mipmap_count))
        faces = 6 if self.height // self.width == 6 else 1
        height = self.height // faces
        block = 8 if self.encoding == 2 else 16
        return faces * (self.compressed_size + sum(
            (((self.width >> level) + 3) // 4) * (((height >> level) + 3) // 4) * block
            for level in range(1, self.mipmap_count)))

    def to_bytes(self) -> bytes:
        self.image_size()
        if len(self.reserved) != 114:
            raise ValueError("The TPC reserved header area must contain 114 bytes.")
        return struct.pack("<IIHHBB", self.compressed_size, self.alpha_mean_bits,
                           self.width, self.height, self.encoding, self.mipmap_count) + self.reserved


@dataclass(frozen=True)
class TPCMipmap:
    """An image range relative to the stored image-data block."""

    width: int
    height: int
    offset: int
    size: int


@dataclass(frozen=True)
class TPCImage:
    """An ordinary image, cube face, or cycle frame and its own mip sequence."""

    kind: str
    index: int
    mipmaps: tuple[TPCMipmap, ...]


def _image_size(width: int, height: int, texture_format: TPCTextureFormat) -> int:
    if texture_format in (TPCTextureFormat.DXT1, TPCTextureFormat.DXT5):
        block = 8 if texture_format == TPCTextureFormat.DXT1 else 16
        return ((width + 3) // 4) * ((height + 3) // 4) * block
    if texture_format == TPCTextureFormat.Invalid:
        raise ValueError("Invalid texture format.")
    return width * height * texture_format.bytes_per_pixel()


class TPC:
    """Represents a TPC file.

    Attributes:
    ----------
        txi: Stores additional information regarding the texture.
    """

    BINARY_TYPE = ResourceType.TPC

    def __init__(self):
        self.header = TPCHeader()
        self._texture_format = TPCTextureFormat.RGB
        self._image_data = bytes(4 * 4 * 3)
        self.embedded_txi = TXI()
        self.external_txi: TXI | None = None
        self.external_txi_source: str | None = None
        from pykotor.resource.formats.tpc.io_tga import _DataTypes
        self.original_datatype_code = _DataTypes.NO_IMAGE_DATA

    @property
    def txi(self) -> str:
        """Embedded TXI text. External selection is available as effective_txi."""
        return self.embedded_txi.text

    @txi.setter
    def txi(self, value: str) -> None:
        self.embedded_txi = TXI(value)

    @property
    def effective_txi(self) -> TXI:
        # An existing empty external document deliberately wins.
        return self.external_txi if self.external_txi is not None else self.embedded_txi

    @property
    def image_data(self) -> bytes:
        """Complete stored image bytes, including padding and unused ranges."""
        return self._image_data

    @property
    def alpha_mean(self) -> float:
        return self.header.alpha_mean

    @alpha_mean.setter
    def alpha_mean(self, value: float) -> None:
        self.header.alpha_mean = value

    def format(self) -> TPCTextureFormat:
        return self._texture_format

    def is_compressed(self) -> bool:
        return self._texture_format in (TPCTextureFormat.DXT1, TPCTextureFormat.DXT5)

    def images(self) -> tuple[TPCImage, ...]:
        """Describe image ranges without moving, decoding, or repacking bytes.

        A short/invalid image range is reported when get() requests it, not by
        discarding data while loading. Raw cycle ranges reflect the renderer's
        shared-mip/overlapping-base interpretation; they are not tiled atlases.
        """
        settings = self.effective_txi.settings()
        width, height = self.header.width, self.header.height
        levels = self.header.mipmap_count
        kind, count, stride = "image", 1, 0
        if settings.texture["cube"]:
            kind, count, height = "face", 6, height // 6
            stride = sum(_image_size(max(1, width >> level), max(1, height >> level), self._texture_format)
                         for level in range(levels))
            if self.is_compressed():
                stride = self.header.compressed_size + sum(
                    _image_size(width >> level, height >> level, self._texture_format)
                    for level in range(1, levels))
        elif settings.controller == "cycle":
            nx, ny = settings.texture["numx"], settings.texture["numy"]
            if nx <= 0 or ny <= 0 or width // nx == 0 or height // ny == 0:
                raise ValueError("Cycle grid dimensions must produce positive frame dimensions.")
            kind, count = "frame", nx * ny
            width, height = width // nx, height // ny
            levels = width.bit_length() if self.is_compressed() or settings.texture["mipmap"] else 1
            if self.is_compressed():
                # The native cycle frame stride is a complete width-based chain,
                # even for rectangular frames and when mip uploads are disabled.
                stride = sum(_image_size(max(1, width >> level), max(1, width >> level), self._texture_format)
                             for level in range(width.bit_length()))
            elif not settings.texture["mipmap"]:
                stride_format = TPCTextureFormat.DXT1 if self.get_bytes_per_pixel() == 3 else TPCTextureFormat.DXT5
                stride = sum(_image_size(max(1, width >> level), max(1, width >> level), stride_format)
                             for level in range(width.bit_length()))
        if width <= 0 or height <= 0:
            raise ValueError("The selected texture layout has no image pixels.")
        images = []
        for index in range(count):
            offset = index * stride
            mipmaps = []
            for level in range(levels):
                w, h = max(1, width >> level), max(1, height >> level)
                size = _image_size(w, h, self._texture_format)
                mipmaps.append(TPCMipmap(w, h, offset, size))
                offset += size
            images.append(TPCImage(kind, index, tuple(mipmaps)))
        return tuple(images)

    def image_count(self) -> int:
        return len(self.images())

    def mipmap_count(self, image: int = 0) -> int:
        return len(self._image(image).mipmaps)

    def _image(self, index: int) -> TPCImage:
        images = self.images()
        if index < 0 or index >= len(images):
            raise IndexError("TPC image index is out of range.")
        return images[index]

    def _mipmap(self, mipmap: int, image: int = 0) -> TPCMipmap:
        mipmaps = self._image(image).mipmaps
        if mipmap < 0 or mipmap >= len(mipmaps):
            raise IndexError("TPC mipmap index is out of range.")
        return mipmaps[mipmap]

    def dimensions(self, image: int = 0) -> tuple[int, int]:
        mip = self._mipmap(0, image)
        return mip.width, mip.height

    def get(self, mipmap: int = 0, *, image: int = 0) -> TPCGetResult:
        mip = self._mipmap(mipmap, image)
        if mip.offset + mip.size > len(self._image_data):
            raise ValueError("The selected mipmap extends beyond the stored image data.")
        return TPCGetResult(mip.width, mip.height, self._texture_format,
                            self._image_data[mip.offset:mip.offset + mip.size])

    def replace_mipmap(self, data: bytes, mipmap: int = 0, *, image: int = 0) -> None:
        """Replace one same-sized range without dropping other faces or padding."""
        mip = self._mipmap(mipmap, image)
        if len(data) != mip.size or mip.offset + mip.size > len(self._image_data):
            raise ValueError("Replacement data must fit the selected stored mipmap.")
        for other in self.images():
            for level, candidate in enumerate(other.mipmaps):
                if other.index == image and level == mipmap:
                    continue
                begin = max(mip.offset, candidate.offset)
                end = min(mip.offset + mip.size, candidate.offset + candidate.size)
                if begin < end and data[begin - mip.offset:end - mip.offset] != self._image_data[begin:end]:
                    raise ValueError("This image range is shared; use an explicit repack to change its layout.")
        self._image_data = self._image_data[:mip.offset] + bytes(data) + self._image_data[mip.offset + mip.size:]

    def convert(
        self,
        convert_format: TPCTextureFormat,
        mipmap: int = 0,
        y_flip: bool | None = None,
        *,
        image: int = 0,
    ) -> TPCConvertResult:
        """Returns a tuple containing the width, height and data of the specified mipmap where the data returned is in the texture format specified.

        Args:
        ----
            convert_format: The format the texture data should be converted to.
            mipmap: The index of the mipmap.

        Returns:
        -------
            A tuple equal to (width, height, data)
        """
        width, height, _, raw_data = self.get(mipmap, image=image)
        if self._texture_format == convert_format and not y_flip:  # Is conversion needed?
            return TPCConvertResult(width, height, bytearray(raw_data))

        if y_flip:
            bytes_per_pixel = 0
            if self._texture_format == TPCTextureFormat.Greyscale:
                bytes_per_pixel = 1
            elif self._texture_format in {TPCTextureFormat.RGB, TPCTextureFormat.RGBA}:
                bytes_per_pixel = 4 if self._texture_format == TPCTextureFormat.RGBA else 3
            # If the image needs to be flipped and it's an uncompressed format
            if bytes_per_pixel > 0:
                raw_data = bytearray(self.flip_image_data(raw_data, width, height, bytes_per_pixel))
                y_flip = False

        if convert_format not in (TPCTextureFormat.Greyscale, TPCTextureFormat.RGB, TPCTextureFormat.RGBA):
            raise ValueError("Compression must be requested explicitly; convert() decodes pixels.")
        data: bytearray = bytearray(raw_data)
        if convert_format == TPCTextureFormat.Greyscale:
            if self._texture_format == TPCTextureFormat.DXT5:
                rgba_data = TPC._dxt5_to_rgba(raw_data, width, height)
                data = TPC._rgba_to_grey(rgba_data, width, height)
            elif self._texture_format == TPCTextureFormat.DXT1:
                rgba_data = TPC._dxt1_to_rgba(raw_data, width, height)
                data = TPC._rgba_to_grey(rgba_data, width, height)
            elif self._texture_format == TPCTextureFormat.RGBA:
                data = TPC._rgba_to_grey(raw_data, width, height)
            elif self._texture_format == TPCTextureFormat.RGB:
                rgba_data = TPC._rgb_to_rgba(raw_data, width, height)
                data = TPC._rgba_to_grey(rgba_data, width, height)

        if convert_format == TPCTextureFormat.RGBA:
            if self._texture_format == TPCTextureFormat.DXT5:
                data = TPC._dxt5_to_rgba(raw_data, width, height)
            elif self._texture_format == TPCTextureFormat.DXT1:
                data = TPC._dxt1_to_rgba(raw_data, width, height)
            elif self._texture_format == TPCTextureFormat.RGB:
                data = TPC._rgb_to_rgba(raw_data, width, height)
            elif self._texture_format == TPCTextureFormat.Greyscale:
                data = TPC._grey_to_rgba(raw_data, width, height)

        if convert_format == TPCTextureFormat.RGB:
            if self._texture_format == TPCTextureFormat.DXT5:
                data = TPC._dxt5_to_rgb(raw_data, width, height)
            elif self._texture_format == TPCTextureFormat.DXT1:
                data = TPC._dxt1_to_rgb(raw_data, width, height)
            elif self._texture_format == TPCTextureFormat.RGBA:
                data = TPC._rgba_to_rgb(raw_data, width, height)
            elif self._texture_format == TPCTextureFormat.Greyscale:
                rgba_data = TPC._grey_to_rgba(raw_data, width, height)
                data = TPC._rgba_to_rgb(rgba_data, width, height)

        if y_flip:
            data = bytearray(self.flip_image_data(data, width, height, convert_format.bytes_per_pixel()))
        return TPCConvertResult(width, height, data)

    def get_bytes_per_pixel(self):
        bytes_per_pixel = 0
        if self._texture_format == TPCTextureFormat.Greyscale:
            bytes_per_pixel = 1
        elif self._texture_format in {TPCTextureFormat.RGB, TPCTextureFormat.RGBA}:
            bytes_per_pixel = 4 if self._texture_format == TPCTextureFormat.RGBA else 3
        return bytes_per_pixel

    @staticmethod
    def flip_image_data(data: bytes | bytearray, width: int, height: int, bytes_per_pixel: int) -> bytes:
        """Flip the image data vertically."""
        flipped_data = bytearray(len(data))
        row_length = width * bytes_per_pixel

        for row in range(height):
            source_start = row * row_length
            source_end = source_start + row_length
            dest_start = (height - 1 - row) * row_length
            flipped_data[dest_start : dest_start + row_length] = data[source_start:source_end]

        return bytes(flipped_data)

    def set_single(
        self,
        width: int,
        height: int,
        data: bytes,
        texture_format: TPCTextureFormat,
    ):
        """Sets the texture data but only for a single mipmap.

        Args:
        ----
            width: The new width.
            height: The new height.
            data: The new texture data.
            texture_format: The texture format.
        """
        self.set_data(width, height, [data], texture_format)

    def set_data(self, width: int, height: int, mipmaps: list[bytes], texture_format: TPCTextureFormat):
        """Explicitly replace the complete texture with an ordinary image."""
        self.set_images(width, height, [mipmaps], texture_format)

    def set_images(self, width: int, height: int, images: list[list[bytes]], texture_format: TPCTextureFormat,
                   *, kind: str = "image", numx: int = 1, numy: int = 1,
                   declared_mipmap_count: int | None = None) -> None:
        """Explicitly create/repack ordinary, cube, or compressed cycle storage.

        width/height describe one face/frame, not the header's atlas dimensions.
        No stock-engine dimension, frame-count or square-frame cap is imposed.
        """
        if width <= 0 or height <= 0 or not images or not images[0]:
            raise ValueError("A texture needs positive dimensions and image data.")
        levels = len(images[0])
        if kind not in ("image", "face", "frame"):
            raise ValueError("TPC image kind must be image, face, or frame.")
        if kind == "image" and len(images) != 1 or kind == "face" and len(images) != 6:
            raise ValueError("An ordinary texture has one image; a cube has six faces.")
        if kind == "frame" and (numx <= 0 or numy <= 0 or len(images) != numx * numy):
            raise ValueError("Cycle image count must match the frame grid.")
        compressed = texture_format in (TPCTextureFormat.DXT1, TPCTextureFormat.DXT5)
        if kind == "frame" and not compressed:
            raise ValueError("Raw cycle textures store an atlas; use set_data() and an explicit cycle TXI.")
        encoding = {TPCTextureFormat.Greyscale: 1, TPCTextureFormat.RGB: 2, TPCTextureFormat.RGBA: 4,
                    TPCTextureFormat.DXT1: 2, TPCTextureFormat.DXT5: 4}[texture_format]
        chunks = []
        for mips in images:
            if len(mips) != levels:
                raise ValueError("Every image must have the same mip sequence.")
            for level, data in enumerate(mips):
                if len(data) != _image_size(max(1, width >> level), max(1, height >> level), texture_format):
                    raise ValueError("Mipmap byte count does not match its dimensions and format.")
            packed = b"".join(bytes(data) for data in mips)
            if kind == "frame":
                stride = sum(_image_size(max(1, width >> level), max(1, width >> level), texture_format)
                             for level in range(width.bit_length()))
                if len(packed) > stride:
                    raise ValueError("The supplied frame chain overlaps the next native frame range.")
                packed += bytes(stride - len(packed))
            chunks.append(packed)
        stored_width = width * numx if kind == "frame" else width
        stored_height = height * (6 if kind == "face" else numy if kind == "frame" else 1)
        declared = declared_mipmap_count if declared_mipmap_count is not None else (1 if kind == "frame" else levels)
        header = TPCHeader(0, self.header.alpha_mean_bits, stored_width, stored_height, encoding, declared, self.header.reserved)
        payload = b"".join(chunks)
        if compressed:
            faces = 6 if stored_height // stored_width == 6 else 1
            if len(payload) % faces:
                raise ValueError("Image size cannot be represented by this TPC header geometry.")
            lower = sum(_image_size(stored_width >> level, (stored_height // faces) >> level, texture_format)
                        for level in range(1, declared))
            header.compressed_size = len(payload) // faces - lower
            if header.compressed_size <= 0:
                raise ValueError("The declared mip count does not describe a compressed TPC payload.")
        if header.image_size() != len(payload):
            raise ValueError("The supplied mip data does not match the native resource extent.")
        header.to_bytes()
        self.header, self._texture_format, self._image_data = header, texture_format, payload
        if kind == "face":
            self.embedded_txi.set("cube", "1")
        elif kind == "frame":
            self.embedded_txi.set("cube", "0")
            self.embedded_txi.set("proceduretype", "cycle")
            self.embedded_txi.set("numx", str(numx))
            self.embedded_txi.set("numy", str(numy))
        # Existing metadata is retained; callers choose whether external settings
        # or earlier procedural directives should change during a repack.

    def _mipmap_size(self, mipmap: int, image: int = 0) -> tuple[int, int]:
        mip = self._mipmap(mipmap, image)
        return mip.width, mip.height

    @staticmethod
    def _calculate_color_indices(
        rgba_block: list[tuple[int, int, int, int]],
        c0: tuple[int, int, int],
        c1: tuple[int, int, int],
    ) -> int:
        """Calculate 2-bit indices for each pixel in a 4x4 block."""
        indices: int = 0
        for i, pixel in enumerate(rgba_block):
            r, g, b, _a = pixel
            dr0, dg0, db0 = r - c0[0], g - c0[1], b - c0[2]
            dr1, dg1, db1 = r - c1[0], g - c1[1], b - c1[2]
            distance0 = dr0**2 + dg0**2 + db0**2
            distance1 = dr1**2 + dg1**2 + db1**2
            index = 0 if distance0 < distance1 else 1
            indices |= index << (i * 2)
        return indices

    @staticmethod
    def _select_representative_colors(rgba_block: list[tuple[int, int, int, int]]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Select representative colors for DXT1 compression."""
        colors = sorted(rgba_block, key=lambda x: (x[0] << 16) + (x[1] << 8) + x[2])
        return colors[0][:3], colors[-1][:3]

    @staticmethod
    def rgba_to_dxt1(
        rgba_data: bytes,
        width: int,
        height: int,
    ) -> bytearray:
        """Convert RGBA data to DXT1 compressed format."""
        if width <= 0 or height <= 0 or len(rgba_data) != width * height * 4:
            raise ValueError("RGBA input does not match the image dimensions.")
        compressed_data = bytearray()
        for y, x in tpc_itertools.product(range(0, height, 4), range(0, width, 4)):
            rgba_block = []
            for dy, dx in tpc_itertools.product(range(4), range(4)):
                index = (min(y + dy, height - 1) * width + min(x + dx, width - 1)) * 4
                rgba_block.append(tuple(rgba_data[index:index + 4]))
            c0, c1 = TPC._select_representative_colors(rgba_block)
            c0_565 = TPC._rgb_to_rgba565(c0)
            c1_565 = TPC._rgb_to_rgba565(c1)
            indices = TPC._calculate_color_indices(rgba_block, c0, c1)
            compressed_data += c0_565.to_bytes(2, byteorder="little")
            compressed_data += c1_565.to_bytes(2, byteorder="little")
            compressed_data += indices.to_bytes(4, byteorder="little")
        return compressed_data

    @staticmethod
    def rgba_to_dxt5(rgba_data: bytes, width: int, height: int) -> bytearray:
        """Encode DXT5 alpha blocks alongside the existing DXT color encoder."""
        colors = TPC.rgba_to_dxt1(rgba_data, width, height)
        compressed = bytearray()
        for block, (y, x) in enumerate(tpc_itertools.product(range(0, height, 4), range(0, width, 4))):
            alpha = [rgba_data[(min(y + dy, height - 1) * width + min(x + dx, width - 1)) * 4 + 3]
                     for dy, dx in tpc_itertools.product(range(4), range(4))]
            high, low = max(alpha), min(alpha)
            # Constant alpha selects endpoint zero, including equal endpoints.
            palette = [high, low] + [((7 - index) * high + index * low) // 7 for index in range(1, 7)]
            indices = sum(min(range(8), key=lambda entry: abs(value - palette[entry])) << (index * 3)
                          for index, value in enumerate(alpha))
            compressed.extend((high, low))
            compressed.extend(indices.to_bytes(6, "little"))
            compressed.extend(colors[block * 8:block * 8 + 8])
        return compressed

    # region Convert to RGBA
    @staticmethod
    def _dxt_to_rgba(data: bytes, width: int, height: int, *, dxt5: bool) -> bytearray:
        block_size = 16 if dxt5 else 8
        expected = ((width + 3) // 4) * ((height + 3) // 4) * block_size
        if len(data) != expected:
            raise ValueError("Compressed mipmap has an incorrect block count.")
        result = bytearray(width * height * 4)
        position = 0
        for by in range(0, height, 4):
            for bx in range(0, width, 4):
                block = data[position:position + block_size]
                position += block_size
                alpha = [255] * 16
                color_offset = 0
                if dxt5:
                    a0, a1 = block[0], block[1]
                    palette = [a0, a1]
                    if a0 > a1:
                        palette.extend(((7 - i) * a0 + i * a1) // 7 for i in range(1, 7))
                    else:
                        palette.extend(((5 - i) * a0 + i * a1) // 5 for i in range(1, 5))
                        palette.extend((0, 255))
                    bits = int.from_bytes(block[2:8], "little")
                    alpha = [palette[(bits >> (3 * i)) & 7] for i in range(16)]
                    color_offset = 8
                c0, c1, indices = struct.unpack_from("<HHI", block, color_offset)
                def rgb(color):
                    r, g, b = (color >> 11) & 31, (color >> 5) & 63, color & 31
                    return ((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2))
                colors = [rgb(c0), rgb(c1)]
                if dxt5 or c0 > c1:
                    colors.extend(tuple(((3 - i) * colors[0][c] + i * colors[1][c]) // 3 for c in range(3)) for i in (1, 2))
                else:
                    colors.extend((tuple((colors[0][c] + colors[1][c]) // 2 for c in range(3)), (0, 0, 0)))
                for y in range(4):
                    for x in range(4):
                        i = 4 * y + x
                        code = (indices >> (2 * i)) & 3
                        if bx + x >= width or by + y >= height:
                            continue
                        a = 0 if not dxt5 and c0 <= c1 and code == 3 else alpha[i]
                        offset = ((by + y) * width + bx + x) * 4
                        result[offset:offset + 4] = bytes((*colors[code], a))
        return result

    @staticmethod
    def _dxt5_to_rgba(data: bytes, width: int, height: int) -> bytearray:
        return TPC._dxt_to_rgba(data, width, height, dxt5=True)

    @staticmethod
    def _dxt1_to_rgba(data: bytes, width: int, height: int) -> bytearray:
        return TPC._dxt_to_rgba(data, width, height, dxt5=False)

    @staticmethod
    def _rgb_to_rgba(
        data: bytes,
        width: int,
        height: int,
    ) -> bytearray:
        new_data = bytearray()
        rgb_reader = BinaryReader.from_bytes(data)

        for _ty, _x in tpc_itertools.product(range(height), range(width)):
            new_data.extend(
                [
                    rgb_reader.read_uint8(),
                    rgb_reader.read_uint8(),
                    rgb_reader.read_uint8(),
                    255,
                ],
            )

        return new_data

    @staticmethod
    def _grey_to_rgba(
        data: bytes,
        width: int,
        height: int,
    ) -> bytearray:
        new_data = bytearray()
        rgb_reader = BinaryReader.from_bytes(data)

        for _y, _x in tpc_itertools.product(range(height), range(width)):
            brightness = rgb_reader.read_uint8()
            new_data.extend([brightness, brightness, brightness, 255])

        return new_data

    # endregion

    # region Convert to Grey
    @staticmethod
    def _rgba_to_grey(
        data: bytes,
        width: int,
        height: int,
    ) -> bytearray:
        new_data = bytearray()
        rgb_reader = BinaryReader.from_bytes(data)

        for _y, _x in tpc_itertools.product(range(height), range(width)):
            r = rgb_reader.read_uint8()
            g = rgb_reader.read_uint8()
            b = rgb_reader.read_uint8()
            rgb_reader.read_uint8()
            highest = r
            if g > highest:
                highest = g
            if b > highest:
                highest = b
            new_data.extend([highest])

        return new_data

    # endregion

    # region Convert to RGB
    @staticmethod
    def _dxt5_to_rgb(data: bytes, width: int, height: int) -> bytearray:
        return TPC._rgba_to_rgb(TPC._dxt5_to_rgba(data, width, height), width, height)

    @staticmethod
    def _dxt1_to_rgb(data: bytes, width: int, height: int) -> bytearray:
        return TPC._rgba_to_rgb(TPC._dxt1_to_rgba(data, width, height), width, height)

    @staticmethod
    def _rgba_to_rgb(
        data: bytes,
        width: int,
        height: int,
    ) -> bytearray:
        new_data = bytearray()
        rgb_reader: BinaryReader = BinaryReader.from_bytes(data)

        for _y, _x in tpc_itertools.product(range(height), range(width)):
            new_data.extend(
                [
                    rgb_reader.read_uint8(),
                    rgb_reader.read_uint8(),
                    rgb_reader.read_uint8(),
                ],
            )
            rgb_reader.skip(1)

        return new_data

    # endregion

    @staticmethod
    def _rgba565_to_rgb888(
        color: int,
    ) -> int:
        blue = color & 0x1F
        green = (color >> 5) & 0x3F
        red = (color >> 11) & 0x1F
        return (blue << 3) + (green << 10) + (red << 19)

    @staticmethod
    def _rgb_to_rgba565(rgb: tuple[int, int, int]) -> int:
        """Convert an RGB tuple to 5:6:5 bit RGB format."""
        r, g, b = rgb
        return (r >> 3) << 11 | (g >> 2) << 5 | b >> 3

    @staticmethod
    def _interpolate(
        weight: float,
        color0: int,
        color1: int,
    ) -> int:
        """Interpolates between two colors.

        Args:
        ----
            weight: float - Blend factor between 0-1
            color0: int - First color
            color1: int - Second color

        Returns:
        -------
            int - Interpolated color

        Processing Logic:
        ----------------
            - Extract blue, green, red channels from each color
            - Interpolate each channel value based on weight
            - Reassemble and return new color.
        """
        color0_blue = color0 & 255
        color0_greed = (color0 >> 8) & 255
        color0_red = (color0 >> 16) & 255

        color1_blue = color1 & 255
        color1_greed = (color1 >> 8) & 255
        color1_red = (color1 >> 16) & 255

        blue = int(((1.0 - weight) * color0_blue) + (weight * color1_blue))
        green = int(((1.0 - weight) * color0_greed) + (weight * color1_greed))
        red = int(((1.0 - weight) * color0_red) + (weight * color1_red))

        return blue + (green << 8) + (red << 16)

    @staticmethod
    def _rgba565_to_rgb(
        color: int,
    ) -> tuple[int, int, int]:
        """Converts a 16-bit 565 RGB color to a tuple of 8-bit RGB values.

        Args:
        ----
            color: 16-bit 565 RGB color value

        Returns:
        -------
            tuple: tuple of (red, green, blue) color component values

        Processing Logic:
        ----------------
            - Extracts the blue component from the lowest 5 bits
            - Extracts the green component from bits 5-10
            - Extracts the red component from bits 11-15
            - Left shifts the components to scale from 5 or 6 bits to 8 bits.
        """
        blue = color & 0x1F
        green = (color >> 5) & 0x3F
        red = (color >> 11) & 0x1F
        return red << 3, green << 2, blue << 3

    @staticmethod
    def _interpolate_rgb(
        weight: float,
        color0: tuple[int, int, int],
        color1: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        color0_blue = color0[2]
        color0_greed = color0[1]
        color0_red = color0[0]

        color1_blue = color1[2]
        color1_greed = color1[1]
        color1_red = color1[0]

        blue = int(((1.0 - weight) * color0_blue) + (weight * color1_blue))
        green = int(((1.0 - weight) * color0_greed) + (weight * color1_greed))
        red = int(((1.0 - weight) * color0_red) + (weight * color1_red))

        return red, green, blue

    @staticmethod
    def _integer48(
        bytes48: bytes,
    ) -> int:
        return bytes48[0] + (bytes48[1] << 8) + (bytes48[2] << 16) + (bytes48[3] << 24) + (bytes48[4] << 32) + (bytes48[5] << 40)


class TPCTextureFormat(IntEnum):
    Invalid = -1
    Greyscale = 0
    RGB = 1
    RGBA = 2
    DXT1 = 3
    DXT5 = 4

    def bytes_per_pixel(self):
        bytes_per_pixel = 0
        if self == TPCTextureFormat.Greyscale:
            bytes_per_pixel = 1
        elif self in {TPCTextureFormat.RGB, TPCTextureFormat.RGBA}:
            bytes_per_pixel = 4 if self == TPCTextureFormat.RGBA else 3
        return bytes_per_pixel
