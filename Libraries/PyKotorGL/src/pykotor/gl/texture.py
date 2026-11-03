"""Prepared texture images and context-owned GPU textures."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np

from OpenGL.GL import (
    GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST, GL_NEAREST_MIPMAP_NEAREST,
    GL_REPEAT, GL_RGB, GL_RGBA, GL_TEXTURE_2D, GL_TEXTURE_3D, GL_TEXTURE_CUBE_MAP,
    GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_BASE_LEVEL, GL_TEXTURE_MAX_LEVEL,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_R, GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T, GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE, glBindTexture,
    glGenTextures, glGetIntegerv, glPixelStorei, glTexImage2D, glTexImage3D,
    glTexParameteri, glTexSubImage2D, glGenerateMipmap,
)
from OpenGL.raw.GL.EXT.texture_compression_s3tc import GL_COMPRESSED_RGB_S3TC_DXT1_EXT, GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
from OpenGL.raw.GL.VERSION.GL_1_3 import glCompressedTexImage2D

from pykotor.gl.texture_controller import TextureController, TextureRandom, normal_mipmaps
from pykotor.resource.formats.tpc import TPC, TPCTextureFormat
from pykotor.resource.formats.txi import TXI, TXISettings


@dataclass(frozen=True)
class TextureLevel:
    width: int
    height: int
    format: TPCTextureFormat
    data: bytes

    @classmethod
    def rgba(cls, pixels: np.ndarray):
        height, width = pixels.shape[:2]
        return cls(width, height, TPCTextureFormat.RGBA, pixels.tobytes())


@dataclass
class TextureData:
    settings: TXISettings
    frames: tuple[tuple[TextureLevel, ...], ...]
    target: int
    controller: TextureController | None
    generate_mipmaps: bool
    transparent: bool
    alpha_mean: float
    height_map: bool

    @classmethod
    def prepare(cls, tpc: TPC, *, image: int | None = None, dirty=(0, 16, 0, 0), seed=1):
        """Decode controller/alpha data in the worker, leaving GL calls to upload."""
        settings = tpc.effective_txi.settings(game=tpc.header.game)
        texture_settings = settings.texture
        height_map = texture_settings["isbumpmap"] == 1 and tpc.format() == TPCTextureFormat.Greyscale
        unprocessed = tpc.original_datatype_code != 0
        file_range = texture_settings["filerange"]
        if file_range > 0 and not texture_settings["cube"]:
            texture_settings["numy"] = file_range
        cube = image is None and texture_settings["cube"]
        cycle = image is None and settings.controller == "cycle"
        procedural = image is None and settings.controller not in (None, "cycle")
        target = GL_TEXTURE_CUBE_MAP if cube else GL_TEXTURE_2D
        controller = None
        generate = False
        frames = []
        base_images = []

        def corrected(pixels):
            if texture_settings["gamma"] != 1:
                values = np.power(pixels.astype(np.float32) / np.float32(255), np.float32(1 / texture_settings["gamma"]))
                pixels = np.trunc(values * np.float32(255)).astype(np.uint8)
            return pixels

        sheet = unprocessed and (cube or texture_settings["numx"] * texture_settings["numy"] != 1) or file_range > 0
        if sheet:
            # TGA contact sheets are unprocessed images; TPC's native frame
            # storage describes a different layout. Decode an ordinary view.
            view = copy(tpc)
            view.embedded_txi, view.external_txi = TXI(), None
            width, height, pixels = view.convert(TPCTextureFormat.RGBA)
            raster = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
            if texture_settings["cube"]:
                face_height = height // 6
                faces = [raster[index * face_height:(index + 1) * face_height] for index in range(6)]
                nx, ny = texture_settings["numx"], texture_settings["numy"]
                fw, fh = width // nx, face_height // ny
                for fy in range(ny):
                    for fx in range(nx):
                        ordered = [None] * 6
                        mapping = zip((1, 0, 2, 3, 4, 5), (3, 1, 0, 2, 2, 0)) if unprocessed else zip(range(6), (0,) * 6)
                        for index, (destination, rotation) in enumerate(mapping):
                            tile = faces[index][fy * fh:(fy + 1) * fh, fx * fw:(fx + 1) * fw]
                            ordered[destination] = np.rot90(tile, -rotation).copy()
                        base_images.extend(ordered)
            else:
                nx, ny = texture_settings["numx"], texture_settings["numy"]
                fw, fh = width // nx, height // ny
                base_images = [raster[y * fh:(y + 1) * fh, x * fw:(x + 1) * fw].copy()
                               for y in range(ny) for x in range(nx)]
            base_images = [corrected(pixels) for pixels in base_images]
            if image is not None:
                base_images = [base_images[image]]
            frames = [tuple(TextureLevel.rgba(level) for level in normal_mipmaps(
                pixels, texture_settings["bumpmapscaling"], texture_settings["mipmap"]))
                if height_map else (TextureLevel.rgba(pixels),) for pixels in base_images]
            generate = texture_settings["mipmap"] and not height_map
        else:
            view = tpc
            if texture_settings["numx"] * texture_settings["numy"] != 1 and not texture_settings["cube"] and settings.controller != "cycle":
                view = copy(tpc)
                view.embedded_txi, view.external_txi = TXI(bytes(tpc.effective_txi)), None
                view.embedded_txi.set("proceduretype", "cycle")
            indices = range(view.image_count()) if cube or cycle else (0 if image is None else image,)
            for index in indices:
                width, height, rgba = view.convert(TPCTextureFormat.RGBA, image=index)
                base = corrected(np.frombuffer(rgba, dtype=np.uint8).reshape(height, width, 4))
                base_images.append(base)
                if height_map:
                    normals = normal_mipmaps(base, texture_settings["bumpmapscaling"], texture_settings["mipmap"])
                    frames.append(tuple(TextureLevel.rgba(pixels) for pixels in normals))
                elif procedural:
                    frames.append((TextureLevel.rgba(base),))
                else:
                    levels = []
                    for mip in range(view.mipmap_count(index) if texture_settings["mipmap"] else 1):
                        width, height, fmt, pixels = view.get(mip, image=index)
                        if texture_settings["gamma"] != 1 or fmt in (TPCTextureFormat.Greyscale, TPCTextureFormat.PVRTC4_RGB, TPCTextureFormat.PVRTC4_RGBA):
                            width, height, rgba = view.convert(TPCTextureFormat.RGBA, mip, image=index)
                            levels.append(TextureLevel.rgba(corrected(np.frombuffer(rgba, dtype=np.uint8).reshape(height, width, 4))))
                        else:
                            levels.append(TextureLevel(width, height, fmt, bytes(pixels)))
                    frames.append(tuple(levels))

        if procedural or cycle:
            controller = TextureController(settings, base_images[0], frames=len(frames) // 6 if cube else len(frames),
                                           dirty=dirty, random=TextureRandom(seed), height_map=height_map)
            if procedural:
                if controller.image.ndim == 4:
                    target = GL_TEXTURE_3D
                    frames = [(TextureLevel(controller.width, controller.height, TPCTextureFormat.RGBA, controller.image.tobytes()),)]
                elif height_map:
                    frames = [tuple(TextureLevel.rgba(pixels) for pixels in normal_mipmaps(
                        controller.image, texture_settings["bumpmapscaling"], texture_settings["mipmap"]))]
                else:
                    frames = [(TextureLevel.rgba(controller.image),)]
                generate = texture_settings["mipmap"] and target != GL_TEXTURE_3D and not height_map
        if target == GL_TEXTURE_3D:
            generate = False
        alpha = np.concatenate([pixels[..., 3].reshape(-1) for pixels in base_images])
        transparent = bool(np.any(alpha < 255)) and not texture_settings["isbumpmap"]
        if procedural and not texture_settings["isbumpmap"]:
            transparent = settings.controller not in ("life", "ringtexdistort", "dirty", "dirty2", "dirty3")
        mean = float(alpha.astype(np.float32).mean() / 255)
        if texture_settings["alphamean"] >= 0:
            mean = texture_settings["alphamean"]
        return cls(settings, tuple(frames), target, controller, generate, transparent, mean, height_map)


class Texture:
    def __init__(self, tex_id: int):
        self._id = int(tex_id)
        self.ids = (self._id,)
        self.target = GL_TEXTURE_2D
        self.settings = TXISettings()
        self.controller = None
        self.generate_mipmaps = False
        self.transparent = False
        self.alpha_mean = 1.0
        self.width = self.height = 64
        self.height_map = False

    @classmethod
    def from_tpc(cls, tpc: TPC, *, image: int | None = None, dirty=(0, 16, 0, 0), seed=1):
        return cls.from_data(TextureData.prepare(tpc, image=image, dirty=dirty, seed=seed))

    @classmethod
    def from_data(cls, data: TextureData):
        count = len(data.frames) // 6 if data.target == GL_TEXTURE_CUBE_MAP else 1 if data.target == GL_TEXTURE_3D else len(data.frames)
        ids = tuple(int(value) for value in np.atleast_1d(glGenTextures(count)))
        texture = cls(ids[0])
        texture.ids, texture.target = ids, data.target
        texture.settings, texture.controller = data.settings, data.controller
        texture.height_map = data.height_map
        texture.generate_mipmaps = data.generate_mipmaps
        texture.transparent, texture.alpha_mean = data.transparent, data.alpha_mean
        texture.width, texture.height = data.frames[0][0].width, data.frames[0][0].height
        alignment = glGetIntegerv(GL_UNPACK_ALIGNMENT)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        for index, levels in enumerate(data.frames):
            glBindTexture(data.target, ids[index // 6] if data.target == GL_TEXTURE_CUBE_MAP else ids[index] if data.target == GL_TEXTURE_2D else ids[0])
            destination = GL_TEXTURE_CUBE_MAP_POSITIVE_X + index % 6 if data.target == GL_TEXTURE_CUBE_MAP else data.target
            for level, pixels in enumerate(levels):
                cls._upload(destination, level, pixels)
            maximum = max(texture.width, texture.height).bit_length() - 1 if data.generate_mipmaps else len(levels) - 1
            settings = data.settings.texture
            if data.generate_mipmaps and (data.target != GL_TEXTURE_CUBE_MAP or index % 6 == 5):
                glGenerateMipmap(data.target)
            base = min(maximum, max(0, settings["downsamplemin"]))
            maximum = max(base, min(maximum, settings["downsamplemax"]))
            glTexParameteri(data.target, GL_TEXTURE_BASE_LEVEL, base)
            glTexParameteri(data.target, GL_TEXTURE_MAX_LEVEL, maximum)
            clamp = settings["clamp"]
            glTexParameteri(data.target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE if data.target == GL_TEXTURE_CUBE_MAP or clamp & 1 else GL_REPEAT)
            glTexParameteri(data.target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE if data.target == GL_TEXTURE_CUBE_MAP or clamp & 2 else GL_REPEAT)
            if data.target != GL_TEXTURE_2D:
                glTexParameteri(data.target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE if data.target == GL_TEXTURE_CUBE_MAP else GL_REPEAT)
            filtered = GL_LINEAR if settings["filter"] else GL_NEAREST
            mip_filtered = GL_LINEAR_MIPMAP_LINEAR if settings["filter"] else GL_NEAREST_MIPMAP_NEAREST
            glTexParameteri(data.target, GL_TEXTURE_MAG_FILTER, filtered)
            glTexParameteri(data.target, GL_TEXTURE_MIN_FILTER, mip_filtered if settings["mipmap"] and data.target != GL_TEXTURE_3D else filtered)
        glPixelStorei(GL_UNPACK_ALIGNMENT, alignment)
        return texture

    @staticmethod
    def _upload(target, level, pixels):
        if target == GL_TEXTURE_3D:
            glTexImage3D(target, level, GL_RGBA, pixels.width, pixels.height, pixels.width, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data)
        elif pixels.format in (TPCTextureFormat.DXT1, TPCTextureFormat.DXT5):
            fmt = GL_COMPRESSED_RGB_S3TC_DXT1_EXT if pixels.format == TPCTextureFormat.DXT1 else GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
            glCompressedTexImage2D(target, level, fmt, pixels.width, pixels.height, 0, len(pixels.data), pixels.data)
        else:
            fmt = GL_RGB if pixels.format == TPCTextureFormat.RGB else GL_RGBA
            glTexImage2D(target, level, fmt, pixels.width, pixels.height, 0, fmt, GL_UNSIGNED_BYTE, pixels.data)

    def advance(self, frame: int, dt: float):
        if self.controller is None:
            return
        changed = self.controller.advance(frame, dt)
        self._id = self.ids[self.controller.index]
        if changed:
            self.use()
            alignment = glGetIntegerv(GL_UNPACK_ALIGNMENT)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            pixels = self.controller.image
            if self.height_map:
                levels = normal_mipmaps(pixels, self.settings.texture["bumpmapscaling"], self.settings.texture["mipmap"])
                for level, image in enumerate(levels):
                    self._upload(self.target, level, TextureLevel.rgba(image))
            else:
                glTexSubImage2D(self.target, 0, 0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, pixels)
                if self.generate_mipmaps:
                    glGenerateMipmap(self.target)
            glPixelStorei(GL_UNPACK_ALIGNMENT, alignment)

    @classmethod
    def volume_color(cls, color=(0, 0, 0, 0)):
        texture = cls(glGenTextures(1))
        texture.target = GL_TEXTURE_3D
        texture.width = texture.height = 1
        texture.use()
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, 1, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, bytes(color))
        for parameter in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER):
            glTexParameteri(GL_TEXTURE_3D, parameter, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, 0)
        return texture

    @classmethod
    def from_color(cls, r=0, g=0, b=0):
        pixels = np.full((64, 64, 3), (r, g, b), dtype=np.uint8)
        texture = cls(glGenTextures(1))
        texture.use()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 64, 64, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels)
        for parameter in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T):
            glTexParameteri(GL_TEXTURE_2D, parameter, GL_REPEAT)
        for parameter in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER):
            glTexParameteri(GL_TEXTURE_2D, parameter, GL_LINEAR)
        return texture

    def use(self):
        glBindTexture(self.target, self._id)
