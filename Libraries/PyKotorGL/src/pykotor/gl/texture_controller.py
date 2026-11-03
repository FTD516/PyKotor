"""CPU texture controllers from K12's CC0 Aurora texture-controller source.

The image buffers belong to a preview, never to the loaded TPC resource. Frame
numbers gate updates so a texture shared by several meshes advances once.
"""
from __future__ import annotations

import numpy as np

from pykotor.resource.formats.txi import TXISettings

_F = np.float32
_COS = np.array((1, .98078525, .9238795, .8314696, .70710677, .55557024, .38268343, .19509032,
                 0, -.19509032, -.38268343, -.55557024, -.70710677, -.8314696, -.9238795, -.98078525,
                 -1, -.98078525, -.9238795, -.8314696, -.70710677, -.55557024, -.38268343, -.19509032,
                 0, .19509032, .38268343, .55557024, .70710677, .8314696, .9238795, .98078525), dtype=np.float32)


def _cos(angle):
    return _COS[np.trunc(np.asarray(angle, dtype=np.float32) / _F(6.2831855 / 32)).astype(np.int64) & 31]


def _sin(angle):
    return _COS[(np.trunc(np.asarray(angle, dtype=np.float32) / _F(6.2831855 / 32)).astype(np.int64) - 8) & 31]


def _bytes(values):
    return (np.trunc(values).astype(np.int64) & 255).astype(np.uint8)


def _resample(grid, width, height, *, half_pixel=False):
    gh, gw = grid.shape
    y, x = np.indices((height, width), dtype=np.float32)
    x *= _F(gw / width)
    y *= _F(gh / height)
    if half_pixel:
        ix, iy = np.trunc(x - _F(.5)).astype(np.int64), np.trunc(y - _F(.5)).astype(np.int64)
    else:
        ix, iy = x.astype(np.int64), y.astype(np.int64)
    fx, fy = x - ix.astype(np.float32), y - iy.astype(np.float32)
    row0 = (_F(1) - fx) * grid[iy, ix] + fx * grid[iy, (ix + 1) % gw]
    row1 = (_F(1) - fx) * grid[(iy + 1) % gh, ix] + fx * grid[(iy + 1) % gh, (ix + 1) % gw]
    return (_F(1) - fy) * row0 + fy * row1


class TextureRandom:
    """MSVC's 15-bit generator used by the engine's rand_wincompatible."""

    def __init__(self, seed=1):
        self.state = seed

    def next(self):
        self.state = (214013 * self.state + 2531011) & 0xFFFFFFFF
        return (self.state >> 16) & 32767

    def array(self, shape):
        count = int(np.prod(shape))
        return np.fromiter((self.next() for _ in range(count)), dtype=np.int32, count=count).reshape(shape)


class TextureController:
    def __init__(self, settings: TXISettings, pixels: np.ndarray, *, frames=1,
                 dirty=(0, 16, 0, 0), random: TextureRandom | None = None, height_map=False):
        self.settings = settings
        self.fields = settings.controller_settings
        self.name = settings.controller
        self.height_map = height_map
        self.source = pixels.copy()
        self.image = pixels.copy()
        self.height, self.width = pixels.shape[:2]
        self.random = TextureRandom() if random is None else random
        self.frames = frames
        self.index = 0
        self.last_frame = -1
        self.frame_time = _F(0)
        self.phase = _F(0)
        self.blend = _F(0)
        self.counter = 0
        self.life_frame = 0
        self.arturo_time = _F(0)
        self.arturo_pass = -1
        if self.name == "water":
            shape = (self.fields["waterheight"], self.fields["waterwidth"])
            self.heights = np.full(shape, 127, dtype=np.float32)
            self.velocity = np.zeros(shape, dtype=np.float32)
            self.force_phase = self.angle_phase = _F(0)
        elif self.name == "wave":
            self.heights = self.random.array((self.height, self.width)).astype(np.float32) % _F(256)
            self.previous = np.full_like(self.heights, 127)
        elif self.name == "perlin":
            self.previous_noise = self._perlin(self.width, self.height)
            self.next_noise = self._perlin(self.width, self.height)
        elif self.name == "life":
            self.image[:] = (0, 0, 0, 255)
            self.image[..., 0] = np.where(self.random.array((self.height, self.width)) % 100 >= 61, 255, 0)
        elif self.name == "arturo":
            gh, gw = self.fields["arturoheight"], self.fields["arturowidth"]
            y, x = np.indices((gh, gw), dtype=np.float32)
            radius = np.sqrt((x / _F(gw) * 2 - 1) ** 2 + (y / _F(gh) * 2 - 1) ** 2)
            self.heights = np.sin((_F(1) - radius) * _F(3.14159274)).astype(np.float32) * _F(127)
            self.previous = np.zeros_like(self.heights)
            self.height_image = np.full((self.height, self.width), 127, dtype=np.uint8)
            self._build_arturo()
            self._build_arturo()
        elif self.name in ("dirty", "dirty2", "dirty3"):
            color, size, self.formula, function = dirty
            # An area size of zero explicitly selects the engine's neutral 16³ texture.
            if size == 0:
                color, size, self.formula, function = 0, 16, 0, 0
            alpha = (color >> 24) & 255
            channels = ((color >> 16) & 255, (color >> 8) & 255, color & 255,
                        abs(alpha - 256 if alpha >= 128 else alpha))
            if function == 1:
                maximum = max(channels[:3])
                a, b, c = (self._perlin(size, size, maximum=maximum) for _ in range(3))
                values = a.astype(np.int32)[:, :, None] + b.astype(np.int32)[:, None, :] + c.astype(np.int32)[None, :, :]
            else:
                values = self.random.array((size, size, size)) & 255
            self.image = np.zeros((size, size, size, 4), dtype=np.uint8)
            for channel, divisor in enumerate(channels):
                if divisor:
                    self.image[..., channel] = values % divisor
            self.width = self.height = size

    def _perlin(self, width, height, *, maximum=None):
        result = np.full((height, width), 127, dtype=np.int32)
        size, scale = 4, _F(1)
        while True:
            noise = self.random.array((size, size)) if maximum is None or maximum else np.zeros((size, size), dtype=np.int32)
            values = ((noise % 128 - 64).astype(np.float32) * scale + _F(127)
                      if maximum is None else (noise % maximum).astype(np.float32) * scale if maximum else np.zeros_like(noise))
            octave = _bytes(values).astype(np.float32)
            # The engine passes Y as the first interpolation coordinate.
            result += _bytes(_resample(octave, height, width)).T.astype(np.int32) - 127
            if size * 2 > max(width, height):
                break
            size *= 2
            scale *= _F(.5)
        return (result & 255).astype(np.uint8)

    def _build_arturo(self):
        gh, gw = self.heights.shape
        rx, ry = self.random.next() % gw, self.random.next() % gh
        neighbors = np.roll(self.heights, 1, 1) + np.roll(self.heights, -1, 1)
        neighbors = neighbors + np.roll(self.heights, 1, 0) + np.roll(self.heights, -1, 0)
        next_heights = neighbors * _F(.5) - self.previous
        next_heights[ry, rx] += _F(self.random.next() % 256 - 128)
        self.previous = self.heights.copy()
        self.heights = np.clip(next_heights + _F(127), 0, 255) - _F(127)

    def _height_pixels(self, heights):
        if self.fields["distort"]:
            y, x = np.indices((self.height, self.width), dtype=np.int64)
            amplitude = _F(self.fields["distortionamplitude"])
            if self.fields["distortangle"]:
                angle = heights.astype(np.float32) * _F(.024543693)
                dx = np.trunc(amplitude * _sin(angle) - _F(.5)).astype(np.int64)
                dy = np.trunc(amplitude * _cos(angle) - _F(.5)).astype(np.int64)
            else:
                dx = dy = np.trunc(amplitude * (heights.astype(np.float32) - 127) * _F(.0078125) - _F(.5)).astype(np.int64)
            return self.source[(y + dy) & (self.height - 1), (x + dx) & (self.width - 1)].copy()
        if self.height_map:
            result = np.repeat(heights[..., None], 4, axis=2)
            result[..., 3] = 255
            return result
        result = np.empty((self.height, self.width, 4), dtype=np.uint8)
        for channel in range(4):
            scale = self.fields["channelscale"][channel] if channel < len(self.fields["channelscale"]) else 1.0
            translate = self.fields["channeltranslate"][channel] if channel < len(self.fields["channeltranslate"]) else 0.0
            values = _bytes(heights.astype(np.float32) * _F(max(0, scale))).astype(np.int32)
            values += int(_F(max(0, translate)) * _F(255)) & 255
            result[..., channel] = values & 255
        return result

    def advance(self, frame: int, dt: float) -> bool:
        if frame == self.last_frame:
            return False
        self.last_frame = frame
        dt = _F(dt)
        if self.name == "cycle":
            self.frame_time += dt
            fps = self.fields["fps"]
            if fps < 0 or self.frame_time * fps > 1:
                self.frame_time = _F(0)
                self.index = (self.index + 1) % self.frames
            return False
        if self.name is None or self.name in ("dirty", "dirty2", "dirty3"):
            return False
        if self.name == "random":
            heights = (self.random.array((self.height, self.width)) & 255).astype(np.uint8)
        elif self.name == "ringtexdistort":
            y, x = np.indices((self.height, self.width), dtype=np.float32)
            radius = np.sqrt(((x + 1) / _F(self.width) - _F(.5)) ** 2 + ((y + 1) / _F(self.width) - _F(.5)) ** 2)
            value = _bytes((_F(.03) * _cos(radius * _F(60) + self.phase * _F(25)) + _F(.5)) * _F(255))
            self.image[..., 0] = self.image[..., 1] = value
            self.image[..., 2:] = 255
            self.phase = _F((self.phase + dt) % _F(6.2831855))
            return True
        elif self.name == "perlin":
            self.blend = _F(self.blend + dt * _F(4))
            if self.blend > 1:
                self.previous_noise = self.next_noise
                self.next_noise = self._perlin(self.width, self.height)
                self.blend %= _F(1)
            heights = _bytes((_F(1) - self.blend) * self.previous_noise + self.blend * self.next_noise)
        elif self.name == "wave":
            delta = _F(min(dt, _F(.04)))
            square = delta * delta
            next_heights = self.heights * (_F(2) - _F(4) * square)
            for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
                next_heights += np.roll(self.heights, shift, axis) * square
            next_heights -= self.previous
            self.previous, self.heights = self.heights, np.clip(next_heights, 40, 215)
            ry, rx = self.random.next() % self.height, self.random.next() % self.width
            self.previous[ry, rx] += self.random.next() % 32
            heights = _bytes(self.heights)
        elif self.name == "water":
            if self.counter & 1:
                self.counter += 1
                return False
            self.counter = 1
            delta = _F(min(dt, _F(.04))) * _F(self.fields["speed"])
            self.force_phase = _F((self.force_phase + _F(self.fields["forcecyclespeed"]) * delta / _F(57.295776)) % _F(6.2831855))
            self.angle_phase = _F((self.angle_phase + _F(self.fields["anglecyclespeed"]) * delta / _F(57.295776)) % _F(6.2831855))
            amplitude = _cos(self.force_phase) * _F(.25) + _F(.2)
            phase = _cos(self.angle_phase)
            kernel = np.array(((.5, 1, .5), (1, 0, 1), (.5, 1, .5)), dtype=np.float32)
            for iy, ix, angle in ((0, 0, .7853982), (0, 2, 2.3561945), (2, 2, 3.926991), (2, 0, 5.4977875)):
                kernel[iy, ix] = amplitude * _cos(phase - _F(angle)) + _F(.5)
            center = self.heights - _F(127)
            force = np.zeros_like(center)
            for iy in range(3):
                for ix in range(3):
                    force += (np.roll(center, (1 - iy, 1 - ix), (0, 1)) - center) * kernel[iy, ix]
            force += self.velocity * _F(-.025)
            next_heights = delta * self.velocity + center + force * _F(.5) * delta * delta + _F(127)
            self.velocity += delta * force
            gh, gw = center.shape
            ry, rx = self.random.next() % gh, self.random.next() % gw
            self.velocity[ry, rx] += self.random.next() % 127
            self.velocity[(next_heights < 40) | (next_heights > 215)] = 0
            self.heights = np.clip(next_heights, 40, 215)
            heights = _bytes(_resample(self.heights, self.width, self.height, half_pixel=True))
            filtered = heights.astype(np.int32) >> 1
            for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
                filtered += np.roll(heights, shift, axis) >> 3
            heights = (filtered & 255).astype(np.uint8)
        elif self.name == "life":
            if frame < self.life_frame + 5:
                return False
            self.life_frame = frame
            alive = self.image[..., 0] != 0
            neighbors = np.zeros(alive.shape, dtype=np.uint8)
            for iy, ix in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                neighbors += np.roll(alive, (iy, ix), (0, 1))
            next_alive = (neighbors | alive.astype(np.uint8)) == 3
            age = self.image[..., 2].astype(np.int32)
            age = np.where(next_alive, np.minimum(age + 4, 255), np.where(alive, 128, np.maximum(age - 4, 0)))
            self.image[..., 0] = next_alive * 255
            self.image[..., 1] = np.where(next_alive, age, 0)
            self.image[..., 2] = age
            self.image[..., 3] = 255
            return True
        elif self.name == "arturo":
            self.arturo_time += dt
            if self.arturo_time <= _F(.03):
                return False
            self.arturo_time %= _F(.03)
            self.arturo_pass = (self.arturo_pass + 1) % 3
            if self.arturo_pass == 0:
                self.blend += dt * _F(self.fields["speed"]) * _F(1.5)
                if self.blend > 1:
                    self._build_arturo()
                    self.blend %= _F(1)
            if self.arturo_pass < 2:
                heightmap = ((_F(1) - self.blend) * _resample(self.previous, self.width, self.height, half_pixel=True)
                             + self.blend * _resample(self.heights, self.width, self.height, half_pixel=True) + _F(127))
                start = self.arturo_pass * (self.height // 2)
                end = self.height if self.arturo_pass == 1 else self.height // 2
                self.height_image[start:end] = _bytes(heightmap[start:end])
                return False
            heights = self.height_image
        self.image = self._height_pixels(heights)
        return True


def height_to_normal(pixels: np.ndarray, scale: float) -> np.ndarray:
    """Aurora's forward-difference normal map, stored as Z, -X, Y, weight."""
    heights = pixels[..., 0].astype(np.float32) / _F(255)
    dx = (heights - np.roll(heights, -1, 1)) * _F(scale)
    dy = (heights - np.roll(heights, -1, 0)) * _F(scale)
    inverse = _F(1) / np.sqrt(dx * dx + dy * dy + _F(1))
    result = np.empty(pixels.shape, dtype=np.uint8)
    result[..., 0] = _bytes(inverse * _F(127) + _F(128))
    result[..., 1] = _bytes(_F(128) - dx * inverse * _F(127))
    result[..., 2] = _bytes(dy * inverse * _F(127) + _F(128))
    result[..., 3] = 255
    return result


def normal_mipmaps(pixels: np.ndarray, scale: float, mipmap: bool) -> list[np.ndarray]:
    height, width = pixels.shape[:2]
    width -= int(bool(width & (width - 1)) and not (width - 1) & (width - 2))
    height -= int(bool(height & (height - 1)) and not (height - 1) & (height - 2))
    if width & (width - 1) or height & (height - 1):
        raise ValueError("Aurora height maps require power-of-two dimensions or an extra edge sample.")
    images = [height_to_normal(pixels, scale)[:height, :width].copy()]
    while mipmap and (width > 1 or height > 1):
        source = images[-1].astype(np.float32)
        y, x = np.indices((max(1, height // 2), max(1, width // 2)))
        vectors = np.zeros(y.shape + (3,), dtype=np.float32)
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            sample = source[(y * 2 + dy) % height, (x * 2 + dx) % width]
            vectors += (sample[..., :3] / _F(127) - _F(1)) * (sample[..., 3:] / _F(255))
        width, height = max(1, width // 2), max(1, height // 2)
        if width == 1 or height == 1 or (width, height) == (2, 2):
            vectors[:] = (1, 0, 0)
        elif (width, height) == (4, 4):
            vectors[..., 1:] *= _F(.5)
        length = np.linalg.norm(vectors, axis=2)
        zero = length == 0
        vectors[zero] = (1, 0, 0)
        result = np.empty(y.shape + (4,), dtype=np.uint8)
        result[..., :3] = _bytes(vectors / np.where(zero, _F(1), length)[..., None] * _F(127) + _F(128))
        result[..., 3] = _bytes(np.minimum(length * _F(.25), _F(1)) * _F(255))
        images.append(result)
    return images
