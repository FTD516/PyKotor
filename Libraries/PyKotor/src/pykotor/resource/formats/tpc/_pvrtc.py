"""PVRTC1 4bpp compression and decoding for K1's packed-chain TPCs.

Decoder adapted from PowerVR Native SDK's PVRTDecompress.cpp (4bpp path).
https://github.com/powervr-graphics/Native_SDK/blob/63f0098acdc11ec9a072c30000d2fa670a955bee/framework/PVRCore/texture/PVRTDecompress.cpp

The MIT License (MIT)
Copyright (c) Imagination Technologies Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

import struct


def _colors(value: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    # Endpoints are expanded to RGB555/A4 before interpolation, not RGBA8888.
    if value & 0x8000:
        a = ((value >> 10) & 31, (value >> 5) & 31,
             (value & 0x1E) | ((value & 0x1E) >> 4), 15)
    else:
        a = (((value & 0xF00) >> 7) | ((value & 0xF00) >> 11),
             ((value & 0xF0) >> 3) | ((value & 0xF0) >> 7),
             ((value & 0xE) << 1) | ((value & 0xE) >> 2), (value & 0x7000) >> 11)
    if value & 0x80000000:
        b = ((value >> 26) & 31, (value >> 21) & 31, (value >> 16) & 31, 15)
    else:
        b = (((value & 0xF000000) >> 23) | ((value & 0xF000000) >> 27),
             ((value & 0xF00000) >> 19) | ((value & 0xF00000) >> 23),
             ((value & 0xF0000) >> 15) | ((value & 0xF0000) >> 19), (value & 0x70000000) >> 27)
    return a, b


def _twiddle(x: int, y: int, width: int, height: int) -> int:
    """Morton order interleaves Y first; excess bits follow the shorter axis."""
    result, shift = 0, 0
    while 1 << shift < min(width, height):
        result |= ((y >> shift) & 1) << (2 * shift)
        result |= ((x >> shift) & 1) << (2 * shift + 1)
        shift += 1
    return result | ((x if width > height else y) >> shift) << (2 * shift)


def decode_pvrtc4(data: bytes, width: int, height: int, *, alpha: bool) -> bytearray:
    """Decode a mip, including the minimum 8x8 storage for smaller images."""
    stored_width, stored_height = max(8, width), max(8, height)
    nx, ny = stored_width // 4, stored_height // 4
    packed = tuple(struct.iter_unpack("<II", data))
    words = []
    for y in range(ny):
        row = []
        for x in range(nx):
            modulation, color = packed[_twiddle(x, y, nx, ny)]
            row.append((modulation, color & 1, _colors(color)))
        words.append(row)

    result = bytearray(width * height * 4)
    for wy in range(ny):
        for wx in range(nx):
            corners = (words[wy][wx], words[wy][(wx + 1) % nx],
                       words[(wy + 1) % ny][wx], words[(wy + 1) % ny][(wx + 1) % nx])
            for y in range(4):
                for x in range(4):
                    px, py = (wx * 4 + 2 + x) % stored_width, (wy * 4 + 2 + y) % stored_height
                    if px >= width or py >= height:
                        continue
                    weights = ((4 - x) * (4 - y), x * (4 - y), (4 - x) * y, x * y)
                    word = corners[((y + 2) // 4) * 2 + (x + 2) // 4]
                    code = (word[0] >> (2 * (((y + 2) % 4) * 4 + (x + 2) % 4))) & 3
                    modulation = ((0, 4, 4, 8) if word[1] else (0, 3, 5, 8))[code]
                    pixel = []
                    for channel in range(4):
                        a, b = (sum(corner[2][endpoint][channel] * weight
                                    for corner, weight in zip(corners, weights)) for endpoint in range(2))
                        a = (a >> 4) + a if channel == 3 else (a >> 6) + (a >> 1)
                        b = (b >> 4) + b if channel == 3 else (b >> 6) + (b >> 1)
                        pixel.append((a * (8 - modulation) + b * modulation) // 8)
                    if not alpha:
                        pixel[3] = 255
                    elif word[1] and code == 2:
                        pixel[3] = 0
                    offset = (py * width + px) * 4
                    result[offset:offset + 4] = bytes(pixel)
    return result


def _pack_endpoint(color: tuple[int, ...], *, first: bool, alpha: bool, upper: bool) -> int:
    """Bound each endpoint in the actual PVRTC opaque/translucent bit layout."""
    def quantize(value: int, bits: int) -> int:
        return (value * ((1 << bits) - 1) + (254 if upper else 0)) // 255

    opaque = not alpha or color[3] == 255 or upper and color[3] > 238
    if first:
        if opaque:
            return 0x8000 | quantize(color[0], 5) << 10 | quantize(color[1], 5) << 5 | quantize(color[2], 4) << 1
        return (min(7, (color[3] + (33 if upper else 0)) // 34) << 12
                | quantize(color[0], 4) << 8 | quantize(color[1], 4) << 4 | quantize(color[2], 3) << 1)
    if opaque:
        return 0x80000000 | quantize(color[0], 5) << 26 | quantize(color[1], 5) << 21 | quantize(color[2], 5) << 16
    return (min(7, (color[3] + (33 if upper else 0)) // 34) << 28
            | quantize(color[0], 4) << 24 | quantize(color[1], 4) << 20 | quantize(color[2], 4) << 16)


def encode_pvrtc4(data: bytes, width: int, height: int, *, alpha: bool) -> bytearray:
    """Encode RGBA8888 into PVRTC1 4bpp RGB/RGBA, with wrap-aware interpolation.

    Endpoints bracket each 4x4 block. Modulations are chosen against the decoded
    interpolated endpoints, including the alternative punch-through alpha mode.
    Minimum-size storage repeats the image edge; it does not resize the image.
    """
    if width <= 0 or height <= 0 or width & (width - 1) or height & (height - 1):
        raise ValueError("PVRTC1 requires positive power-of-two image dimensions.")
    if len(data) != width * height * 4:
        raise ValueError("RGBA input does not match the image dimensions.")
    stored_width, stored_height = max(8, width), max(8, height)
    nx, ny = stored_width // 4, stored_height // 4
    pixels = [tuple(data[(min(y, height - 1) * width + min(x, width - 1)) * 4:
                         (min(y, height - 1) * width + min(x, width - 1)) * 4 + 4])
              for y in range(stored_height) for x in range(stored_width)]
    colors = []
    endpoints = []
    for wy in range(ny):
        for wx in range(nx):
            block = [pixels[(wy * 4 + y) * stored_width + wx * 4 + x] for y in range(4) for x in range(4)]
            low = tuple(min(pixel[channel] for pixel in block) for channel in range(4))
            high = tuple(max(pixel[channel] for pixel in block) for channel in range(4))
            color = (_pack_endpoint(low, first=True, alpha=alpha, upper=False)
                     | _pack_endpoint(high, first=False, alpha=alpha, upper=True))
            colors.append(color)
            endpoints.append(_colors(color))

    modulations = [[0, 0] for _ in colors]
    errors = [[0, 0] for _ in colors]
    channels = 4 if alpha else 3
    for wy in range(ny):
        for wx in range(nx):
            corners = (endpoints[wy * nx + wx], endpoints[wy * nx + (wx + 1) % nx],
                       endpoints[((wy + 1) % ny) * nx + wx], endpoints[((wy + 1) % ny) * nx + (wx + 1) % nx])
            for y in range(4):
                for x in range(4):
                    px, py = (wx * 4 + 2 + x) % stored_width, (wy * 4 + 2 + y) % stored_height
                    target = pixels[py * stored_width + px]
                    weights = ((4 - x) * (4 - y), x * (4 - y), (4 - x) * y, x * y)
                    a, b = [], []
                    for channel in range(channels):
                        av, bv = (sum(corner[endpoint][channel] * weight
                                      for corner, weight in zip(corners, weights)) for endpoint in range(2))
                        a.append((av >> 4) + av if channel == 3 else (av >> 6) + (av >> 1))
                        b.append((bv >> 4) + bv if channel == 3 else (bv >> 6) + (bv >> 1))
                    word = (py // 4) * nx + px // 4
                    shift = 2 * ((py % 4) * 4 + px % 4)
                    for mode, values in enumerate(((0, 3, 5, 8), (0, 4, 4, 8))[:2 if alpha else 1]):
                        distances = []
                        for code, modulation in enumerate(values):
                            prediction = [(av * (8 - modulation) + bv * modulation) // 8 for av, bv in zip(a, b)]
                            if mode == 1 and code == 2:
                                prediction[3] = 0
                            distances.append(sum((target[channel] - prediction[channel]) ** 2 for channel in range(channels)))
                        code = min(range(4), key=distances.__getitem__)
                        errors[word][mode] += distances[code]
                        modulations[word][mode] |= code << shift

    words = [b""] * len(colors)
    for wy in range(ny):
        for wx in range(nx):
            index = wy * nx + wx
            mode = int(alpha and errors[index][1] < errors[index][0])
            words[_twiddle(wx, wy, nx, ny)] = struct.pack("<II", modulations[index][mode], colors[index] | mode)
    return bytearray(b"".join(words))
