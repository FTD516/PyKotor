"""PVRTC1 4bpp decoding, as used by K1's packed-chain TPCs.

Adapted from PowerVR Native SDK's PVRTDecompress.cpp (4bpp path).
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
