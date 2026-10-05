"""Minimal PNG codec (stdlib only).

The ScreenObserver must run anywhere the agent runs, including
machines with no Pillow/numpy installed, so screenshot decoding
is done with ``zlib``/``struct`` alone.  Supports the PNGs that
screenshot tools actually produce: bit depth 8, non-interlaced,
color types 0 (gray), 2 (RGB), 3 (palette), 4 (gray+alpha) and
6 (RGBA).  Anything else is a structured ``invalid_image``
error, never a crash.
"""
from __future__ import annotations

import struct
import zlib

from afnan_ai.screen.base import ScreenErrorCode, ScreenException

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PixelImage:
    """A decoded image: rows of (r, g, b) tuples."""

    def __init__(self, width: int, height: int, rows):
        self.width = width
        self.height = height
        self.rows = rows

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        return self.rows[y][x]

    def fingerprint_bytes(self, grid: int = 32) -> bytes:
        """A tiny grayscale downsample, for change detection."""
        out = bytearray()
        step_x = max(1, self.width // grid)
        step_y = max(1, self.height // grid)
        for y in range(0, self.height, step_y):
            for x in range(0, self.width, step_x):
                r, g, b = self.rows[y][x]
                out.append((r * 30 + g * 59 + b * 11) // 100)
        return bytes(out)


def _invalid(message: str) -> ScreenException:
    return ScreenException(message, code=ScreenErrorCode.INVALID_IMAGE)


def decode_png(data: bytes) -> PixelImage:
    if not data or not data.startswith(_SIGNATURE):
        raise _invalid("Screenshot data is not a PNG image")
    pos = len(_SIGNATURE)
    header = None
    palette: list[tuple[int, int, int]] = []
    idat = bytearray()
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            header = struct.unpack(">IIBBBBB", chunk)
        elif ctype == b"PLTE":
            palette = [
                tuple(chunk[i:i + 3]) for i in range(0, len(chunk) - 2, 3)
            ]
        elif ctype == b"IDAT":
            idat.extend(chunk)
        elif ctype == b"IEND":
            break
    if header is None:
        raise _invalid("PNG image has no header (IHDR)")
    width, height, bit_depth, color_type, _comp, _filt, interlace = header
    if bit_depth != 8 or interlace != 0:
        raise _invalid(
            f"Unsupported PNG format (bit depth {bit_depth}, "
            f"interlace {interlace}); expected 8-bit non-interlaced"
        )
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise _invalid(f"Unsupported PNG color type {color_type}")
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as e:
        raise _invalid(f"Corrupt PNG image data: {e}") from e

    stride = width * channels
    rows: list[list[tuple[int, int, int]]] = []
    prev = bytearray(stride)
    offset = 0
    for _y in range(height):
        if offset >= len(raw):
            raise _invalid("Truncated PNG image data")
        filter_type = raw[offset]
        offset += 1
        line = bytearray(raw[offset:offset + stride])
        offset += stride
        _unfilter(line, prev, filter_type, channels)
        rows.append(_to_rgb(line, channels, palette))
        prev = line
    return PixelImage(width, height, rows)


def _unfilter(line: bytearray, prev: bytearray, filter_type: int, bpp: int):
    if filter_type == 0:
        return
    for i in range(len(line)):
        left = line[i - bpp] if i >= bpp else 0
        up = prev[i]
        up_left = prev[i - bpp] if i >= bpp else 0
        if filter_type == 1:
            line[i] = (line[i] + left) & 0xFF
        elif filter_type == 2:
            line[i] = (line[i] + up) & 0xFF
        elif filter_type == 3:
            line[i] = (line[i] + (left + up) // 2) & 0xFF
        elif filter_type == 4:
            line[i] = (line[i] + _paeth(left, up, up_left)) & 0xFF


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _to_rgb(line: bytearray, channels: int, palette) -> list:
    out = []
    if channels == 3:
        for i in range(0, len(line), 3):
            out.append((line[i], line[i + 1], line[i + 2]))
    elif channels == 4:
        for i in range(0, len(line), 4):
            out.append((line[i], line[i + 1], line[i + 2]))
    elif channels == 2:
        for i in range(0, len(line), 2):
            g = line[i]
            out.append((g, g, g))
    else:  # gray or palette
        for value in line:
            if palette:
                out.append(palette[value] if value < len(palette)
                           else (0, 0, 0))
            else:
                out.append((value, value, value))
    return out


def encode_png(width: int, height: int, rows) -> bytes:
    """Encode rows of (r, g, b) tuples as a PNG (for tests/tools)."""
    raw = bytearray()
    for row in rows:
        raw.append(0)
        for r, g, b in row:
            raw.extend((r & 0xFF, g & 0xFF, b & 0xFF))

    def chunk(ctype: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + ctype
            + payload
            + struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        _SIGNATURE
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )
