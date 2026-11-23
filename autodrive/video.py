"""Minimal video writer: Motion-JPEG in an AVI container.

No ffmpeg / OpenCV / imageio on this machine, and H.264 is not something that can
be implemented in a few hundred lines, so the videos are written as **MJPEG-AVI**
(a real video container with one JPEG per frame).  It plays in VLC, Windows Media
Player / 电影和电视, and converts to MP4 with a single ffmpeg call:

    ffmpeg -i run.avi -c:v libx264 -pix_fmt yuv420p run.mp4

The writer is deliberately simple: append JPEG payloads to a ``movi`` list, then
write the ``idx1`` index so players can seek.
"""
from __future__ import annotations

import io
import struct

from PIL import Image

AVIF_HASINDEX = 0x10
AVIIF_KEYFRAME = 0x10


def _chunk(fourcc: bytes, payload: bytes) -> bytes:
    out = fourcc + struct.pack("<I", len(payload)) + payload
    if len(payload) % 2:
        out += b"\x00"
    return out


def _list(list_type: bytes, payload: bytes) -> bytes:
    return b"LIST" + struct.pack("<I", len(payload) + 4) + list_type + payload


def _jpeg(frame: Image.Image, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    frame.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=1, optimize=False)
    return buf.getvalue()


def write_mjpeg_avi(path: str, frames, fps: int = 10, quality: int = 88) -> str:
    """``frames`` is an iterable of PIL images (all the same size)."""
    frames = list(frames)
    if not frames:
        raise ValueError("no frames to write")
    fpayloads = [_jpeg(f, quality) for f in frames]
    w, h = frames[0].size
    n = len(fpayloads)
    max_payload = max(len(p) for p in fpayloads)

    # ---- header chunks -----------------------------------------------------
    avih = struct.pack(
        "<IIIIIIIIII4I",
        int(1_000_000 / max(fps, 1)),   # dwMicroSecPerFrame
        max_payload * fps,              # dwMaxBytesPerSec
        0,                              # dwPaddingGranularity
        AVIF_HASINDEX,                  # dwFlags
        n,                              # dwTotalFrames
        0,                              # dwInitialFrames
        1,                              # dwStreams
        max_payload,                    # dwSuggestedBufferSize
        w, h,                           # dwWidth, dwHeight
        0, 0, 0, 0,                     # dwReserved[4]
    )
    strh = (
        b"vids"
        + b"MJPG"
        + struct.pack("<IHHIIIIIIII", 0, 0, 0, 0, 1, fps, 0, n, max_payload, 0xFFFFFFFF, 0)
        + struct.pack("<hhhh", 0, 0, w, h)
    )
    strf = struct.pack(
        "<IiiHH4sIiiII",
        40, w, h, 1, 24, b"MJPG", w * h * 3, 0, 0, 0, 0
    )
    hdrl = _list(b"hdrl", _chunk(b"avih", avih) + _list(b"strl", _chunk(b"strh", strh) + _chunk(b"strf", strf)))

    # ---- movi list ---------------------------------------------------------
    movi_payload = b""
    index = b""
    # offsets in idx1 are relative to the 'movi' fourcc, which sits 4 bytes
    # before the first chunk inside the list data
    base = 4
    for payload in fpayloads:
        index += struct.pack("<4sIII", b"00dc", AVIIF_KEYFRAME, base, len(payload))
        chunk = _chunk(b"00dc", payload)
        movi_payload += chunk
        base += len(chunk)
    movi = _list(b"movi", movi_payload)

    body = b"AVI " + hdrl + movi + _chunk(b"idx1", index)
    data = b"RIFF" + struct.pack("<I", len(body)) + body
    with open(path, "wb") as fh:
        fh.write(data)
    return path
