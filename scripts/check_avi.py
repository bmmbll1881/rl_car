"""Structural validator for the MJPEG-AVI videos written by ``autodrive.video``.

There is no ffmpeg / media player available inside this sandbox, so instead of
"it looked fine" the file is verified explicitly:

  * RIFF sizes are consistent and the ``hdrl`` children have the exact byte
    length the AVI specification requires (avih 56, strh 56, strf 40);
  * ``avih`` reports the frame count that ``movi`` actually contains;
  * every ``idx1`` entry points at a ``00dc`` chunk with a matching size;
  * every frame payload decodes as a JPEG of the declared width/height.

    python scripts/check_avi.py figs/video_ppo_lap.avi
"""
from __future__ import annotations

import io
import struct
import sys

from PIL import Image


def find(data: bytes, fourcc: bytes, start: int = 0) -> int:
    i = data.find(fourcc, start)
    if i < 0:
        raise SystemExit(f"missing chunk {fourcc!r}")
    return i


def iter_chunks(data: bytes, start: int, end: int):
    """Yield (fourcc, data_start, size, chunk_pos) for the RIFF chunks in a range."""
    pos = start
    while pos + 8 <= end:
        fourcc = data[pos:pos + 4]
        size, = struct.unpack_from("<I", data, pos + 4)
        yield fourcc, pos + 8, size, pos
        pos += 8 + size + (size % 2)


def main(path):
    data = open(path, "rb").read()
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok = ok and cond
        print(("  PASS  " if cond else "  FAIL  ") + msg)

    print(f"{path}  ({len(data):,} bytes)")
    check(data[:4] == b"RIFF" and data[8:12] == b"AVI ", "RIFF/AVI signature")
    riff_size = struct.unpack_from("<I", data, 4)[0]
    check(riff_size == len(data) - 8, f"RIFF size {riff_size} == file-8 ({len(data)-8})")

    # top-level chunks (sequential walk, never a byte search: JPEG payloads can
    # contain arbitrary fourcc-looking bytes)
    top = list(iter_chunks(data, 12, len(data)))
    check(any(c == b"LIST" and data[s:s + 4] == b"hdrl" for c, s, _, _ in top), "LIST 'hdrl' present")
    check(any(c == b"LIST" and data[s:s + 4] == b"movi" for c, s, _, _ in top), "LIST 'movi' present")
    check(any(c == b"idx1" for c, _, _, _ in top), "'idx1' present")

    hdrl = next((s, z) for c, s, z, _ in top if c == b"LIST" and data[s:s + 4] == b"hdrl")
    movi = next((s, z, p) for c, s, z, p in top if c == b"LIST" and data[s:s + 4] == b"movi")
    idx1 = next((s, z) for c, s, z, _ in top if c == b"idx1")

    hdrl_children = list(iter_chunks(data, hdrl[0] + 4, hdrl[0] + hdrl[1]))
    avih = next((s, z) for c, s, z, _ in hdrl_children if c == b"avih")
    check(avih[1] == 56, f"avih chunk size {avih[1]} (expected 56)")
    fields = struct.unpack_from("<IIIIIIIIII4I", data, avih[0])
    usec, _, _, flags, total, _, streams, bufsize, w, h = fields[:10]
    fps = round(1_000_000 / usec) if usec else 0
    check(streams == 1, f"stream count {streams}")
    check(flags & 0x10 == 0x10, "AVIF_HASINDEX flag set")
    print(f"        {w}x{h}, {total} frames, {fps} fps")

    strl = next((s, z) for c, s, z, _ in hdrl_children if c == b"LIST" and data[s:s + 4] == b"strl")
    stream_children = list(iter_chunks(data, strl[0] + 4, strl[0] + strl[1]))
    strh = next((s, z) for c, s, z, _ in stream_children if c == b"strh")
    strf = next((s, z) for c, s, z, _ in stream_children if c == b"strf")
    check(strh[1] == 56, f"strh chunk size {strh[1]} (expected 56)")
    check(data[strh[0]:strh[0] + 4] == b"vids", "stream type 'vids'")
    check(data[strh[0] + 4:strh[0] + 8] == b"MJPG", "stream handler 'MJPG'")
    check(strf[1] == 40, f"strf chunk size {strf[1]} (expected 40)")
    bi_w, bi_h, planes, bpp = struct.unpack_from("<iiHH", data, strf[0] + 4)
    check((bi_w, bi_h) == (w, h), f"bitmap {bi_w}x{bi_h} matches avih")
    check(bpp == 24, f"biBitCount {bpp}")
    check(data[strf[0] + 16:strf[0] + 20] == b"MJPG", "biCompression 'MJPG'")

    # ---- movi frames -------------------------------------------------------
    # idx1 offsets are relative to the 'movi' fourcc (the de-facto convention,
    # also what ffmpeg's riff demuxer assumes): first frame -> offset 4
    movi_pos = movi[2] + 8
    frames = {}
    for fourcc, ds, size, cpos in iter_chunks(data, movi[0] + 4, movi[0] + movi[1]):
        if fourcc in (b"00dc", b"00db"):
            frames[cpos - movi_pos] = (size, data[ds:ds + size])
    check(len(frames) == total, f"movi contains {len(frames)} frames (avih says {total})")

    # ---- index -------------------------------------------------------------
    isize = idx1[1]
    n_idx = isize // 16
    check(n_idx == total, f"idx1 has {n_idx} entries (expected {total})")
    bad = 0
    for k in range(n_idx):
        ckid, cflags, off, csize = struct.unpack_from("<4sIII", data, idx1[0] + 16 * k)
        if ckid != b"00dc" or off not in frames or frames[off][0] != csize:
            bad += 1
    check(bad == 0, f"idx1 offsets resolve to real chunks (bad entries: {bad})")

    # ---- payload decodes ---------------------------------------------------
    bad_decode, sizes = 0, set()
    for off in sorted(frames):
        _, payload = frames[off]
        try:
            im = Image.open(io.BytesIO(payload))
            im.load()
            sizes.add(im.size)
            if im.format != "JPEG":
                bad_decode += 1
        except Exception:
            bad_decode += 1
    check(bad_decode == 0, f"all {len(frames)} payloads decode as JPEG (bad: {bad_decode})")
    check(sizes == {(w, h)}, f"decoded frame sizes {sizes}")

    print("RESULT:", "OK - container and every frame verified" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
