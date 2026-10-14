"""Tests for the MP4 writer (render/io.py).

The writer used to delegate to imageio's pyav plugin, which assigns the
stream's dimensions lazily on the first frame; recent PyAV rejects that
once the codec is open, so a render died partway through with "Cannot
change width after codec is open". These tests write more than one
frame — the case that failed — and read the result back.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _have_backend() -> bool:
    from render.io import _try_imports
    b = _try_imports()
    return "av" in b or "imageio" in b


def _gradient(h: int, w: int, phase: float) -> np.ndarray:
    """A scene-linear frame that changes with `phase`."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    f = np.zeros((h, w, 3), dtype=np.float32)
    f[..., 0] = (x / max(w - 1, 1) + phase) % 1.0
    f[..., 1] = y / max(h - 1, 1)
    f[..., 2] = phase
    return f


def test_writes_multiple_frames_and_reads_back() -> None:
    if not _have_backend():
        print("skip: no video backend installed")
        return
    from render.io import VideoWriter

    n, h, w = 12, 64, 96
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "clip.mp4"
        with VideoWriter(path, fps=24) as vw:
            for i in range(n):
                vw.append(_gradient(h, w, i / n))
        assert path.exists() and path.stat().st_size > 0

        try:
            import av
        except ImportError:
            print("ok: test_writes_multiple_frames_and_reads_back (write only)")
            return
        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            assert stream.width == w and stream.height == h
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        # Every frame must survive the encoder's buffer through close().
        assert len(frames) == n, f"wrote {n} frames, read back {len(frames)}"
        # The clip is not a freeze frame.
        assert len({f.tobytes() for f in frames}) > 1
    print("ok: test_writes_multiple_frames_and_reads_back")


def test_odd_dimensions_are_accepted() -> None:
    """h264 needs even dimensions; an odd-sized render must still
    produce a file rather than raising."""
    if not _have_backend():
        print("skip: no video backend installed")
        return
    from render.io import VideoWriter

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "odd.mp4"
        with VideoWriter(path, fps=12) as vw:
            for i in range(4):
                vw.append(_gradient(45, 81, i / 4))
        assert path.exists() and path.stat().st_size > 0
    print("ok: test_odd_dimensions_are_accepted")


def test_frame_size_change_is_rejected_clearly() -> None:
    """A mid-stream size change is a caller bug; it should say so rather
    than surface as an opaque codec error."""
    if not _have_backend():
        print("skip: no video backend installed")
        return
    from render.io import VideoWriter

    with tempfile.TemporaryDirectory() as d:
        try:
            with VideoWriter(Path(d) / "bad.mp4", fps=24) as vw:
                vw.append(_gradient(32, 64, 0.0))
                vw.append(_gradient(48, 64, 0.5))
        except ValueError as e:
            assert "size changed" in str(e)
        else:
            raise AssertionError("expected ValueError on a size change")
    print("ok: test_frame_size_change_is_rejected_clearly")


def test_empty_clip_closes_cleanly() -> None:
    if not _have_backend():
        print("skip: no video backend installed")
        return
    from render.io import VideoWriter

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "empty.mp4"
        with VideoWriter(path, fps=24):
            pass
        # Nothing was written, so no file is expected — but closing must
        # not raise.
        assert not path.exists() or path.stat().st_size >= 0
    print("ok: test_empty_clip_closes_cleanly")


def main_runner() -> int:
    test_writes_multiple_frames_and_reads_back()
    test_odd_dimensions_are_accepted()
    test_frame_size_change_is_rejected_clearly()
    test_empty_clip_closes_cleanly()
    print("\nall video io tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_runner())
