"""Image and video I/O with ACES sRGB output transform.

All renderer output is scene-linear float32 (HWC, RGB). This module owns
the conversion to display-encoded 8/16-bit when needed (PNG, MP4 frames)
and writes HDR EXR / TIFF when an HDR backend is available.

The tone-mapping is the ACES filmic curve fitted by Stephen Hill
(Krzysztof Narkowicz' approximation), followed by sRGB OETF. This is
"close enough to ACES" for masters and matches the look most viewers
expect; replace with a full OCIO config when one is configured.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ---- ACES tone-mapping --------------------------------------------------

_ACES_INPUT = np.array(
    [
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777],
    ],
    dtype=np.float32,
)

_ACES_OUTPUT = np.array(
    [
        [1.60475, -0.53108, -0.07367],
        [-0.10208, 1.10813, -0.00605],
        [-0.00327, -0.07276, 1.07602],
    ],
    dtype=np.float32,
)


def _rrt_odt_fit(v: np.ndarray) -> np.ndarray:
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    return a / np.maximum(b, 1e-10)


def aces_filmic(linear_rgb: np.ndarray, exposure_stops: float = 0.0) -> np.ndarray:
    """Scene-linear RGB -> display-linear sRGB. Input/output float32 HWC."""
    x = linear_rgb.astype(np.float32, copy=False)
    if exposure_stops != 0.0:
        x = x * np.float32(2.0 ** exposure_stops)
    x = x @ _ACES_INPUT.T
    x = _rrt_odt_fit(x)
    x = x @ _ACES_OUTPUT.T
    return np.clip(x, 0.0, 1.0)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Apply the sRGB OETF to display-linear values in [0,1]."""
    a = 0.055
    out = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        (1 + a) * np.power(np.maximum(linear, 1e-12), 1 / 2.4) - a,
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def to_uint8_srgb(linear_rgb: np.ndarray, exposure_stops: float = 0.0) -> np.ndarray:
    display_lin = aces_filmic(linear_rgb, exposure_stops=exposure_stops)
    display = srgb_encode(display_lin)
    return (display * 255.0 + 0.5).astype(np.uint8)


def to_uint16_srgb(linear_rgb: np.ndarray, exposure_stops: float = 0.0) -> np.ndarray:
    display_lin = aces_filmic(linear_rgb, exposure_stops=exposure_stops)
    display = srgb_encode(display_lin)
    return (display * 65535.0 + 0.5).astype(np.uint16)


# ---- File writers -------------------------------------------------------

def _try_imports():
    """Return a dict of available backends. Looked up lazily."""
    backends = {}
    try:
        import OpenEXR  # type: ignore
        import Imath  # type: ignore
        backends["openexr"] = (OpenEXR, Imath)
    except Exception:
        pass
    try:
        import imageio.v3 as iio  # type: ignore
        backends["imageio"] = iio
    except Exception:
        pass
    try:
        import tifffile  # type: ignore
        backends["tifffile"] = tifffile
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore
        backends["pil"] = Image
    except Exception:
        pass
    return backends


def write_exr(path: str | Path, linear_rgb: np.ndarray) -> None:
    """Write scene-linear EXR. Falls back to 32-bit float TIFF, then npy."""
    path = Path(path)
    backends = _try_imports()
    arr = np.ascontiguousarray(linear_rgb, dtype=np.float32)
    if "openexr" in backends:
        OpenEXR, Imath = backends["openexr"]
        h, w, _ = arr.shape
        header = OpenEXR.Header(w, h)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        header["channels"] = {c: Imath.Channel(pt) for c in "RGB"}
        ex = OpenEXR.OutputFile(str(path.with_suffix(".exr")), header)
        ex.writePixels(
            {
                "R": arr[..., 0].tobytes(),
                "G": arr[..., 1].tobytes(),
                "B": arr[..., 2].tobytes(),
            }
        )
        ex.close()
        return
    if "tifffile" in backends:
        backends["tifffile"].imwrite(str(path.with_suffix(".tif")), arr)
        return
    np.save(str(path.with_suffix(".npy")), arr)


def write_png(path: str | Path, linear_rgb: np.ndarray, exposure_stops: float = 0.0) -> None:
    """Write tonemapped 8-bit sRGB PNG."""
    path = Path(path).with_suffix(".png")
    rgb = to_uint8_srgb(linear_rgb, exposure_stops=exposure_stops)
    backends = _try_imports()
    if "pil" in backends:
        backends["pil"].fromarray(rgb, mode="RGB").save(path)
        return
    if "imageio" in backends:
        backends["imageio"].imwrite(str(path), rgb)
        return
    raise RuntimeError("no PNG backend available; install Pillow or imageio")


# ---- Video writer -------------------------------------------------------

class VideoWriter:
    """Streams scene-linear frames to a tonemapped MP4 via imageio-ffmpeg.

    Use as a context manager:

        with VideoWriter("out.mp4", fps=24, exposure_stops=0.0) as vw:
            for frame in renderer.iter_frames():
                vw.append(frame)
    """

    def __init__(self, path: str | Path, fps: int = 24, exposure_stops: float = 0.0, codec: str = "libx264"):
        self.path = str(Path(path).with_suffix(".mp4"))
        self.fps = fps
        self.exposure_stops = exposure_stops
        self.codec = codec
        self._writer = None

    def __enter__(self):
        backends = _try_imports()
        if "imageio" not in backends:
            raise RuntimeError(
                "VideoWriter needs imageio-ffmpeg. Install with: pip install imageio[ffmpeg]"
            )
        iio = backends["imageio"]
        self._writer = iio.imopen(self.path, "w", plugin="pyav")
        self._writer.init_video_stream(self.codec, fps=self.fps)
        return self

    def append(self, linear_rgb: np.ndarray) -> None:
        rgb = to_uint8_srgb(linear_rgb, exposure_stops=self.exposure_stops)
        self._writer.write_frame(rgb)

    def __exit__(self, *exc):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return False
