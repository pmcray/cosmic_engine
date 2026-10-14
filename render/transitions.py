"""Trumbull-style transition renderers used by the continuity engine.

The first family is the slit-scan tunnel — a parameterized promotion of
the ZPHC effect in `infinite_director.py`. It accepts the tail of shot A
and the head of shot B and produces n bridge frames in which the tunnel
geometry collapses through a flash and re-opens onto the new shot.

Future transitions to add here: ink-in-tank SPH dye dispersal, schlieren
shockwave, frame-buffer feedback warp.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


def render_slitscan_bridge(
    tail_a: list[np.ndarray],
    head_b: list[np.ndarray],
    n: int,
    palette: tuple[tuple[float, float, float], ...] | None = None,
    intensity: float = 1.0,
) -> Iterator[np.ndarray]:
    """Pure-numpy slit-scan tunnel transition.

    The Taichi version in `infinite_director.ZPHCTransitionRenderer` is
    GPU-fast but cannot be cleanly mixed with externally supplied frames.
    This numpy implementation operates directly on the inbound/outbound
    image buffers so it can carry their color and motion forward.
    """
    if n <= 0:
        return
    h, w = tail_a[0].shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx / (w - 1)) * 2.0 - 1.0
    v = (yy / (h - 1)) * 2.0 - 1.0
    r = np.sqrt(u * u + v * v) + 1e-5
    theta = np.arctan2(v, u)

    pal = np.array(
        palette if palette is not None else ((0.1, 0.3, 0.9), (0.9, 0.1, 0.3)),
        dtype=np.float32,
    )

    for i in range(n):
        progress = (i + 1) / (n + 1)
        time = (i + 1) * 0.1
        z = 1.0 / r + time * 10.0
        # mix in palette swirl
        mix_val = (np.sin(theta * 3.0 + time) + 1.0) * 0.5
        if len(pal) >= 2:
            color = pal[0][None, None, :] * mix_val[..., None] + pal[1][None, None, :] * (
                1.0 - mix_val[..., None]
            )
        else:
            color = np.broadcast_to(pal[0], (h, w, 3)).copy()
        slit = np.abs(np.sin(z * 50.0))[..., None]
        color = color * (1.0 + slit * 2.0)

        # vignette + noise modulation
        vignette = np.exp(-r * 2.0)[..., None]
        noise = (np.sin(z * 5.0 + theta * 8.0) * np.cos(z * 3.0 - time * 2.0) + 1.0)[..., None]
        tunnel = color * vignette * noise * intensity

        # blend A -> tunnel -> B with a flash at progress=0.5
        a = tail_a[i]
        b = head_b[i]
        if progress < 0.5:
            base = a * (1.0 - 2.0 * progress) + tunnel * (2.0 * progress)
        else:
            base = tunnel * (1.0 - 2.0 * (progress - 0.5)) + b * (2.0 * (progress - 0.5))
        # blinding flash at peak
        flash = max(0.0, 1.0 - abs(progress - 0.5) * 4.0)
        base = base + flash * 0.6
        yield base.astype(np.float32)
