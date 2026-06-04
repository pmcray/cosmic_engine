"""Continuity engine: blend palette and motion across shot boundaries.

The graph executor delivers two adjacent shot frame streams plus the
Transition record. This module produces the blended frames that bridge
them. Three families are implemented:

  - `crossfade`: weighted linear blend in scene-linear RGB.
  - `histogram_match`: align channel CDFs of the inbound frames toward
    the outbound shot's first frame, then crossfade. This kills sudden
    palette pops at cuts.
  - `slitscan`: hand off to a parameterized slit-scan renderer (the
    Trumbull substrate). The actual renderer lives in
    `render.transitions`; this module just exposes the dispatch.

Histogram matching uses a per-channel CDF-mapping (Reinhard-style only
on luminance is also possible, but per-channel preserves the dye-like
shifts we want for Trumbull-flavored palettes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np


def _smoothstep(t: np.ndarray | float) -> np.ndarray | float:
    return t * t * (3.0 - 2.0 * t)


def crossfade(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear blend in scene-linear RGB. alpha=0 -> a, alpha=1 -> b."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return (a * (1.0 - alpha) + b * alpha).astype(np.float32)


def histogram_match(src: np.ndarray, ref: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Per-channel CDF match src -> ref. Inputs HWC float32 scene-linear.

    `strength` in [0, 1] interpolates between the source and the matched
    result so the effect can be eased in across a transition.
    """
    if src.shape[-1] != 3 or ref.shape[-1] != 3:
        raise ValueError("expected RGB input")
    out = np.empty_like(src)
    n = 4096
    for c in range(3):
        s = src[..., c].ravel()
        r = ref[..., c].ravel()
        # Build CDFs on a shared bin grid spanning both ranges.
        lo = float(min(s.min(), r.min()))
        hi = float(max(s.max(), r.max()))
        if hi <= lo:
            out[..., c] = src[..., c]
            continue
        bins = np.linspace(lo, hi, n + 1)
        s_hist, _ = np.histogram(s, bins=bins, density=False)
        r_hist, _ = np.histogram(r, bins=bins, density=False)
        s_cdf = np.cumsum(s_hist).astype(np.float64)
        r_cdf = np.cumsum(r_hist).astype(np.float64)
        s_cdf /= max(s_cdf[-1], 1.0)
        r_cdf /= max(r_cdf[-1], 1.0)
        # For each src value, find its CDF, then invert ref CDF.
        centers = 0.5 * (bins[:-1] + bins[1:])
        src_centers_idx = np.clip(
            np.searchsorted(bins, src[..., c]) - 1, 0, n - 1
        )
        src_cdf_vals = s_cdf[src_centers_idx]
        # invert r_cdf: for each cdf value find the bin center
        ref_idx = np.searchsorted(r_cdf, src_cdf_vals)
        ref_idx = np.clip(ref_idx, 0, n - 1)
        matched = centers[ref_idx].reshape(src[..., c].shape)
        out[..., c] = src[..., c] * (1.0 - strength) + matched.astype(np.float32) * strength
    return out


def apply_palette_match(src: np.ndarray, anchor_rgbs, strength: float = 0.5) -> np.ndarray:
    """Gentle pull toward palette anchor centroid in scene-linear RGB.

    Computes the centroid of anchor colors and lifts/biases the source
    image's mean toward it. This is a coarse but cheap "look transfer"
    used when no full reference frame is available.
    """
    if not anchor_rgbs:
        return src
    centroid = np.mean(np.array(anchor_rgbs, dtype=np.float32), axis=0)
    src_mean = src.reshape(-1, 3).mean(axis=0)
    bias = (centroid - src_mean) * strength
    return np.clip(src + bias, 0.0, None).astype(np.float32)


# ---- Transition dispatch ------------------------------------------------

@dataclass
class TransitionPlan:
    kind: str
    duration_frames: int
    histogram_strength: float = 0.7
    smoothstep: bool = True
    params: dict = None


class TransitionEngine:
    """Yields the bridge frames between two adjacent shots.

    Inputs are the tail frames of shot A and the head frames of shot B
    (both at least `plan.duration_frames` long). For slit-scan, this
    delegates to a registered transition renderer.
    """

    def __init__(self, plan: TransitionPlan):
        self.plan = plan

    def bridge(
        self,
        tail_a: list[np.ndarray],
        head_b: list[np.ndarray],
    ) -> Iterator[np.ndarray]:
        n = self.plan.duration_frames
        if n == 0 or self.plan.kind == "hard_cut":
            return iter(())
        if len(tail_a) < n or len(head_b) < n:
            raise ValueError(
                f"need >= {n} frames each side; got {len(tail_a)}/{len(head_b)}"
            )
        if self.plan.kind == "crossfade":
            return self._crossfade(tail_a, head_b, n)
        if self.plan.kind == "match_cut":
            return self._match_cut(tail_a, head_b, n)
        if self.plan.kind == "slitscan":
            return self._slitscan(tail_a, head_b, n)
        raise ValueError(f"unknown transition kind: {self.plan.kind}")

    def _crossfade(self, tail_a, head_b, n) -> Iterator[np.ndarray]:
        ref = head_b[0]
        for i in range(n):
            raw_t = (i + 1) / (n + 1)
            t = _smoothstep(raw_t) if self.plan.smoothstep else raw_t
            a = tail_a[i]
            if self.plan.histogram_strength > 0.0:
                a = histogram_match(a, ref, strength=self.plan.histogram_strength * t)
            b = head_b[i]
            yield crossfade(a, b, alpha=t)

    def _match_cut(self, tail_a, head_b, n) -> Iterator[np.ndarray]:
        ref = head_b[0]
        for i in range(n):
            t = (i + 1) / (n + 1)
            matched = histogram_match(tail_a[i], ref, strength=t)
            yield matched

    def _slitscan(self, tail_a, head_b, n) -> Iterator[np.ndarray]:
        # Import lazily; the slit-scan renderer pulls Taichi.
        from render.transitions import render_slitscan_bridge
        params = self.plan.params or {}
        yield from render_slitscan_bridge(tail_a, head_b, n, **params)
