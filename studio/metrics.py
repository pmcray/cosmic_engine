"""Hand-coded, deterministic, no-ML quality metrics for shot artifacts.

The metrics are split into three classes:

  - Photometric: mean luma, contrast, dynamic range, blown-out / black
    fractions. Computed on the *tonemapped* (display-encoded) image so
    they match what a viewer sees.

  - Structural: edge density via a Sobel filter; palette adherence via
    L2 distance from the shot's declared palette centroid.

  - Temporal & artifact: frame-to-frame mean absolute delta (a single
    proxy that conflates motion and noise — useful as a "boil
    detector"), frozen-frame run length, NaN/Inf count, fraction of
    near-degenerate single-color frames.

A composite quality score is provided as a weighted blend, but the
producer is expected to read individual metrics rather than the
composite when deciding what to mutate.

Frames are scene-linear float32 HWC. The critic owns the tonemap.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from scene.manifest import Shot, ShotGraph
from scene.palette import Palette, get_palette


_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# ---- low-level metric helpers ----------------------------------------

def _luma(frame: np.ndarray) -> np.ndarray:
    return frame @ _LUMA


def _tonemap(frame: np.ndarray) -> np.ndarray:
    from render.io import aces_filmic, srgb_encode
    return srgb_encode(aces_filmic(frame))


def _sobel_edge_density(gray: np.ndarray) -> float:
    if gray.ndim != 2:
        gray = _luma(gray)
    # 3x3 Sobel, vectorized via numpy slicing.
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    gx = (
        -1 * gray[:-2, :-2] + 1 * gray[:-2, 2:]
        + -2 * gray[1:-1, :-2] + 2 * gray[1:-1, 2:]
        + -1 * gray[2:, :-2] + 1 * gray[2:, 2:]
    )
    gy = (
        -1 * gray[:-2, :-2] + -2 * gray[:-2, 1:-1] + -1 * gray[:-2, 2:]
        + 1 * gray[2:, :-2] + 2 * gray[2:, 1:-1] + 1 * gray[2:, 2:]
    )
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag.mean())


def _palette_distance(frame: np.ndarray, anchors: np.ndarray) -> float:
    px = frame.reshape(-1, 3).astype(np.float32)
    # mean per-pixel min-distance to any anchor in scene-linear RGB.
    # anchors: (k, 3). compute (n, k) distances chunked to bound memory.
    chunk = 4096
    total = 0.0
    n = 0
    for i in range(0, px.shape[0], chunk):
        block = px[i : i + chunk]
        d = np.linalg.norm(block[:, None, :] - anchors[None, :, :], axis=2)
        total += float(d.min(axis=1).sum())
        n += block.shape[0]
    return total / max(n, 1)


# ---- accumulators ----------------------------------------------------

@dataclass
class _StreamAccum:
    seen: int = 0
    nan_pixels: int = 0
    inf_pixels: int = 0
    sum_luma: float = 0.0
    sum_contrast: float = 0.0  # per-frame std of display luma
    sum_blown: float = 0.0
    sum_black: float = 0.0
    sum_edge: float = 0.0
    sum_palette_dist: float = 0.0
    sum_dyn_range_db: float = 0.0
    sum_temporal_abs_diff: float = 0.0
    temporal_n: int = 0
    frozen_pairs: int = 0
    degenerate_frames: int = 0
    prev_display: np.ndarray | None = None


def _ingest_frame(
    acc: _StreamAccum,
    frame: np.ndarray,
    anchors: np.ndarray | None,
) -> None:
    acc.seen += 1
    nan_mask = ~np.isfinite(frame)
    if nan_mask.any():
        acc.nan_pixels += int(np.isnan(frame).sum())
        acc.inf_pixels += int(np.isinf(frame).sum())
        frame = np.where(nan_mask, 0.0, frame).astype(np.float32, copy=False)

    display = _tonemap(frame)
    dl = _luma(display)
    acc.sum_luma += float(dl.mean())
    acc.sum_contrast += float(dl.std())
    acc.sum_blown += float((dl > 0.98).mean())
    acc.sum_black += float((dl < 0.02).mean())
    if dl.std() < 1e-4:
        acc.degenerate_frames += 1

    p1, p99 = np.percentile(dl, (1.0, 99.0))
    dyn = 20.0 * np.log10(max(float(p99), 1e-6) / max(float(p1), 1e-6))
    acc.sum_dyn_range_db += float(dyn)

    acc.sum_edge += _sobel_edge_density(dl)

    if anchors is not None and anchors.size > 0:
        acc.sum_palette_dist += _palette_distance(frame, anchors)

    if acc.prev_display is not None and acc.prev_display.shape == display.shape:
        diff = float(np.mean(np.abs(display - acc.prev_display)))
        acc.sum_temporal_abs_diff += diff
        acc.temporal_n += 1
        if diff < 1e-5:
            acc.frozen_pairs += 1
    acc.prev_display = display


# ---- public API ------------------------------------------------------

@dataclass
class ShotMetrics:
    shot_id: str
    frames_seen: int
    mean_luma: float
    mean_contrast: float
    mean_dyn_range_db: float
    blown_frac: float
    black_frac: float
    edge_density: float
    palette_distance: float
    temporal_abs_diff: float
    frozen_pair_count: int
    degenerate_frame_count: int
    nan_pixels: int
    inf_pixels: int
    composite_score: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetricSet:
    shots: dict[str, ShotMetrics]
    aggregate: dict

    def to_dict(self) -> dict:
        return {
            "shots": {k: v.to_dict() for k, v in self.shots.items()},
            "aggregate": self.aggregate,
        }


# Composite is intentionally simple and easy to override. Higher is
# better; bounded roughly to [0, 1] for typical scenes.
_COMPOSITE_WEIGHTS = {
    "dyn_range": 0.25,
    "edge_density": 0.20,
    "palette": 0.20,
    "temporal": 0.15,
    "blown": -0.10,
    "black": -0.05,
    "degenerate": -0.25,
    "nan": -0.50,
}


def _normalize(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(max(0.0, min(1.0, (v - lo) / (hi - lo))))


def _composite(m: ShotMetrics) -> float:
    score = 0.0
    score += _COMPOSITE_WEIGHTS["dyn_range"] * _normalize(m.mean_dyn_range_db, 6.0, 40.0)
    score += _COMPOSITE_WEIGHTS["edge_density"] * _normalize(m.edge_density, 0.005, 0.08)
    score += _COMPOSITE_WEIGHTS["palette"] * (1.0 - _normalize(m.palette_distance, 0.0, 0.6))
    score += _COMPOSITE_WEIGHTS["temporal"] * _normalize(m.temporal_abs_diff, 0.005, 0.08)
    score += _COMPOSITE_WEIGHTS["blown"] * _normalize(m.blown_frac, 0.005, 0.15)
    score += _COMPOSITE_WEIGHTS["black"] * _normalize(m.black_frac, 0.05, 0.6)
    if m.frames_seen:
        score += _COMPOSITE_WEIGHTS["degenerate"] * (m.degenerate_frame_count / m.frames_seen)
    if m.nan_pixels or m.inf_pixels:
        score += _COMPOSITE_WEIGHTS["nan"]
    return float(max(0.0, min(1.0, score + 0.5)))


def _finalize(acc: _StreamAccum, shot: Shot, has_palette: bool) -> ShotMetrics:
    n = max(acc.seen, 1)
    n_t = max(acc.temporal_n, 1)
    notes: list[str] = []
    if acc.nan_pixels or acc.inf_pixels:
        notes.append(f"non-finite pixels: nan={acc.nan_pixels} inf={acc.inf_pixels}")
    if acc.degenerate_frames > 0:
        notes.append(f"{acc.degenerate_frames} near-degenerate frames (std < 1e-4)")
    if acc.frozen_pairs >= 3:
        notes.append(f"{acc.frozen_pairs} frozen frame-pairs (temporal diff < 1e-5)")
    avg_blown = acc.sum_blown / n
    if avg_blown > 0.1:
        notes.append(f"{avg_blown:.0%} pixels blown out on average")
    if acc.sum_black / n > 0.6:
        notes.append("majority black: shot may have failed to render")
    metrics = ShotMetrics(
        shot_id=shot.id,
        frames_seen=acc.seen,
        mean_luma=acc.sum_luma / n,
        mean_contrast=acc.sum_contrast / n,
        mean_dyn_range_db=acc.sum_dyn_range_db / n,
        blown_frac=avg_blown,
        black_frac=acc.sum_black / n,
        edge_density=acc.sum_edge / n,
        palette_distance=(acc.sum_palette_dist / n) if has_palette else -1.0,
        temporal_abs_diff=acc.sum_temporal_abs_diff / n_t if acc.temporal_n else 0.0,
        frozen_pair_count=acc.frozen_pairs,
        degenerate_frame_count=acc.degenerate_frames,
        nan_pixels=acc.nan_pixels,
        inf_pixels=acc.inf_pixels,
        composite_score=0.0,
        notes=notes,
    )
    metrics.composite_score = _composite(metrics)
    return metrics


class MetricsCritic:
    """Compute deterministic metrics on shot frame streams or artifacts."""

    def evaluate_frames(
        self,
        shot: Shot,
        frames: Iterable[np.ndarray],
    ) -> ShotMetrics:
        anchors = self._palette_anchors(shot)
        acc = _StreamAccum()
        for f in frames:
            _ingest_frame(acc, f, anchors)
        return _finalize(acc, shot, has_palette=anchors is not None)

    def evaluate_shot_artifact(
        self,
        shot: Shot,
        shot_artifact_dir: Path | str,
    ) -> ShotMetrics:
        from .artifacts import load_keyframes
        frames = load_keyframes(shot_artifact_dir)
        metrics = self.evaluate_frames(shot, frames)
        (Path(shot_artifact_dir) / "metrics.json").write_text(
            json.dumps(metrics.to_dict(), indent=2)
        )
        return metrics

    def evaluate_artifact(self, artifact_dir: Path | str) -> MetricSet:
        from scene.manifest import load_manifest
        artifact_dir = Path(artifact_dir)
        graph = load_manifest(artifact_dir / "manifest.json")
        shots: dict[str, ShotMetrics] = {}
        for shot in graph.shots:
            sdir = artifact_dir / "shots" / shot.id
            if not sdir.exists():
                continue
            shots[shot.id] = self.evaluate_shot_artifact(shot, sdir)
        aggregate = self._aggregate(shots, graph)
        (artifact_dir / "metrics.json").write_text(
            json.dumps({"shots": {k: v.to_dict() for k, v in shots.items()}, "aggregate": aggregate}, indent=2)
        )
        return MetricSet(shots=shots, aggregate=aggregate)

    def _palette_anchors(self, shot: Shot) -> np.ndarray | None:
        try:
            pal: Palette = get_palette(shot.palette.name)
        except KeyError:
            return None
        return np.asarray(pal.anchors, dtype=np.float32)

    def _aggregate(
        self,
        shots: dict[str, ShotMetrics],
        graph: ShotGraph,
    ) -> dict:
        if not shots:
            return {"shot_count": 0, "composite_score": 0.0}
        scores = [m.composite_score for m in shots.values()]
        weights = [graph.shot_by_id(sid).duration_frames for sid in shots]
        total = sum(weights) or 1
        weighted = sum(s * w for s, w in zip(scores, weights)) / total
        worst = min(shots.values(), key=lambda m: m.composite_score)
        return {
            "shot_count": len(shots),
            "composite_score": round(weighted, 4),
            "worst_shot": {
                "id": worst.shot_id,
                "composite_score": round(worst.composite_score, 4),
                "notes": worst.notes,
            },
            "any_nonfinite": any(m.nan_pixels or m.inf_pixels for m in shots.values()),
        }
