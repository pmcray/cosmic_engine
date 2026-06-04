"""Tests for studio.artifacts and studio.metrics. CPU only."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_keyframe_indices() -> None:
    from studio.artifacts import _pick_keyframe_indices

    assert _pick_keyframe_indices(0, 6) == set()
    assert _pick_keyframe_indices(3, 6) == {0, 1, 2}
    idx = _pick_keyframe_indices(96, 6)
    assert idx == {0, 19, 38, 57, 76, 95}
    print("ok: test_keyframe_indices")


def test_artifact_store_layout() -> None:
    from scene.manifest import PaletteRef, Shot, ShotGraph
    from studio.artifacts import ArtifactStore

    g = ShotGraph(
        title="t",
        shots=[
            Shot(
                id="A",
                renderer="constant",
                params={"color": (0.1, 0.2, 0.3)},
                palette=PaletteRef("trumbull_2001"),
                duration_frames=8,
                fps=24,
                resolution=(8, 8),
            )
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        store = ArtifactStore(root=d)
        art = store.begin_render(g, label="unit")
        assert art.path.parent == Path(d)
        assert (art.path / "manifest.json").exists()
        assert (art.path / "provenance.json").exists()

        sa = art.begin_shot(g.shots[0])
        rng = np.random.default_rng(0)
        for i in range(g.shots[0].duration_frames):
            f = rng.random((8, 8, 3), dtype=np.float32) * 0.5
            sa.on_frame(i, f)
        sa.finalize()

        # Synthesize a tiny "video" file just so finalize copies something.
        fake_video = Path(d) / "v.mp4"
        fake_video.write_bytes(b"\x00\x00")
        art.finalize(video_path=fake_video)

        # 6 keyframes for an 8-frame shot.
        kf_npy = list((art.path / "shots" / "A" / "keyframes").glob("*.npy"))
        assert len(kf_npy) == 6, [p.name for p in kf_npy]

        timing = json.loads((art.path / "timing.json").read_text())
        assert timing["shots"]["A"]["frames_seen"] == 8
        assert timing["shots"]["A"]["keyframes_saved"] == 6
    print("ok: test_artifact_store_layout")


def test_metrics_on_constant_frames() -> None:
    from scene.manifest import PaletteRef, Shot
    from studio.metrics import MetricsCritic

    shot = Shot(
        id="black",
        renderer="constant",
        params={"color": (0.0, 0.0, 0.0)},
        palette=PaletteRef("neutral"),
        duration_frames=8,
        resolution=(16, 16),
    )
    crit = MetricsCritic()
    frames = [np.zeros((16, 16, 3), dtype=np.float32) for _ in range(8)]
    m = crit.evaluate_frames(shot, frames)
    assert m.frames_seen == 8
    assert m.mean_luma < 0.01
    assert m.black_frac > 0.9
    assert m.degenerate_frame_count == 8
    assert m.composite_score < 0.4  # mostly-black ought to score poorly
    assert any("near-degenerate" in n for n in m.notes)
    print("ok: test_metrics_on_constant_frames")


def test_metrics_flags_nan() -> None:
    from scene.manifest import PaletteRef, Shot
    from studio.metrics import MetricsCritic

    shot = Shot(
        id="bad",
        renderer="constant",
        palette=PaletteRef("neutral"),
        duration_frames=4,
        resolution=(8, 8),
    )
    rng = np.random.default_rng(1)
    frames = [rng.random((8, 8, 3), dtype=np.float32) for _ in range(4)]
    frames[2][0, 0, 0] = np.nan
    frames[3][1, 1, 1] = np.inf

    m = MetricsCritic().evaluate_frames(shot, frames)
    assert m.nan_pixels >= 1
    assert m.inf_pixels >= 1
    assert any("non-finite" in n for n in m.notes)
    print("ok: test_metrics_flags_nan")


def test_metrics_palette_distance_lower_for_matching() -> None:
    from scene.manifest import PaletteRef, Shot
    from studio.metrics import MetricsCritic

    # NFB is near-grayscale; Trumbull has saturated magentas/cyans.
    nfb_shot = Shot(
        id="nfb",
        renderer="constant",
        palette=PaletteRef("nfb_universe_1960"),
        duration_frames=3,
        resolution=(16, 16),
    )
    trum_shot = Shot(
        id="trum",
        renderer="constant",
        palette=PaletteRef("trumbull_2001"),
        duration_frames=3,
        resolution=(16, 16),
    )
    gray = [np.full((16, 16, 3), 0.4, dtype=np.float32) for _ in range(3)]
    crit = MetricsCritic()
    m_nfb = crit.evaluate_frames(nfb_shot, gray)
    m_trum = crit.evaluate_frames(trum_shot, gray)
    assert m_nfb.palette_distance < m_trum.palette_distance, (
        m_nfb.palette_distance,
        m_trum.palette_distance,
    )
    print("ok: test_metrics_palette_distance_lower_for_matching")


def test_metrics_temporal_static_vs_random() -> None:
    from scene.manifest import PaletteRef, Shot
    from studio.metrics import MetricsCritic

    shot = Shot(id="t", renderer="constant", palette=PaletteRef("neutral"),
                duration_frames=8, resolution=(16, 16))
    static = [np.full((16, 16, 3), 0.3, dtype=np.float32) for _ in range(8)]
    rng = np.random.default_rng(2)
    noisy = [rng.random((16, 16, 3), dtype=np.float32) for _ in range(8)]
    crit = MetricsCritic()
    m_static = crit.evaluate_frames(shot, static)
    m_noisy = crit.evaluate_frames(shot, noisy)
    assert m_static.temporal_abs_diff < 1e-4
    assert m_noisy.temporal_abs_diff > 0.05
    assert m_static.frozen_pair_count == 7
    assert m_noisy.frozen_pair_count == 0
    print("ok: test_metrics_temporal_static_vs_random")


def test_critic_evaluates_artifact_end_to_end() -> None:
    """Push frames through the artifact store, then evaluate from disk
    via the critic and confirm a shot with structure outscores one with
    none. Uses synthetic frames directly so we don't need any renderer."""
    from scene.manifest import PaletteRef, Shot, ShotGraph
    from studio.artifacts import ArtifactStore
    from studio.metrics import MetricsCritic

    g = ShotGraph(
        title="art",
        shots=[
            Shot(id="black", renderer="constant",
                 params={"color": (0.0, 0.0, 0.0)},
                 palette=PaletteRef("neutral"),
                 duration_frames=6, resolution=(16, 16)),
            Shot(id="lively", renderer="constant",
                 palette=PaletteRef("nfb_universe_1960"),
                 duration_frames=6, resolution=(16, 16)),
        ],
    )

    rng = np.random.default_rng(11)
    lively_frames = [
        np.clip(0.4 + 0.15 * rng.standard_normal((16, 16, 3), dtype=np.float32), 0, 1)
        for _ in range(6)
    ]
    black_frames = [np.zeros((16, 16, 3), dtype=np.float32) for _ in range(6)]

    with tempfile.TemporaryDirectory() as d:
        store = ArtifactStore(root=d)
        art = store.begin_render(g, label="end2end")
        for shot, frames in zip(g.shots, (black_frames, lively_frames)):
            sa = art.begin_shot(shot)
            for i, f in enumerate(frames):
                sa.on_frame(i, f)
            sa.finalize()
        (art.path / "video.mp4").write_bytes(b"\x00")
        art.finalize()

        result = MetricsCritic().evaluate_artifact(art.path)
        assert set(result.shots.keys()) == {"black", "lively"}
        assert result.shots["black"].composite_score < result.shots["lively"].composite_score
        assert result.aggregate["worst_shot"]["id"] == "black"
        assert (art.path / "metrics.json").exists()
        assert (art.path / "shots" / "black" / "metrics.json").exists()
    print("ok: test_critic_evaluates_artifact_end_to_end")


def main() -> int:
    test_keyframe_indices()
    test_artifact_store_layout()
    test_metrics_on_constant_frames()
    test_metrics_flags_nan()
    test_metrics_palette_distance_lower_for_matching()
    test_metrics_temporal_static_vs_random()
    test_critic_evaluates_artifact_end_to_end()
    print("\nall studio tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
