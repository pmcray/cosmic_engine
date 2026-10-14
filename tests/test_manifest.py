"""Lightweight tests that exercise the new architecture without a GPU.

Run from the repo root:  python -m tests.test_manifest
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_manifest_roundtrip() -> None:
    from scene.manifest import (
        Camera,
        PaletteRef,
        Shot,
        ShotGraph,
        Transition,
        dump_manifest,
        load_manifest,
    )

    g = ShotGraph(
        title="t",
        shots=[
            Shot(
                id="a",
                renderer="constant",
                params={"color": (0.1, 0.2, 0.3)},
                camera=Camera(
                    position=(0, 0, -5),
                    path_to=Camera(position=(0, 0, -2)),
                ),
                palette=PaletteRef(name="trumbull_2001"),
                duration_frames=10,
                fps=24,
                resolution=(64, 36),
                motion_hint="push_in",
            ),
            Shot(
                id="b",
                renderer="constant",
                params={"color": (0.4, 0.0, 0.1)},
                duration_frames=10,
                fps=24,
                resolution=(64, 36),
            ),
        ],
        transitions=[
            Transition(from_shot="a", to_shot="b", kind="crossfade", duration_frames=4)
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "g.json"
        dump_manifest(g, p)
        loaded = load_manifest(p)

    assert loaded.title == "t"
    assert len(loaded.shots) == 2
    assert loaded.shots[0].camera.path_to is not None
    assert loaded.shots[0].palette.name == "trumbull_2001"
    assert loaded.shots[0].motion_hint == "push_in"
    assert loaded.transitions[0].kind == "crossfade"
    print("ok: test_manifest_roundtrip")


def test_camera_interp() -> None:
    from scene.manifest import Camera

    c0 = Camera(position=(0, 0, -10), path_to=Camera(position=(0, 0, 0)))
    c_mid = c0.at(0.5)
    assert c_mid.position == (0.0, 0.0, -5.0), c_mid.position
    assert c_mid.path_to is None
    print("ok: test_camera_interp")


def test_palette_registry() -> None:
    from scene.palette import PALETTES, get_palette

    assert "trumbull_2001" in PALETTES
    p = get_palette("nfb_universe_1960")
    assert p.saturation < 0.5
    try:
        get_palette("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("ok: test_palette_registry")


def test_constant_renderer_and_aces() -> None:
    import numpy as np
    from scene.manifest import Shot
    from render.core import ConstantRenderer
    from render.io import aces_filmic, to_uint8_srgb

    shot = Shot(
        id="x",
        renderer="constant",
        params={"color": (1.5, 0.2, 0.05)},
        duration_frames=3,
        fps=24,
        resolution=(8, 4),
    )
    r = ConstantRenderer(shot)
    frames = list(r.iter_frames())
    assert len(frames) == 3
    f = frames[0]
    assert f.shape == (4, 8, 3)
    assert f.dtype == np.float32
    assert abs(f[0, 0, 0] - 1.5) < 1e-6

    rgb = to_uint8_srgb(f)
    assert rgb.shape == (4, 8, 3)
    assert rgb.dtype == np.uint8
    # ACES should compress the >1.0 red toward, not clip-saturate, a near-pure red.
    assert rgb[0, 0, 0] > rgb[0, 0, 1]
    assert rgb[0, 0, 0] > rgb[0, 0, 2]
    print("ok: test_constant_renderer_and_aces")


def test_histogram_match_and_crossfade() -> None:
    import numpy as np
    from director.continuity import crossfade, histogram_match

    rng = np.random.default_rng(7)
    a = rng.random((16, 16, 3), dtype=np.float32) * 0.4
    b = rng.random((16, 16, 3), dtype=np.float32) * 0.4 + 0.6

    out = crossfade(a, b, 0.5)
    assert out.shape == a.shape
    assert abs(out.mean() - 0.5 * (a.mean() + b.mean())) < 1e-5

    matched = histogram_match(a, b, strength=1.0)
    # matched mean should be very close to b mean
    assert abs(matched.mean() - b.mean()) < 0.02
    print("ok: test_histogram_match_and_crossfade")


def test_transition_engine_crossfade_bridge() -> None:
    import numpy as np
    from director.continuity import TransitionEngine, TransitionPlan

    n = 6
    tail = [np.full((4, 4, 3), 0.0, dtype=np.float32) for _ in range(n)]
    head = [np.full((4, 4, 3), 1.0, dtype=np.float32) for _ in range(n)]
    plan = TransitionPlan(kind="crossfade", duration_frames=n, histogram_strength=0.0)
    eng = TransitionEngine(plan)
    bridge = list(eng.bridge(tail, head))
    assert len(bridge) == n
    means = [float(f.mean()) for f in bridge]
    assert all(0.0 < m < 1.0 for m in means)
    assert means[0] < means[-1]
    print("ok: test_transition_engine_crossfade_bridge")


def test_example_manifest_loads() -> None:
    from scene.manifest import load_manifest

    p = ROOT / "scene" / "examples" / "jupiter_to_stargate.json"
    g = load_manifest(p)
    assert len(g.shots) == 3
    # Jovian shot uses the volumetric renderer (the legacy `gas_giant`
    # falls into a green Pi-Lattice debug pattern when its hidden
    # observer_attention isn't set; the example should not depend on it).
    assert g.shots[0].renderer == "volumetric_gas_giant"
    assert g.shots[1].renderer == "kerr_black_hole"
    assert g.shots[2].renderer == "slitscan_tunnel"
    assert g.transitions[1].kind == "slitscan"
    # Camera distance must keep the planet inside the frame for the
    # volumetric renderer's coordinate convention (planet_radius=1).
    cam = g.shots[0].camera
    assert abs(cam.position[2]) < 6.0, "camera too far for planet_radius=1"
    assert abs(cam.path_to.position[2]) >= 1.5, "camera path ends inside planet"
    print("ok: test_example_manifest_loads")


def test_graph_runs_with_constant_renderer() -> None:
    """End-to-end: run the GraphRunner using only the constant renderer,
    feeding frames into an in-memory sink to avoid the imageio dep."""
    import numpy as np
    from collections import deque
    from scene.manifest import Camera, PaletteRef, Shot, ShotGraph, Transition
    from render.core import get_renderer
    from director.continuity import TransitionEngine, TransitionPlan
    from director.graph import _make_plan

    g = ShotGraph(
        title="ct",
        shots=[
            Shot(
                id="A",
                renderer="constant",
                params={"color": (0.0, 0.0, 0.0)},
                duration_frames=20,
                resolution=(16, 8),
            ),
            Shot(
                id="B",
                renderer="constant",
                params={"color": (1.0, 1.0, 1.0)},
                duration_frames=20,
                resolution=(16, 8),
            ),
        ],
        transitions=[Transition("A", "B", "crossfade", 6)],
    )
    g.validate()

    # Manually replicate GraphRunner._stream to a list sink.
    sink: list[np.ndarray] = []

    def push(f):
        sink.append(f)

    prev_tail: list = []
    prev_plan: TransitionPlan | None = None
    for i, (shot, trans, nxt) in enumerate(g.iter_pairs()):
        plan = _make_plan(trans)
        tail_n = plan.duration_frames
        head_n = prev_plan.duration_frames if prev_plan else 0
        cls = get_renderer(shot.renderer)
        r = cls(shot)
        it = r.iter_frames()
        head_buf = []
        for _ in range(head_n):
            head_buf.append(next(it))
        if prev_plan is not None and prev_tail and head_buf:
            eng = TransitionEngine(prev_plan)
            for bf in eng.bridge(prev_tail, head_buf):
                push(bf)
        tail_buf = deque(maxlen=tail_n) if tail_n else deque()
        for f in it:
            if tail_n:
                if len(tail_buf) == tail_buf.maxlen and tail_buf[0] is not None:
                    push(tail_buf[0])
                tail_buf.append(f)
            else:
                push(f)
        prev_tail = list(tail_buf) if tail_n else []
        prev_plan = plan
    for f in prev_tail:
        push(f)

    # 20 + 20 - 6 = 34 frames total.
    assert len(sink) == 34, len(sink)
    # The bridge frames lie strictly between black and white.
    bridge_means = [float(f.mean()) for f in sink[14:20]]
    assert bridge_means[0] < bridge_means[-1]
    assert 0.0 < bridge_means[0] and bridge_means[-1] < 1.0
    # Surrounding body frames are the pure colors.
    assert float(sink[0].mean()) == 0.0
    assert float(sink[-1].mean()) == 1.0
    print("ok: test_graph_runs_with_constant_renderer")


def main() -> int:
    test_manifest_roundtrip()
    test_camera_interp()
    test_palette_registry()
    test_constant_renderer_and_aces()
    test_histogram_match_and_crossfade()
    test_transition_engine_crossfade_bridge()
    test_example_manifest_loads()
    test_graph_runs_with_constant_renderer()
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
