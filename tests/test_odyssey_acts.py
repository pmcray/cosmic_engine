"""CPU-side tests for the Odyssey act curves and their graph adapters."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_jovian_pose_endpoints_and_monotony() -> None:
    from render.odyssey_acts import jovian_approach_pose

    p0 = jovian_approach_pose(0.0)
    p1 = jovian_approach_pose(1.0)
    # Dolly closes distance monotonically.
    assert p0["cam_dist"] > p1["cam_dist"]
    prev = p0["cam_dist"]
    for i in range(1, 11):
        d = jovian_approach_pose(i / 10.0)["cam_dist"]
        assert d <= prev + 1e-9
        prev = d
    # No roll until the onset, roll by the end.
    assert jovian_approach_pose(0.5)["cam_roll"] == 0.0
    assert p1["cam_roll"] > 0.0
    # Flash only fires in the terminal window.
    assert jovian_approach_pose(0.9)["exposure"] == 1.0
    assert p1["exposure"] > 5.0
    # Sun direction is a unit-ish horizontal pair plus a sinking ly.
    sx, sy, sz = p0["sun_dir"]
    assert abs(math.hypot(sx, sz) - 1.0) < 1e-6
    assert p0["sun_dir"][1] > p1["sun_dir"][1]
    print("ok: test_jovian_pose_endpoints_and_monotony")


def test_jovian_pose_flash_gain_zero_disables_whiteout() -> None:
    from render.odyssey_acts import jovian_approach_pose

    assert jovian_approach_pose(1.0, flash_gain=0.0)["exposure"] == 1.0
    print("ok: test_jovian_pose_flash_gain_zero_disables_whiteout")


def test_corridor_pose_tumble_and_flash() -> None:
    from render.odyssey_acts import corridor_pose

    assert corridor_pose(0.0)["roll"] == 0.0
    assert abs(corridor_pose(1.0)["roll"] - math.pi / 2.0) < 1e-9
    mid = corridor_pose(0.55)["roll"]
    assert 0.0 < mid < math.pi / 2.0
    # Entry flash decays.
    assert corridor_pose(0.0)["exposure"] > corridor_pose(0.5)["exposure"]
    assert abs(corridor_pose(1.0)["exposure"] - 1.0) < 1e-6
    print("ok: test_corridor_pose_tumble_and_flash")


def test_infinite_pose_mass_ramp_and_fade() -> None:
    from render.odyssey_acts import infinite_pose

    p0 = infinite_pose(0.0)
    p1 = infinite_pose(1.0)
    assert p0["cam_dist"] > p1["cam_dist"]
    assert p1["bh_mass"] > p0["bh_mass"]
    assert p0["exposure"] == 1.0
    assert p1["exposure"] == 0.0
    # fade_onset beyond 1 disables the fade entirely.
    assert infinite_pose(1.0, fade_onset=2.0)["exposure"] == 1.0
    print("ok: test_infinite_pose_mass_ramp_and_fade")


def test_odyssey_renderers_registered() -> None:
    import render.adapters  # noqa: F401 — registration side effect
    from render.core import list_renderers

    names = set(list_renderers())
    assert {"odyssey_jovian_approach", "stargate_corridor",
            "odyssey_infinite"} <= names
    print("ok: test_odyssey_renderers_registered")


def test_composer_places_odyssey_phrases() -> None:
    """Across a handful of seeds the Odyssey phrases should actually get
    picked by the composer (they sit in four phases' template pools)."""
    from studio.voyage_composer import compose_voyage

    seen = set()
    for seed in range(8):
        g = compose_voyage(target_seconds=600.0, seed=seed)
        seen |= {s.renderer for s in g.shots}
    assert "stargate_corridor" in seen
    assert {"odyssey_jovian_approach", "odyssey_infinite"} & seen
    print("ok: test_composer_places_odyssey_phrases")


def test_stargate_corridor_adapter_smoke() -> None:
    """Tiny CPU render through the graph adapter: correct shape,
    orientation contract (H, W, 3), finite values."""
    try:
        import taichi  # noqa: F401
    except ImportError:
        print("skip: taichi not installed")
        return

    from scene.manifest import Shot
    from render.core import get_renderer

    shot = Shot(id="t", renderer="stargate_corridor",
                params={"samples": 1}, duration_frames=2,
                resolution=(64, 36))
    r = get_renderer("stargate_corridor")(shot)
    frame = r.render_frame(0, 0.0, shot.camera.at(0.0))
    assert frame.shape == (36, 64, 3)
    assert frame.dtype == np.float32
    assert np.isfinite(frame).all()
    assert frame.min() >= 0.0
    print("ok: test_stargate_corridor_adapter_smoke")


def main_runner() -> int:
    test_jovian_pose_endpoints_and_monotony()
    test_jovian_pose_flash_gain_zero_disables_whiteout()
    test_corridor_pose_tumble_and_flash()
    test_infinite_pose_mass_ramp_and_fade()
    test_odyssey_renderers_registered()
    test_composer_places_odyssey_phrases()
    test_stargate_corridor_adapter_smoke()
    print("\nall odyssey act tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_runner())
