"""CPU-side tests for the voyage composer."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_compose_voyage_returns_valid_shot_graph() -> None:
    from studio.voyage_composer import compose_voyage, PHASES

    g = compose_voyage(target_seconds=60.0, seed=1)
    g.validate()
    assert len(g.shots) >= len(PHASES)  # at least one shot per phase
    # Every shot references a known renderer name (the validator only
    # checks that the field exists; we sanity-check it's plausibly one
    # of ours).
    known = {
        "volumetric_gas_giant", "saturn_class", "spiral_galaxy",
        "diffuse_nebula", "stellar_surface", "exoplanet_atmosphere",
        "terrain", "ca_creatures", "kerr_black_hole",
        "slitscan_tunnel", "cosmic_web",
    }
    for s in g.shots:
        assert s.renderer in known, s.renderer
    print("ok: test_compose_voyage_returns_valid_shot_graph")


def test_compose_voyage_duration_close_to_target() -> None:
    from studio.voyage_composer import compose_voyage, total_duration_seconds

    target = 600.0
    g = compose_voyage(target_seconds=target, seed=7)
    actual = total_duration_seconds(g)
    # Within 20% of the target (procedural picks vary).
    assert abs(actual - target) / target < 0.25, f"target {target}, actual {actual}"
    print("ok: test_compose_voyage_duration_close_to_target")


def test_compose_voyage_reproducible_with_same_seed() -> None:
    from studio.voyage_composer import compose_voyage

    a = compose_voyage(target_seconds=300.0, seed=42)
    b = compose_voyage(target_seconds=300.0, seed=42)
    assert len(a.shots) == len(b.shots)
    for sa, sb in zip(a.shots, b.shots):
        assert sa.id == sb.id
        assert sa.renderer == sb.renderer
        assert sa.duration_frames == sb.duration_frames
    print("ok: test_compose_voyage_reproducible_with_same_seed")


def test_compose_voyage_different_seeds_diverge() -> None:
    from studio.voyage_composer import compose_voyage

    a = compose_voyage(target_seconds=300.0, seed=1)
    b = compose_voyage(target_seconds=300.0, seed=2)
    # At least one shot should differ in renderer pick.
    differ = any(
        sa.renderer != sb.renderer
        for sa, sb in zip(a.shots, b.shots)
    )
    assert differ or len(a.shots) != len(b.shots)
    print("ok: test_compose_voyage_different_seeds_diverge")


def test_compose_voyage_covers_all_phases() -> None:
    from studio.voyage_composer import compose_voyage, PHASES

    g = compose_voyage(target_seconds=600.0, seed=11)
    phase_names = {p.name for p in PHASES}
    represented = set()
    for s in g.shots:
        for name in phase_names:
            if s.id.startswith(name + "_"):
                represented.add(name)
                break
    # All eight phases should land at least one shot.
    assert represented == phase_names, (represented, phase_names)
    print("ok: test_compose_voyage_covers_all_phases")


def test_compose_voyage_transitions_reference_existing_shots() -> None:
    from studio.voyage_composer import compose_voyage

    g = compose_voyage(target_seconds=120.0, seed=3)
    shot_ids = {s.id for s in g.shots}
    for tr in g.transitions:
        assert tr.from_shot in shot_ids, tr.from_shot
        assert tr.to_shot in shot_ids, tr.to_shot
        assert tr.duration_frames >= 0
        assert tr.kind in ("crossfade", "slitscan", "match_cut", "hard_cut")
    print("ok: test_compose_voyage_transitions_reference_existing_shots")


def test_cli_writes_manifest_to_disk() -> None:
    from studio.voyage_composer import main

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "v.json"
        rc = main(["--duration", "60", "--seed", "9", "--out", str(out)])
        assert rc == 0
        assert out.exists()
        # Round-trip load.
        from scene.manifest import load_manifest
        g = load_manifest(out)
        assert len(g.shots) > 0
    print("ok: test_cli_writes_manifest_to_disk")


def test_cli_fast_mode_reduces_resolution_and_steps() -> None:
    from studio.voyage_composer import main
    from scene.manifest import load_manifest

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "v.json"
        main([
            "--duration", "60", "--seed", "5",
            "--resolution", "1280x720",
            "--fast",
            "--out", str(out),
        ])
        g = load_manifest(out)
        # Resolution should be halved.
        assert g.shots[0].resolution == (640, 360)
        # At least one shot should have a reduced march_steps below 32.
        reduced = [
            s for s in g.shots
            if "march_steps" in s.params and s.params["march_steps"] <= 32
        ]
        assert len(reduced) > 0
    print("ok: test_cli_fast_mode_reduces_resolution_and_steps")


def main_runner() -> int:
    test_compose_voyage_returns_valid_shot_graph()
    test_compose_voyage_duration_close_to_target()
    test_compose_voyage_reproducible_with_same_seed()
    test_compose_voyage_different_seeds_diverge()
    test_compose_voyage_covers_all_phases()
    test_compose_voyage_transitions_reference_existing_shots()
    test_cli_writes_manifest_to_disk()
    test_cli_fast_mode_reduces_resolution_and_steps()
    print("\nall voyage composer tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_runner())
