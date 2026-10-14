"""Tests for studio.producer and studio.session.

End-to-end uses the constant renderer and write_video=False so the
whole loop runs on CPU with only numpy installed.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _toy_graph():
    from scene.manifest import PaletteRef, Shot, ShotGraph, Transition
    return ShotGraph(
        title="toy",
        shots=[
            Shot(
                id="A",
                renderer="constant",
                params={"color": (0.3, 0.3, 0.3)},
                palette=PaletteRef("trumbull_2001"),
                duration_frames=12,
                resolution=(32, 18),
            ),
            Shot(
                id="B",
                renderer="constant",
                params={"color": (0.6, 0.4, 0.2)},
                palette=PaletteRef("trumbull_2001"),
                duration_frames=12,
                resolution=(32, 18),
            ),
        ],
        transitions=[
            Transition("A", "B", "crossfade", 4),
        ],
    )


def test_paramspec_perturb_float_within_bounds() -> None:
    from studio.producer import ParamSpec

    spec = ParamSpec(
        name="x",
        get=lambda g: 0.5,
        set=lambda g, v: None,
        kind="float",
        bounds=(0.0, 1.0),
        perturb_scale=0.5,
    )
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = spec.perturb(0.5, rng)
        assert 0.0 <= v <= 1.0
    print("ok: test_paramspec_perturb_float_within_bounds")


def test_paramspec_choice_changes_value_when_possible() -> None:
    from studio.producer import ParamSpec

    spec = ParamSpec(
        name="c",
        get=lambda g: None,
        set=lambda g, v: None,
        kind="choice",
        choices=("a", "b", "c"),
    )
    rng = np.random.default_rng(0)
    seen = {spec.perturb("a", rng) for _ in range(40)}
    assert seen <= {"b", "c"}
    assert "a" not in seen
    # singleton choice is a no-op.
    solo = ParamSpec(name="s", get=lambda g: None, set=lambda g, v: None,
                     kind="choice", choices=("z",))
    assert solo.perturb("z", rng) == "z"
    print("ok: test_paramspec_choice_changes_value_when_possible")


def test_helpers_drive_a_real_graph() -> None:
    from studio.producer import (
        MutationSpace,
        shot_palette,
        shot_param,
        shot_seed,
        transition_kind,
    )

    g = _toy_graph()
    space = MutationSpace()
    space.add(shot_param("A", "color", kind="choice",
                         choices=((0.1, 0.1, 0.1), (0.9, 0.9, 0.9))))
    space.add(shot_palette("A", choices=("trumbull_2001", "nfb_universe_1960")))
    space.add(shot_seed("A", bounds=(0, 1_000_000)))
    space.add(transition_kind("A", "B"))

    rng = np.random.default_rng(0)
    for spec in space:
        current = spec.get(g)
        new = spec.perturb(current, rng)
        spec.set(g, new)
    g.validate()
    print("ok: test_helpers_drive_a_real_graph")


def test_random_producer_returns_validating_graph() -> None:
    from studio.producer import MutationSpace, RandomProducer, shot_param

    g = _toy_graph()
    space = MutationSpace([
        shot_param("A", "color", kind="choice",
                   choices=((0.1, 0.1, 0.1), (0.5, 0.5, 0.5), (0.9, 0.9, 0.9))),
    ])
    p = RandomProducer(g, space, seed=7)
    for _ in range(10):
        new_g, mutation = p.propose([])
        new_g.validate()
        assert mutation["param"].startswith("shots[A].params.color")
        assert mutation["policy"] == "random"
    print("ok: test_random_producer_returns_validating_graph")


def test_session_explore_end_to_end() -> None:
    from studio import (
        ArtifactStore,
        MetricsCritic,
        MutationSpace,
        RandomProducer,
        Session,
        shot_param,
    )

    g = _toy_graph()
    space = MutationSpace([
        shot_param("A", "color", kind="choice",
                   choices=((0.0, 0.0, 0.0), (0.3, 0.3, 0.3), (0.6, 0.6, 0.6))),
        shot_param("B", "color", kind="choice",
                   choices=((0.1, 0.0, 0.4), (0.7, 0.2, 0.1))),
    ])

    with tempfile.TemporaryDirectory() as d:
        store = ArtifactStore(root=d)
        producer = RandomProducer(g, space, seed=11)
        session = Session(
            base_graph=g,
            store=store,
            producer=producer,
            critic=MetricsCritic(),
            proxy={"resolution": (16, 16), "duration_scale": 0.5},
            write_video=False,
        )
        history = session.explore(budget=5, label_prefix="t")
        # base + 5 = 6 attempts.
        assert len(history) == 6
        for a in history:
            assert (a.artifact_path / "manifest.json").exists()
            assert (a.artifact_path / "metrics.json").exists()
            assert 0.0 <= a.score <= 1.0
        best = session.best()
        assert best is not None
        for a in history:
            assert a.score <= best.score + 1e-9

        log_path = session.save(Path(d) / "session.json")
        log = json.loads(log_path.read_text())
        assert log["best"]["iteration"] == best.iteration
        assert len(log["attempts"]) == 6
    print("ok: test_session_explore_end_to_end")


def test_session_caches_repeated_manifest() -> None:
    """Producing the same mutation twice must not re-render."""
    from studio import (
        ArtifactStore,
        MetricsCritic,
        MutationSpace,
        RandomProducer,
        Session,
        shot_param,
    )

    g = _toy_graph()
    # Choice with a single option: every proposal produces the same graph.
    space = MutationSpace([
        shot_param("A", "color", kind="choice",
                   choices=((0.42, 0.42, 0.42),)),
    ])

    with tempfile.TemporaryDirectory() as d:
        store = ArtifactStore(root=d)
        producer = RandomProducer(g, space, seed=0)
        session = Session(
            base_graph=g, store=store, producer=producer,
            critic=MetricsCritic(),
            proxy={"resolution": (16, 16), "duration_scale": 0.5},
            write_video=False,
        )
        session.explore(budget=4, label_prefix="cache")
        # 5 attempts, but only 2 unique manifests (base graph + the perturbed graph).
        rendered_dirs = list(Path(d).iterdir())
        assert len(rendered_dirs) <= 3, [p.name for p in rendered_dirs]
        cached = sum(1 for a in session.history if a.cached)
        assert cached >= 1
    print("ok: test_session_caches_repeated_manifest")


def test_epsilon_greedy_exploits_best() -> None:
    """With epsilon=0 the producer always perturbs from current best."""
    from studio import (
        ArtifactStore,
        EpsilonGreedyProducer,
        MetricsCritic,
        MutationSpace,
        Session,
        shot_palette,
    )

    g = _toy_graph()
    space = MutationSpace([
        shot_palette("A", choices=("nfb_universe_1960", "trumbull_2001", "jwst_nircam")),
    ])

    with tempfile.TemporaryDirectory() as d:
        store = ArtifactStore(root=d)
        producer = EpsilonGreedyProducer(g, space, epsilon=0.0, seed=3)
        session = Session(
            base_graph=g, store=store, producer=producer,
            critic=MetricsCritic(),
            proxy={"resolution": (16, 16), "duration_scale": 0.5},
            write_video=False,
        )
        session.explore(budget=3, label_prefix="eg")
        # First non-base attempt has nothing to exploit but the base.
        # Subsequent ones must claim policy=exploit and exploit_iter set.
        non_base = [a for a in session.history if a.mutation.get("policy") == "exploit"]
        assert len(non_base) >= 1
        for a in non_base:
            assert "exploit_iter" in a.mutation
    print("ok: test_epsilon_greedy_exploits_best")


def main() -> int:
    test_paramspec_perturb_float_within_bounds()
    test_paramspec_choice_changes_value_when_possible()
    test_helpers_drive_a_real_graph()
    test_random_producer_returns_validating_graph()
    test_session_explore_end_to_end()
    test_session_caches_repeated_manifest()
    test_epsilon_greedy_exploits_best()
    print("\nall session tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
