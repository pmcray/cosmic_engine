"""Mutation space + producers for the studio loop.

A ParamSpec is a single knob the producer is allowed to turn: a path
into the ShotGraph, a value kind, bounds or choices, and (for
continuous knobs) a Gaussian perturbation scale.

The convenience constructors `shot_param`, `shot_palette`, `shot_seed`,
and `transition_kind` cover the common cases. Anything else can be
expressed by passing a custom get/set pair to `ParamSpec`.

Two producers are provided:

  - RandomProducer    : pick one spec at random, perturb from base.
  - EpsilonGreedyProducer : with probability epsilon explore as above,
                            otherwise perturb from the current best
                            attempt's manifest.

Both return `(mutated_graph, mutation_description)`. The description
is a small JSON-safe dict the Session logs verbatim so attempts are
auditable.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from scene.manifest import ShotGraph, load_manifest
from scene.palette import PALETTES


@dataclass
class ParamSpec:
    name: str
    get: Callable[[ShotGraph], Any]
    set: Callable[[ShotGraph, Any], None]
    kind: str  # "float" | "int" | "choice"
    bounds: tuple | None = None
    choices: tuple = ()
    perturb_scale: float = 0.1

    def perturb(self, current: Any, rng: np.random.Generator) -> Any:
        if self.kind == "choice":
            if not self.choices:
                return current
            if len(self.choices) == 1:
                return self.choices[0]
            others = [c for c in self.choices if c != current] or list(self.choices)
            return others[int(rng.integers(len(others)))]
        if self.kind in ("float", "int"):
            if self.bounds is None:
                raise ValueError(f"{self.name}: float/int spec needs bounds")
            lo, hi = self.bounds
            span = hi - lo
            sigma = max(self.perturb_scale * span, 1e-9)
            cur = float(current) if current is not None else (lo + hi) * 0.5
            new = cur + float(rng.normal(0.0, sigma))
            new = max(lo, min(hi, new))
            return int(round(new)) if self.kind == "int" else float(new)
        raise ValueError(f"unknown spec kind: {self.kind}")


@dataclass
class MutationSpace:
    specs: list[ParamSpec] = field(default_factory=list)

    def __iter__(self):
        return iter(self.specs)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, i):
        return self.specs[i]

    def add(self, spec: ParamSpec) -> "MutationSpace":
        self.specs.append(spec)
        return self


# ---- common spec constructors ----------------------------------------

def shot_param(
    shot_id: str,
    key: str,
    kind: str,
    bounds: tuple | None = None,
    choices: Sequence = (),
    perturb_scale: float = 0.1,
) -> ParamSpec:
    def _get(g):
        return g.shot_by_id(shot_id).params.get(key)

    def _set(g, v):
        g.shot_by_id(shot_id).params[key] = v

    return ParamSpec(
        name=f"shots[{shot_id}].params.{key}",
        get=_get,
        set=_set,
        kind=kind,
        bounds=bounds,
        choices=tuple(choices),
        perturb_scale=perturb_scale,
    )


def shot_palette(shot_id: str, choices: Sequence[str] | None = None) -> ParamSpec:
    pal_choices = tuple(choices) if choices else tuple(PALETTES.keys())

    def _get(g):
        return g.shot_by_id(shot_id).palette.name

    def _set(g, v):
        g.shot_by_id(shot_id).palette.name = v

    return ParamSpec(
        name=f"shots[{shot_id}].palette.name",
        get=_get,
        set=_set,
        kind="choice",
        choices=pal_choices,
    )


def shot_seed(shot_id: str, bounds: tuple = (0, 2**31 - 1)) -> ParamSpec:
    def _get(g):
        return g.shot_by_id(shot_id).seed

    def _set(g, v):
        g.shot_by_id(shot_id).seed = int(v)

    return ParamSpec(
        name=f"shots[{shot_id}].seed",
        get=_get,
        set=_set,
        kind="int",
        bounds=bounds,
        perturb_scale=0.5,
    )


def transition_kind(
    from_shot: str,
    to_shot: str,
    choices: Sequence[str] = ("crossfade", "slitscan", "match_cut", "hard_cut"),
) -> ParamSpec:
    def _find(g):
        for t in g.transitions:
            if t.from_shot == from_shot and t.to_shot == to_shot:
                return t
        raise KeyError(f"no transition {from_shot}->{to_shot}")

    def _get(g):
        return _find(g).kind

    def _set(g, v):
        _find(g).kind = v

    return ParamSpec(
        name=f"transitions[{from_shot}->{to_shot}].kind",
        get=_get,
        set=_set,
        kind="choice",
        choices=tuple(choices),
    )


# ---- producers --------------------------------------------------------

def _jsonable(v):
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)


class Producer:
    def __init__(self, base_graph: ShotGraph, space: MutationSpace, *, seed: int | None = None):
        self.base = base_graph
        self.space = space
        self.rng = np.random.default_rng(seed)

    def propose(self, history) -> tuple[ShotGraph, dict]:
        raise NotImplementedError


class RandomProducer(Producer):
    def propose(self, history):
        g = copy.deepcopy(self.base)
        if len(self.space) == 0:
            return g, {"param": "_noop", "from": None, "to": None}
        spec = self.space[int(self.rng.integers(len(self.space)))]
        current = spec.get(g)
        new = spec.perturb(current, self.rng)
        spec.set(g, new)
        return g, {
            "param": spec.name,
            "from": _jsonable(current),
            "to": _jsonable(new),
            "policy": "random",
        }


class EpsilonGreedyProducer(RandomProducer):
    def __init__(
        self,
        base_graph: ShotGraph,
        space: MutationSpace,
        *,
        epsilon: float = 0.3,
        seed: int | None = None,
    ):
        super().__init__(base_graph, space, seed=seed)
        self.epsilon = float(epsilon)

    def propose(self, history):
        if not history or self.rng.random() < self.epsilon:
            return super().propose(history)
        best = max(history, key=lambda a: a.score)
        g = load_manifest(best.artifact_path / "manifest.json")
        if len(self.space) == 0:
            return g, {"param": "_noop", "from": None, "to": None, "policy": "exploit"}
        spec = self.space[int(self.rng.integers(len(self.space)))]
        current = spec.get(g)
        new = spec.perturb(current, self.rng)
        spec.set(g, new)
        return g, {
            "param": spec.name,
            "from": _jsonable(current),
            "to": _jsonable(new),
            "policy": "exploit",
            "exploit_iter": best.iteration,
        }
