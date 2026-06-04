"""Studio: producer / critic / editor loop for the shot graph.

Step 1 of the loop is the artifact store: every render produces a
self-contained, addressable directory that downstream critics can
evaluate without re-rendering. Step 2 is the metrics critic: a fast,
deterministic, no-ML scorer that grades a shot on temporal coherence,
palette adherence, dynamic range, motion magnitude, edge density, and
common artifact classes (black frames, NaN runs, frozen runs).

A later PR will add the producer (parameter-mutation policy), the VLM
critic (Claude vision / LLaVA / CLIP), and the human-in-the-loop UI.
"""
from .artifacts import ArtifactStore, RenderArtifact, ShotArtifact
from .metrics import MetricsCritic, MetricSet, ShotMetrics
from .provenance import capture_provenance

__all__ = [
    "ArtifactStore",
    "RenderArtifact",
    "ShotArtifact",
    "MetricsCritic",
    "MetricSet",
    "ShotMetrics",
    "capture_provenance",
]
