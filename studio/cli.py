"""Small CLI for poking at the artifact store and critic.

    python -m studio.cli list                  # list recent renders
    python -m studio.cli evaluate <path>       # run MetricsCritic on an artifact
    python -m studio.cli show <path>           # print metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import ArtifactStore
from .metrics import MetricsCritic


def _cmd_list(args: argparse.Namespace) -> int:
    store = ArtifactStore(args.root)
    for p in store.list_renders():
        m = store.load_metrics(p)
        score = m.get("aggregate", {}).get("composite_score") if m else None
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "  -  "
        print(f"  {score_str}  {p.name}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    critic = MetricsCritic()
    result = critic.evaluate_artifact(Path(args.artifact))
    print(json.dumps(result.aggregate, indent=2))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    metrics_file = Path(args.artifact) / "metrics.json"
    if not metrics_file.exists():
        print(f"no metrics.json at {metrics_file}; run 'evaluate' first", file=sys.stderr)
        return 1
    print(metrics_file.read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="studio")
    parser.add_argument("--root", default="renders", help="artifact store root")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp_list = sub.add_parser("list", help="list recent renders")
    sp_list.set_defaults(func=_cmd_list)

    sp_eval = sub.add_parser("evaluate", help="score a render with MetricsCritic")
    sp_eval.add_argument("artifact", help="path to render artifact directory")
    sp_eval.set_defaults(func=_cmd_evaluate)

    sp_show = sub.add_parser("show", help="print an existing metrics.json")
    sp_show.add_argument("artifact", help="path to render artifact directory")
    sp_show.set_defaults(func=_cmd_show)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
