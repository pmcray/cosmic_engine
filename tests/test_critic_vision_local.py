"""CPU-side tests for the LocalVLMCritic prompt + parser layer.

The actual model loading is skipped on CPU; we verify the
torch-free pieces (prompt assembly, JSON parsing, composite math,
keyframe selection, and the ShotMetrics shape the critic returns
from synthetic scores).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_prompt_includes_image_blocks_and_axes() -> None:
    from scene.manifest import PaletteRef, Shot
    from studio.critic_vision_local import build_shot_prompt

    shot = Shot(
        id="m51_wide",
        renderer="spiral_galaxy",
        palette=PaletteRef("trumbull_2001"),
        duration_frames=96,
        motion_hint="push_in",
    )
    imgs = [Path(f"keyframes/f{i:04d}.png") for i in range(6)]
    msg = build_shot_prompt(shot, imgs)
    assert len(msg) == 1
    content = msg[0]["content"]
    image_blocks = [c for c in content if c["type"] == "image"]
    text_blocks = [c for c in content if c["type"] == "text"]
    assert len(image_blocks) == 6
    assert len(text_blocks) == 1
    txt = text_blocks[0]["text"]
    for axis in ("composition", "color", "narrative_fit", "motion_quality",
                 "artistic_feel", "technical_artifacts", "one_suggestion"):
        assert axis in txt
    print("ok: test_prompt_includes_image_blocks_and_axes")


def test_parse_well_formed_response() -> None:
    from studio.critic_vision_local import parse_vlm_response

    raw = (
        '{"composition": 7, "color": 8, "narrative_fit": 6, '
        '"motion_quality": 9, "artistic_feel": 7, '
        '"technical_artifacts": 10, "one_suggestion": "tighten focal point"}'
    )
    p = parse_vlm_response(raw)
    assert p is not None
    assert p["composition"] == 7
    assert p["technical_artifacts"] == 10
    assert "tighten" in p["one_suggestion"]
    print("ok: test_parse_well_formed_response")


def test_parse_handles_markdown_fences_and_preamble() -> None:
    from studio.critic_vision_local import parse_vlm_response

    raw = (
        "Sure, here's my evaluation:\n"
        "```json\n"
        '{"composition": 6, "color": 5, "narrative_fit": 7, '
        '"motion_quality": 8, "artistic_feel": 5, '
        '"technical_artifacts": 9, "one_suggestion": "increase saturation",}\n'
        "```\n"
        "Hope that helps!"
    )
    p = parse_vlm_response(raw)
    assert p is not None
    assert p["color"] == 5
    print("ok: test_parse_handles_markdown_fences_and_preamble")


def test_parse_rejects_missing_axes() -> None:
    from studio.critic_vision_local import parse_vlm_response

    raw = '{"composition": 5, "color": 5}'  # missing axes
    assert parse_vlm_response(raw) is None
    print("ok: test_parse_rejects_missing_axes")


def test_parse_rejects_bad_json() -> None:
    from studio.critic_vision_local import parse_vlm_response

    assert parse_vlm_response("not json at all") is None
    assert parse_vlm_response("") is None
    print("ok: test_parse_rejects_bad_json")


def test_parse_clamps_out_of_range_scores() -> None:
    from studio.critic_vision_local import parse_vlm_response

    raw = (
        '{"composition": 25, "color": -3, "narrative_fit": 7, '
        '"motion_quality": 8, "artistic_feel": 7, '
        '"technical_artifacts": 10, "one_suggestion": "ok"}'
    )
    p = parse_vlm_response(raw)
    assert p["composition"] == 10
    assert p["color"] == 0
    print("ok: test_parse_clamps_out_of_range_scores")


def test_composite_score_in_unit_range() -> None:
    from studio.critic_vision_local import composite_from_scores

    perfect = {a: 10 for a in (
        "composition", "color", "narrative_fit", "motion_quality",
        "artistic_feel", "technical_artifacts",
    )}
    worst = {a: 0 for a in perfect}
    mid = {a: 5 for a in perfect}
    assert composite_from_scores(perfect) == 1.0
    assert composite_from_scores(worst) == 0.0
    assert abs(composite_from_scores(mid) - 0.5) < 1e-6
    print("ok: test_composite_score_in_unit_range")


def test_keyframes_for_shot_picks_evenly_spaced() -> None:
    from studio.critic_vision_local import keyframes_for_shot

    with tempfile.TemporaryDirectory() as d:
        kdir = Path(d) / "keyframes"
        kdir.mkdir()
        # Create 12 fake keyframes.
        for i in range(12):
            (kdir / f"f{i:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        picks = keyframes_for_shot(Path(d), max_n=6)
        assert len(picks) == 6
        # First and last are always included.
        assert picks[0].name == "f0000.png"
        assert picks[-1].name == "f0011.png"
    print("ok: test_keyframes_for_shot_picks_evenly_spaced")


def test_keyframes_returns_empty_when_dir_missing() -> None:
    from studio.critic_vision_local import keyframes_for_shot

    with tempfile.TemporaryDirectory() as d:
        assert keyframes_for_shot(Path(d), max_n=6) == []
    print("ok: test_keyframes_returns_empty_when_dir_missing")


def test_critic_protocol_compatible_with_metrics_critic() -> None:
    """LocalVLMCritic must expose the same evaluate_shot_artifact /
    evaluate_artifact methods that Session passes a critic through."""
    from studio.critic_vision_local import LocalVLMCritic
    from studio.metrics import MetricsCritic

    metric_methods = set(m for m in dir(MetricsCritic) if not m.startswith("_"))
    vlm_methods = set(m for m in dir(LocalVLMCritic) if not m.startswith("_"))
    for required in ("evaluate_shot_artifact", "evaluate_artifact"):
        assert required in metric_methods, required
        assert required in vlm_methods, required
    print("ok: test_critic_protocol_compatible_with_metrics_critic")


def test_critic_constructor_no_model_load() -> None:
    """Constructor must not load the model; that happens on first use."""
    from studio.critic_vision_local import LocalVLMCritic

    c = LocalVLMCritic(model_name="Qwen/Qwen2-VL-2B-Instruct")
    assert c._model is None
    assert c._processor is None
    assert c.model_name == "Qwen/Qwen2-VL-2B-Instruct"
    print("ok: test_critic_constructor_no_model_load")


def main() -> int:
    test_prompt_includes_image_blocks_and_axes()
    test_parse_well_formed_response()
    test_parse_handles_markdown_fences_and_preamble()
    test_parse_rejects_missing_axes()
    test_parse_rejects_bad_json()
    test_parse_clamps_out_of_range_scores()
    test_composite_score_in_unit_range()
    test_keyframes_for_shot_picks_evenly_spaced()
    test_keyframes_returns_empty_when_dir_missing()
    test_critic_protocol_compatible_with_metrics_critic()
    test_critic_constructor_no_model_load()
    print("\nall LocalVLMCritic tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
