"""Local multimodal VLM critic for the studio loop.

Wraps a Hugging Face vision-language model (default: Qwen2-VL-2B-Instruct)
into the same protocol as MetricsCritic, so it slots into Session
interchangeably:

    from studio import Session
    session = Session(..., critic=LocalVLMCritic())

The critic runs entirely on the local GPU. No API keys, no rate
limits, no network dependency after the first model download. The
default Qwen2-VL-2B fits comfortably in fp16 on a T4 alongside the
renderer kernels (~5 GB), with a Phi-3.5-vision or Qwen2.5-VL-7B-AWQ
fallback for users who want bigger.

The critic sees up to N keyframes per shot (default 6) plus a manifest
snippet describing renderer / palette / motion intent, and asks the
model to return a structured JSON score block:

    {
      "composition":           0-10,
      "color":                 0-10,
      "narrative_fit":         0-10,
      "motion_quality":        0-10,
      "artistic_feel":         0-10,
      "technical_artifacts":   0-10 (10 = none visible),
      "one_suggestion":        str (one concrete improvement)
    }

Composite score is a weighted average of the six axes; the suggestion
lands in `ShotMetrics.notes` for downstream UIs.

The module is torch/transformers-guarded so the rest of the codebase
imports cleanly on CPU hosts that don't have the ML stack installed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scene.manifest import Shot, ShotGraph, load_manifest
from studio.metrics import MetricSet, ShotMetrics


# Default model name (Hugging Face). Qwen2-VL-2B-Instruct is the
# best small VLM we found for cinematic scoring; ~5 GB in fp16, fits
# on a T4 alongside the renderer kernels. Override via constructor.
DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


# Composite-score weights for the six axes. Sum to 1.0; the resulting
# composite is in [0, 1].
_AXES = (
    "composition",
    "color",
    "narrative_fit",
    "motion_quality",
    "artistic_feel",
    "technical_artifacts",
)
_AXIS_WEIGHTS = {
    "composition": 0.20,
    "color": 0.20,
    "narrative_fit": 0.15,
    "motion_quality": 0.15,
    "artistic_feel": 0.20,
    "technical_artifacts": 0.10,
}


# ============================================================
# Prompt construction & JSON parsing (pure-Python, testable on CPU)
# ============================================================

def build_shot_prompt(shot: Shot, image_paths: list[Path]) -> list[dict]:
    """Build the Qwen2-VL chat messages for one shot evaluation.

    The structure matches the standard transformers vision-chat
    template: a single user turn with N image blocks followed by one
    text block.
    """
    content: list[dict] = []
    for p in image_paths:
        content.append({"type": "image", "image": str(p)})
    instructions = (
        f"You are a cinematographer evaluating one shot from a science-"
        f"visualization film.\n\n"
        f"Shot id: {shot.id}\n"
        f"Renderer: {shot.renderer}\n"
        f"Palette: {shot.palette.name}\n"
        f"Motion intent: {shot.motion_hint or 'unspecified'}\n"
        f"Duration: {shot.duration_frames} frames @ {shot.fps}fps\n\n"
        "The keyframes above are sampled across the shot in time order.\n\n"
        "Score the shot on these six axes (0-10 integers, 10 best):\n"
        "  composition        - framing, focal point clarity\n"
        "  color              - palette execution, channel balance\n"
        "  narrative_fit      - does the motion serve the stated intent?\n"
        "  motion_quality     - smooth motion, no shaking or jumps\n"
        "  artistic_feel      - cinematic; Trumbull/Kubrick worthy?\n"
        "  technical_artifacts - 10 = no visible artifacts; 0 = severe\n\n"
        "Provide one_suggestion: one concrete improvement, <= 25 words.\n\n"
        "Respond as valid JSON ONLY, no markdown fences, no preamble:\n"
        '{"composition": int, "color": int, "narrative_fit": int, '
        '"motion_quality": int, "artistic_feel": int, '
        '"technical_artifacts": int, "one_suggestion": str}'
    )
    content.append({"type": "text", "text": instructions})
    return [{"role": "user", "content": content}]


def parse_vlm_response(text: str) -> dict | None:
    """Parse a VLM response into the expected score dict.

    Tolerant of common LLM quirks: leading/trailing prose, markdown
    code fences, trailing commas. Returns None if no valid score block
    can be recovered.
    """
    if not text:
        return None
    # Strip code fences if present.
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip("` \n")
    # Find the first balanced { ... } block.
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL)
    if not m:
        return None
    block = m.group(0)
    # Best-effort trailing-comma cleanup.
    block = re.sub(r",(\s*[}\]])", r"\1", block)
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None
    # Validate axes are present and in range.
    out = {}
    for axis in _AXES:
        v = data.get(axis)
        if not isinstance(v, (int, float)):
            return None
        out[axis] = max(0, min(10, int(round(v))))
    s = data.get("one_suggestion", "")
    out["one_suggestion"] = str(s)[:240]
    return out


def composite_from_scores(scores: dict) -> float:
    """Weighted average of the six axes, mapped to [0, 1]."""
    total = 0.0
    for axis, weight in _AXIS_WEIGHTS.items():
        total += weight * float(scores.get(axis, 0)) / 10.0
    return float(max(0.0, min(1.0, total)))


def keyframes_for_shot(shot_artifact_dir: Path, max_n: int = 6) -> list[Path]:
    """Pick up to `max_n` keyframe PNGs for a shot artifact, evenly spaced."""
    kdir = Path(shot_artifact_dir) / "keyframes"
    if not kdir.exists():
        return []
    pngs = sorted(kdir.glob("*.png"))
    if not pngs:
        return []
    if len(pngs) <= max_n:
        return list(pngs)
    idx = [int(round(i * (len(pngs) - 1) / (max_n - 1))) for i in range(max_n)]
    return [pngs[i] for i in idx]


# ============================================================
# Critic class
# ============================================================

class LocalVLMCritic:
    """Multimodal VLM critic running locally on the GPU.

    Mirrors the MetricsCritic protocol exactly so Session can use it
    interchangeably (or alongside it in a multi-critic stack).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device_map: str = "auto",
        max_new_tokens: int = 256,
        max_keyframes_per_shot: int = 6,
        torch_dtype: str = "auto",
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.max_keyframes_per_shot = max_keyframes_per_shot
        self.torch_dtype = torch_dtype
        self._model = None
        self._processor = None

    # ---- lazy model load --------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "LocalVLMCritic requires torch. Install it before constructing "
                "the critic, e.g. on Colab `!pip install torch transformers`."
            ) from e
        try:
            from transformers import (
                AutoProcessor,
                Qwen2VLForConditionalGeneration,
            )
        except ImportError as e:
            raise RuntimeError(
                "LocalVLMCritic requires transformers. Install with "
                "`pip install transformers accelerate qwen-vl-utils`."
            ) from e

        dtype = self.torch_dtype
        if dtype == "auto":
            import torch
            dtype = (
                torch.float16
                if torch.cuda.is_available()
                else torch.float32
            )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device_map,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)

    # ---- public API (matches MetricsCritic) -------------------------

    def evaluate_shot_artifact(
        self,
        shot: Shot,
        shot_artifact_dir: Path | str,
    ) -> ShotMetrics:
        self._ensure_loaded()
        image_paths = keyframes_for_shot(
            Path(shot_artifact_dir), max_n=self.max_keyframes_per_shot
        )
        if not image_paths:
            return _empty_shot_metrics(shot, note="no keyframes found")
        messages = build_shot_prompt(shot, image_paths)
        text_response = self._chat(messages)
        scores = parse_vlm_response(text_response)
        if scores is None:
            return _empty_shot_metrics(
                shot, note=f"vlm response failed to parse: {text_response[:200]!r}"
            )
        return _scores_to_shot_metrics(shot, scores, n_keyframes=len(image_paths))

    def evaluate_artifact(self, artifact_dir: Path | str) -> MetricSet:
        artifact_dir = Path(artifact_dir)
        graph = load_manifest(artifact_dir / "manifest.json")
        shots: dict[str, ShotMetrics] = {}
        for shot in graph.shots:
            sdir = artifact_dir / "shots" / shot.id
            if not sdir.exists():
                continue
            metrics = self.evaluate_shot_artifact(shot, sdir)
            shots[shot.id] = metrics
            (sdir / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2))
        aggregate = _aggregate(shots, graph)
        (artifact_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "critic": "LocalVLMCritic",
                    "model": self.model_name,
                    "shots": {k: v.to_dict() for k, v in shots.items()},
                    "aggregate": aggregate,
                },
                indent=2,
            )
        )
        return MetricSet(shots=shots, aggregate=aggregate)

    # ---- internal: model chat ---------------------------------------

    def _chat(self, messages: list[dict]) -> str:
        """Run one Qwen2-VL chat turn. Returns the assistant text."""
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None
        import torch

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # qwen_vl_utils is the canonical helper, but we can also extract
        # the images by walking the messages if it isn't available.
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs = _extract_images(messages)
            video_inputs = None
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        text_out = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return text_out[0] if text_out else ""


# ============================================================
# Helpers
# ============================================================

def _extract_images(messages):
    """Fallback image extraction when qwen_vl_utils isn't installed."""
    from PIL import Image
    images = []
    for m in messages:
        for c in m.get("content", []):
            if c.get("type") == "image":
                images.append(Image.open(c["image"]).convert("RGB"))
    return images


def _scores_to_shot_metrics(
    shot: Shot, scores: dict, n_keyframes: int
) -> ShotMetrics:
    composite = composite_from_scores(scores)
    sugg = scores.get("one_suggestion", "")
    axis_summary = ", ".join(
        f"{axis} {scores[axis]}/10" for axis in _AXES if axis in scores
    )
    notes = [f"VLM: {axis_summary}"]
    if sugg:
        notes.append(f"suggestion: {sugg}")
    return ShotMetrics(
        shot_id=shot.id,
        frames_seen=n_keyframes,
        mean_luma=0.0,
        mean_contrast=0.0,
        mean_dyn_range_db=0.0,
        blown_frac=0.0,
        black_frac=0.0,
        edge_density=0.0,
        palette_distance=-1.0,
        temporal_abs_diff=0.0,
        frozen_pair_count=0,
        degenerate_frame_count=0,
        nan_pixels=0,
        inf_pixels=0,
        composite_score=composite,
        notes=notes,
    )


def _empty_shot_metrics(shot: Shot, note: str) -> ShotMetrics:
    return ShotMetrics(
        shot_id=shot.id,
        frames_seen=0,
        mean_luma=0.0,
        mean_contrast=0.0,
        mean_dyn_range_db=0.0,
        blown_frac=0.0,
        black_frac=0.0,
        edge_density=0.0,
        palette_distance=-1.0,
        temporal_abs_diff=0.0,
        frozen_pair_count=0,
        degenerate_frame_count=0,
        nan_pixels=0,
        inf_pixels=0,
        composite_score=0.0,
        notes=[note],
    )


def _aggregate(shots: dict[str, ShotMetrics], graph: ShotGraph) -> dict:
    if not shots:
        return {"shot_count": 0, "composite_score": 0.0}
    weights = [graph.shot_by_id(sid).duration_frames for sid in shots]
    scores = [m.composite_score for m in shots.values()]
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
    }
