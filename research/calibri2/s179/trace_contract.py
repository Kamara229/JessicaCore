"""S179: research-only, inference-time trace collector; never accepts TRAIN gold.

The caller must feed actual conditional second-step decoder logits. S160 span
head scores are NOT conditional token logits and must not be substituted.
For fastpath / non-S93 rows with no second step, use logits=None.
No tests, model execution, or filesystem writes happen on import.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class S179TraceCollector:
    """Store one final inference-only trace per prompt, without expected labels."""

    traces: dict[str, dict] = field(default_factory=dict)

    def record(
        self, *, prompt: str, route: str, generated: list[int], eos_id: int,
        conditional_step2_logits: list[float] | None,
        guard_events: list[str], source_votes_5: int | None = None,
        source_votes_10: int | None = None,
    ) -> str:
        if not isinstance(prompt, str) or not prompt or not isinstance(route, str):
            raise ValueError("nonempty prompt and route required")
        if not isinstance(generated, list) or not generated or not all(
            isinstance(x, int) and x >= 0 for x in generated
        ):
            raise ValueError("actual generated ids required")
        if not isinstance(eos_id, int) or eos_id < 0:
            raise ValueError("EOS token id required")
        if any(not isinstance(x, str) for x in guard_events):
            raise ValueError("guard events must be inference-only names")
        for votes, maximum in ((source_votes_5, 5), (source_votes_10, 10)):
            if votes is not None and (not isinstance(votes, int) or not 0 <= votes <= maximum):
                raise ValueError("invalid votes")
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if sha in self.traces:
            raise ValueError("each prompt must be recorded exactly once")
        conditional = None
        if conditional_step2_logits is not None:
            values = conditional_step2_logits
            if not isinstance(values, list) or eos_id >= len(values):
                raise ValueError("actual conditional logits must include EOS index")
            if any(not isinstance(x, (float, int)) or math.isnan(x) for x in values):
                raise ValueError("invalid conditional logits")
            candidates = sorted(
                ((float(score), idx) for idx, score in enumerate(values)
                 if math.isfinite(score)), key=lambda row: (-row[0], row[1]))
            if not candidates:
                raise ValueError("no finite conditional logits")
            conditional = {
                "top5_token_ids_and_logits": [
                    [idx, score] for score, idx in candidates[:5]],
                "eos_logit": float(values[eos_id]),
                "eos_rank": next((i + 1 for i, (_, idx) in enumerate(candidates)
                                  if idx == eos_id), None),
                "score_type": "actual_ar_decoder_conditional_step2_logits",
            }
        self.traces[sha] = {
            "prompt_sha256": sha,
            "route": route,
            "full_generated_ids": list(generated),
            "terminated_in_eos": generated[-1] == eos_id,
            "guard_events": list(guard_events),
            "consensus_votes5": source_votes_5,
            "consensus_votes10": source_votes_10,
            "conditional_step2": conditional,
            "gold_seen_at_inference": False,
        }
        return sha

    def export(self, destination: Path, *, expected_count: int = 456) -> None:
        """Export only complete inference records; caller controls filesystem."""
        if len(self.traces) != expected_count:
            raise AssertionError(
                f"S179 incomplete source coverage {len(self.traces)}/{expected_count}")
        destination.write_text(
            json.dumps({"stage": "S179_TRACE_ONLY", "number": expected_count,
                        "labels_used": False,
                        "cases": [self.traces[k] for k in sorted(self.traces)]},
                       ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )


def posthoc_score(traces: dict[str, dict], expected_by_prompt_sha: dict[str, list[int]]):
    """Callable ONLY after export/inference; never imported by a selector."""
    if set(traces) != set(expected_by_prompt_sha):
        raise AssertionError("S179 posthoc 456 key coverage differs")
    misses = []
    for sha in sorted(traces):
        generated = traces[sha]["full_generated_ids"]
        expected = expected_by_prompt_sha[sha]
        if generated != expected:
            misses.append({
                "prompt_sha256": sha,
                "first_token_error": not generated or generated[0] != expected[0],
                "second_token_error": (
                    len(expected) > 2 and len(generated) > 1 and
                    generated[0] == expected[0] and generated[1] != expected[1]),
                "eos_error": generated[-1] != expected[-1],
                "route": traces[sha]["route"],
                "guard_events": traces[sha]["guard_events"],
            })
    return {"total": len(traces), "full_eos_correct": len(traces) - len(misses),
            "incorrect_count": len(misses), "incorrect": misses}
