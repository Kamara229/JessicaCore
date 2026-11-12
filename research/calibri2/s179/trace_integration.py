"""S179 research-only observational hooks around frozen S176/S160 CPU decoding.

Does NOT modify S160, S168, S176, model weights, selection or chosen tokens.
Capture the actual autoregressive second-step logits through a read-only
deterministic replay *after* the original decoder has produced its answer.
No labels enter this file's replay, selector or tracing paths.
Never run this module in a Jessica chat, production, Render or Supabase.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from calibri2.s176 import s174_legacy_compatible_j7 as s176
from calibri2.s168 import novel_train_frames_generalization_cpu as s168
from calibri2.s160 import real_parallel_slots_j7 as s160
from calibri2.s179.trace_contract import S179TraceCollector

ORIGINAL_AUDIT = s168.s160.audit_cpu_outputs
ORIGINAL_DECODER = s168.ORIGINAL_DECODER
COLLECTOR = S179TraceCollector()
REPLAY_BY_SHA: dict[str, dict] = {}
CAPTURING = False
FIRST_COHORT_DONE = False


def _sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _conditional_step2_replay(
    model, sources, first, later, lm_head, tok, question, pointer, generated,
):
    """Replay only token-1 prefill and token-2 conditional forward, READ ONLY.

    Identical S160/101 logits computation through step==1, including pointer
    score adjustment. The copy_net may override the final second token AFTER
    these logits; this is recorded as a separate decoder guard note. A
    two-token fastpath never evaluates autoregressive step 2: return None.
    """
    if not generated or generated[0] == tok.eos_token_id:
        return None
    from calibri2.s82 import real_mps_generic_token_context_copy as s82
    from v176.operators import clear

    ids = s160.prompt_tokens(tok, question)
    kept = s82.candidate_positions(tok, ids) if pointer is not None else []
    source_ids = torch.tensor([ids[i] for i in kept], dtype=torch.long)
    with torch.inference_mode():
        clear(sources)
        prefill = model(
            input_ids=torch.tensor([ids], dtype=torch.long),
            past_key_values=None, use_cache=True, output_hidden_states=True)
        cache = prefill.past_key_values
        if cache is None:
            raise AssertionError("S179 replay missing frozen S176 prefill KV cache")
        h0 = prefill.hidden_states[-1][0]
        first_logits = prefill.logits[0, -1] + F.linear(first(h0[-1]), lm_head)
        if int(first_logits.argmax()) != generated[0]:
            raise AssertionError("S179 replay differs from actual S160 first token")
        keys = h0[torch.tensor(kept, dtype=torch.long)].detach().clone() if kept else None
        del prefill
        clear(sources)
        second = model(
            input_ids=torch.tensor([[generated[0]]], dtype=torch.long),
            past_key_values=cache, use_cache=True, output_hidden_states=True)
        h1 = second.hidden_states[-1][0][-1]
        logits = second.logits[0, -1] + F.linear(later(h1), lm_head)
        if pointer is not None and kept and int(logits.argmax()) != tok.eos_token_id:
            bonus = pointer.full_vocab_bonus(h1, keys, source_ids)
            logits = logits + torch.zeros_like(logits).index_add(
                0, source_ids, bonus)
        out = logits.detach().cpu().float().tolist()
        del second, cache
        clear(sources)
    return out


def _traced_decoder(
    model, sources, first, later, lm_head, tok, question, pointer,
    copy_net, parallel_head,
):
    """Execute original frozen decoder first; replay only for observation."""
    generated, route_note = ORIGINAL_DECODER(
        model, sources, first, later, lm_head, tok, question, pointer,
        copy_net, parallel_head)
    if CAPTURING:
        sha = _sha(question)
        if sha in REPLAY_BY_SHA:
            raise AssertionError("S179 repeated prompt inside original cohort")
        route = dict(route_note)
        fastpath = bool(route.get("parallel"))
        logits = None if fastpath else _conditional_step2_replay(
            model, sources, first, later, lm_head, tok, question, pointer,
            generated)
        # Real same-KV replay parity: a non-overridden frozen autoregressive
        # step-2 token MUST equal the actual conditional logit argmax.
        # Neural copy may override token 2 only after these scores are formed.
        if (logits is not None and len(generated) >= 2 and
                not route.get("neural_override_applied")):
            replay_second_id = max(range(len(logits)), key=logits.__getitem__)
            if replay_second_id != generated[1]:
                raise AssertionError(
                    "S179 real conditional step2 logits differ from frozen "
                    "S160 generated token (KV/profile/replay drift)")
        REPLAY_BY_SHA[sha] = {
            "conditional": logits,
            "generated": list(generated),
            "route_note": route,
            "fastpath": fastpath,
        }
    return generated, route_note


def _traced_audit(
    cpu, old, new, old_ids, new_ids, routes, tok, bank,
    copy_net, bridge, full, expansion, parent_old, parent_new, head,
):
    """Only the complete original TRAIN cohort; never pass expected to trace."""
    global CAPTURING, FIRST_COHORT_DONE
    original = len(old) == 456 and len(new) == 128 and len(old_ids) == 456 and len(new_ids) == 128
    if not original or FIRST_COHORT_DONE:
        return ORIGINAL_AUDIT(
            cpu, old, new, old_ids, new_ids, routes, tok, bank, copy_net,
            bridge, full, expansion, parent_old, parent_new, head)
    CAPTURING = True
    try:
        outcome = ORIGINAL_AUDIT(
            cpu, old, new, old_ids, new_ids, routes, tok, bank, copy_net,
            bridge, full, expansion, parent_old, parent_new, head)
    finally:
        CAPTURING = False
    predicted_old = outcome[0]
    if set(predicted_old) != set(old):
        raise AssertionError("S179 original TRAIN 456 output coverage missing")
    # Record output and route facts, but NO TRAIN gold or expected answer ids.
    for key in sorted(old):
        question = old[key]["question"]  # ONLY field accessible from source row
        sha = _sha(question)
        generated = list(predicted_old[key])
        route = routes[key]
        events = []
        if route == "S93":
            replay = REPLAY_BY_SHA.get(sha)
            if replay is None or replay["generated"] != generated:
                raise AssertionError("S179 missing or mismatched S93 replay")
            conditional = replay["conditional"]
            events.extend(
                str(k) for k, v in s168.S167_ROUTE_TRACE[question]["counter_deltas"].items()
                if v)
            if replay["fastpath"]:
                events.append("fastpath_no_autoregressive_step2")
            for k in ("candidate_present", "neural_override_applied"):
                if replay["route_note"].get(k):
                    events.append(k)
        else:
            conditional = None
            events.append("non_S93_frozen_parent_only_no_replay")
        COLLECTOR.record(
            prompt=question, route=route, generated=generated,
            eos_id=tok.eos_token_id,
            conditional_step2_logits=conditional,
            guard_events=events)
    if len(COLLECTOR.traces) != 456:
        raise AssertionError("S179 expected complete 456 original TRAIN traces")
    FIRST_COHORT_DONE = True
    return outcome


def main():
    if len(sys.argv) < 2:
        raise SystemExit("S179 requires S176 pinned source inputs and output path")
    out = Path(sys.argv[-1])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "S179_ORIGINAL456_SOURCE_ONLY_TRACES.json").exists():
        raise FileExistsError("S179 diagnostic must not overwrite prior trace")
    s168.s160.audit_cpu_outputs = _traced_audit
    s168.ORIGINAL_DECODER = _traced_decoder
    try:
        # Execution only when explicitly launched in a research-only workflow.
        s176.main()
    finally:
        s168.s160.audit_cpu_outputs = ORIGINAL_AUDIT
        s168.ORIGINAL_DECODER = ORIGINAL_DECODER
    if not FIRST_COHORT_DONE:
        raise AssertionError("S179 full source cohort was not encountered")
    COLLECTOR.export(out / "S179_ORIGINAL456_SOURCE_ONLY_TRACES.json")
    # Posthoc labels are intentionally NOT joined by this inference program.
    (out / "S179_DIAGNOSTIC_STATUS.json").write_text(json.dumps({
        "stage": "S179_READ_ONLY_456_SECOND_TOKEN_EOS_CAPTURE",
        "trace_count": 456, "expected_labels_in_trace": False,
        "source_weights_changed": False, "model_accepted": False,
        "no_training": True, "no_QA_TEST": True,
        "Jessica_LIVE_Render_Supabase_APK_untouched": True,
    }, indent=2, sort_keys=True))
    print(json.dumps({"event": "S179_TRACE_456_RECORDED_NO_GOLD",
                      "cases": 456, "model_accepted": False}), flush=True)


if __name__ == "__main__":
    main()
