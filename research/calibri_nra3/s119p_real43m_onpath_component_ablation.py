"""S119p: causal component ablation INSIDE frozen ORIGINAL 43M+GT decoder.

Analyze why physically trained S119m NRA3/structured/core/NPA1 changed 0/6
autoregressive native answers in S119n and why S119o certified stable top1
at 16/16 native positions. Only the SIX PRE-DECLARED UNLABELED S119n questions
and their first generated native token are audited.

At each physical original prefix, evaluate actual full 49,152-head logits
under: original baseline; S119m FULL; no CORE; no STRUCTURE; no NPA1; ZERO.
Components are turned off temporarily ONLY in scoped research gate masking,
never tuned to any answers. Compare 320D weighted contributions and their
cancellation, output full-vocab L-inf perturbation, top1 margin/stability,
and counterfactual no-one-component on-path logits. This is diagnostic:
leave-one-out is not a Shapley effect, structural context competence is NOT
measured; source prompts are neither novel blind QA nor answer labels.

No training, no optimizer, no new weights, no S118 SynTagRusB GOLD / old
EVAL, no J7-616 replay, bpp or paired CPU gate. The approach requires
TWO physical original decoder forward calls even for FULL, not one-pass
CPU-compatible architecture. No LIVE/main/Render/Supabase/Jessica/APK.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original, sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import (
    PROMPTS, STAGE as S119N_STAGE, REPORT as S119N_REPORT,
    load_parent, prompt_features, first_original_pass,
    actual_original_onpath_second_pass,
)
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import (
    strict_chat_source_ids,
)
from calibri2.s160 import real_parallel_slots_j7 as s160

STAGE = "CALIBRI_S119P_REAL_S142_43M_GT_S119M_COMPONENT_ABLATION_GATE0"
REPORT = "S119P_REAL43M_NATIVE_FIRST_TOKEN_COMPONENT_ABLATION_ONLY.json"
COMPONENTS = ("core_growth", "structured_NRA3", "NPA1_planning")
MASKS = {
    "trained_full": (1, 1, 1),
    "without_core": (0, 1, 1),
    "without_structure": (1, 0, 1),
    "without_planning": (1, 1, 0),
    "zero_exact_original": (0, 0, 0),
}
SEED = 119128


@contextmanager
def temporary_gate_mask(adapter, mask):
    """Never persist or mutate parent S119m learned FP16 gates/weights."""
    check(tuple(mask) in MASKS.values() and len(mask) == 3,
          "pre-registered masks only, no gate amplification")
    old = adapter.midlayer_gates.detach().clone()
    try:
        with torch.no_grad():
            adapter.midlayer_gates.copy_(
                old * torch.tensor(mask, dtype=old.dtype, device=old.device))
        yield
    finally:
        with torch.no_grad():
            adapter.midlayer_gates.copy_(old)
        check(torch.equal(adapter.midlayer_gates.detach(), old),
              "S119m original learned three gates must be restored")


def first_token_metrics(base, variants, components, gates):
    """Actual native LM-head A/B with pure tensor validity and margin math."""
    check(base.ndim == 2 and base.shape[0] == 1 and base.shape[1] >= 3
          and set(variants) == set(MASKS)
          and all(x.shape == base.shape for x in variants.values())
          and components.shape == (1, 3, 320) and gates.shape == (3,),
          "same full native logits and three physical H320 components")
    check(bool(torch.isfinite(base).all())
          and bool(torch.isfinite(components).all())
          and bool(torch.isfinite(gates).all())
          and all(bool(torch.isfinite(x).all()) for x in variants.values()),
          "all real on-path probes and weights finite")
    zero = variants["zero_exact_original"]
    check(torch.allclose(base, zero, atol=2e-4, rtol=1e-5)
          and torch.equal(base.argmax(-1), zero.argmax(-1)),
          "ZERO-gate physical original decoder and 49152 logits parity")
    top2 = torch.topk(base[0].float(), k=2).values
    margin = float((top2[0] - top2[1]).item())
    full = variants["trained_full"]
    full_max_shift = float((full - base).abs().max().item())
    source_native = int(base.argmax(-1).item())
    trained_native = int(full.argmax(-1).item())
    component_vectors = (
        components.detach().float()[0]
        * torch.tanh(gates.detach().float()).unsqueeze(-1)
    )
    component_norms = [float(torch.linalg.vector_norm(v).item())
                       for v in component_vectors]
    residual = component_vectors.sum(dim=0)
    sum_norms = sum(component_norms)
    residual_norm = float(torch.linalg.vector_norm(residual).item())
    cancellation_ratio = (
        max(0., min(1., 1. - residual_norm / sum_norms))
        if sum_norms > 1e-15 else 0.
    )
    without = {}
    for name, component in zip(
        ("without_core", "without_structure", "without_planning"), COMPONENTS
    ):
        counterfactual = variants[name]
        without[component] = {
            "native_first_id_without_component": int(
                counterfactual.argmax(-1).item()),
            "max_abs_full_vocab_logits_FULL_vs_without": float(
                (full-counterfactual).abs().max().item()),
            "max_abs_full_vocab_logits_BASE_vs_without": float(
                (base-counterfactual).abs().max().item()),
            "candidate_token_changed_when_component_omitted": bool(
                int(counterfactual.argmax(-1).item()) != trained_native),
        }
    return {
        "original_native_first_id": source_native,
        "trained_full_native_first_id": trained_native,
        "same_native_first_token": source_native == trained_native,
        "original_top1_top2_margin": margin,
        "full_onpath_max_abs_native_vocab_logit_shift": full_max_shift,
        "sufficient_argmax_stability_margin_gt_2x_Linf":
            margin > 2. * full_max_shift,
        "trained_gates_tanh": [float(v) for v in
                               torch.tanh(gates.detach().float()).tolist()],
        "weighted_H320_component_norms": dict(zip(
            COMPONENTS, component_norms)),
        "full_H320_residual_norm": residual_norm,
        "sum_individual_weighted_norms": sum_norms,
        "component_cancellation_ratio_0_none_1_total": cancellation_ratio,
        "causal_component_leave_one_out_native_logits": without,
        "zero_gate_full_vocab_original_parity": True,
    }


def verify_original_research_report(report, s93, s142, parent):
    check(report.get("stage") == S119N_STAGE
          and report.get("output_gate") == 0
          and report.get("candidate_accepted") is False
          and report.get("physical_full_original_native_vocab_size") == 49152
          and report.get("n_unlabeled_author_prompts") == len(PROMPTS) == 6
          and report.get("prompts_where_trained_onpath_changes_generated_native_BPE") == 0
          and report.get("total_actual_original43m_forward_calls") == 54
          and report.get("no_heldout_QA_gold_answers") is True
          and report.get("original_616_J7_EOS_NOT_replayed") is True
          and report.get("no_live_activation") is True
          and report.get("original_S93_CB61_SHA256") == sha(s93)
          and report.get("original_S142_43M_GT_SHA256") == sha(s142)
          and report.get("S119m_trained_complete_physical_FP16_SHA256") ==
              sha(parent),
          "S119n genuine immutable ORIGINAL 43M+GT research ancestry only")
    records = report.get("research_records")
    check(isinstance(records, list) and len(records) == len(PROMPTS),
          "S119n exactly six physical native generation records")
    for rec, (group, question) in zip(records, PROMPTS):
        check(rec.get("category") == group
              and rec.get("source_question") == question
              and rec.get("initial_question_SHA256") ==
                  hashlib.sha256(question.encode("utf8")).hexdigest()
              and rec.get("answer_grade_NOT_MEASURED") is True
              and rec.get("different_native_generated_token_sequences") is False,
              "S119p no new prompts or quality GOLD, exact original sources only")
        for key in ("original_native_greedy", "actual_deep_onpath_native_greedy"):
            item=rec.get(key, {})
            check(isinstance(item.get("generated_native_token_ids"), list)
                  and len(item["generated_native_token_ids"]) > 0
                  and item["generated_native_token_ids"][0] >= 0,
                  "S119n first native token must be source-only")
    return records


def run(s93: Path, side: Path, s142: Path, parent: Path,
        parent_report: Path, s119n_report: Path, out: Path):
    check(not out.exists() and s119n_report.name == S119N_REPORT,
          "isolated immutable prior S119n research JSON, no overwrite")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    prior = json.loads(s119n_report.read_text(encoding="utf8"))
    records = verify_original_research_report(prior, s93, s142, parent)
    _, adapter, original_old_fp16 = load_parent(
        s93, side, s142, parent, parent_report)
    model, sources, bridge, full, expansion = load_actual_original(
        s93, side, s142)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL, MODEL_SHA
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    check(tok.vocab_size == 49152, "S142 original native LM vocabulary only")
    model.model.add_module("nra3_s119p_diag", adapter)
    model.eval()
    original_sha = old_original_param_digest(model)
    first_hooks = set(model.model.layers[5]._forward_hooks)
    learned_gates = adapter.midlayer_gates.detach().clone()
    check(adapter.output_gate == 0 and not adapter.candidate_accepted
          and not adapter.activation_allowed,
          "research candidate gate0, never LIVE activation")
    observations = []
    physical_forwards = 0
    for (group, question), previous in zip(PROMPTS, records):
        native_prefix = strict_chat_source_ids(tok, question)
        features = prompt_features(tok, question)
        base, original12 = first_original_pass(
            model, sources, bridge, full, expansion, native_prefix, features)
        physical_forwards += 1
        with torch.enable_grad():
            components = adapter.delta_components(original12, contextual=True)
        check(components.shape == (1, 3, 320),
              "all THREE learned internal regions actually active in source")
        variants = {}
        for name, mask in MASKS.items():
            with temporary_gate_mask(adapter, mask):
                variants[name] = actual_original_onpath_second_pass(
                    model, sources, bridge, full, expansion, adapter,
                    native_prefix, features, original12)
            physical_forwards += 1
        metrics = first_token_metrics(
            base, variants, components, learned_gates)
        check(metrics["original_native_first_id"] ==
              previous["original_native_greedy"]["generated_native_token_ids"][0]
              and metrics["trained_full_native_first_id"] ==
              previous["actual_deep_onpath_native_greedy"]["generated_native_token_ids"][0],
              "S119p must REPRODUCE previously published first native tokens")
        check(set(model.model.layers[5]._forward_hooks) == first_hooks
              and torch.equal(adapter.midlayer_gates, learned_gates),
              "no persistent hooks, no gate updates between physical probes")
        observations.append({
            "group": group,
            "source_question_SHA256":
                hashlib.sha256(question.encode("utf8")).hexdigest(),
            "original_native_prefix_length": len(native_prefix),
            **metrics,
        })
    check(physical_forwards == 6*(1 + len(MASKS)) == 36
          and old_original_param_digest(model) == original_sha,
          "all exact frozen original 43M+GT remain unchanged")
    inherited_prefix = "model.nra3_calibri_joint.s119m_onpath."
    for full_name, old in original_old_fp16.items():
        short = full_name[len(inherited_prefix):]
        check(torch.equal(adapter.state_dict()[short].detach().half().cpu(), old),
              "every old S115 trained FP16 neural weight unchanged")
    check(torch.equal(adapter.midlayer_gates, learned_gates)
          and set(model.model.layers[5]._forward_hooks) == first_hooks
          and not adapter.candidate_accepted and adapter.output_gate == 0,
          "no mutated trained new gate, no lingering original-layer hook")
    report = {
        "stage": STAGE,
        "original_s119n_source_report_SHA256": sha(s119n_report),
        "original_s119m_trained_research_checkpoint_SHA256": sha(parent),
        "original_s142_43M_GT_SHA256": sha(s142),
        "original_s93_CB61_SHA256": sha(s93),
        "source_prompts_not_blind_heldout_or_quality": True,
        "original_full_native_token_vocab": tok.vocab_size,
        "real_original_cpu_forward_calls": physical_forwards,
        "six_original_s119n_fixed_first_token_sources_only": True,
        "no_training_no_optimizer_no_gate_amplification": True,
        "original_full_43M_GT_and_old_53751_FP16_unchanged": True,
        "preexisting_S119m_all_FP16_unchanged": True,
        "no_new_616_J7_EOS_replay": True,
        "full_bpp_and_paired_KV_CPU_NOT_MEASURED": True,
        "original_first_full12_prefix_pass_is_NOT_one_pass_CPU_design": True,
        "not_independent_generalization_and_no_answer_grading": True,
        "not_causal_shapley_or_definitive_skill_attribution": True,
        "protected_S118_B_GOLD_NOT_LOADED": True,
        "no_spent_EVAL128_or_DEV": True,
        "no_LIVE_main_Render_Supabase_Jessica_APK": True,
        "candidate_accepted": False,
        "output_gate": 0,
        "research_observations": observations,
        "summary": {
            "count_prompt_first_tokens": len(observations),
            "full_vs_baseline_native_first_token_changes": sum(
                not x["same_native_first_token"] for x in observations),
            "first_tokens_certified_stable_by_margin": sum(
                x["sufficient_argmax_stability_margin_gt_2x_Linf"]
                for x in observations),
            "max_first_token_full_native_logits_Linf": max(
                x["full_onpath_max_abs_native_vocab_logit_shift"]
                for x in observations),
            "max_three_component_cancellation_ratio": max(
                x["component_cancellation_ratio_0_none_1_total"]
                for x in observations),
            "max_leave_one_out_logit_Linf_by_component": {
                component: max(
                    x["causal_component_leave_one_out_native_logits"][component][
                        "max_abs_full_vocab_logits_FULL_vs_without"]
                    for x in observations)
                for component in COMPONENTS},
        },
        "decision": "DIAGNOSTIC_ONLY_NO_QUALITY_OR_ACCEPTANCE_CLAIMS",
    }
    out.mkdir(parents=True)
    (out / REPORT).write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf8")
    print("S119P_REAL_ORIGINAL_ONPATH_COMPONENT_ABLATION " +
          json.dumps(report["summary"] | {
              "original_pretrained_exact": True, "old_NRA3_FP16_exact": True,
              "accepted": False
          }, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for key in ("s93","side","s142","parent","parent_report",
                "s119n_report","out"):
        parser.add_argument("--" + key.replace("_","-"),
                            required=True, type=Path)
    args = parser.parse_args()
    run(args.s93,args.side,args.s142,args.parent,args.parent_report,
        args.s119n_report,args.out)


if __name__ == "__main__":
    main()
