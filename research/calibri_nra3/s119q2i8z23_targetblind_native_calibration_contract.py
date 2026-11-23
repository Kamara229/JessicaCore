"""Z23: frozen S142/Z22 true native49k reranking calibration, target-blind.

No new model training; original logits and same-paid actual h320 read.
Both original model and trained aligned/permuted 10240-param Z22
shadows FROZEN. Original native top64 fixed BEFORE predictions.
For each state, compute four precommitted scales 0,2,8,16 using
only the original native BPE2 logits and frozen model prediction,
without DEV human gold. Original model branch not run per candidate.
Gold is used ONLY in separate posthoc diagnostic function after all
choices generated. No oracle candidate insertion, masking of EOS, or
deployment rule derived from sample.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z23_new28dev_frozen_future_calibration_source import (
    SCALES,TOP_K,
)
from research.calibri_nra3.s119q2i8z21_future_state_lowrank16_matched_contract import (
    TinyFutureState,STATES,VOCAB,HIDDEN,
)
EOS_ID=2

def check(c,m):
    if not c:raise ValueError("Z23 target-blind frozen predict-state calibration: "+m)

@torch.no_grad()
def target_blind_calibrated_candidates(native_logits,actual_final_h320,
                                       frozen_orig_native_embedding,shadows):
    check(isinstance(native_logits,torch.Tensor)
        and native_logits.shape==(1,VOCAB)
        and bool(torch.isfinite(native_logits).all())
        and isinstance(actual_final_h320,torch.Tensor)
        and actual_final_h320.shape==(1,HIDDEN)
        and bool(torch.isfinite(actual_final_h320).all())
        and isinstance(frozen_orig_native_embedding,torch.Tensor)
        and frozen_orig_native_embedding.shape==(VOCAB,HIDDEN)
        and not frozen_orig_native_embedding.requires_grad
        and set(shadows)==set(STATES)
        and all(isinstance(shadows[x],TinyFutureState)
                and shadows[x].output_gate==0
                and shadows[x].candidate_accepted is False
                and not any(p.requires_grad for p in shadows[x].parameters())
                for x in STATES)
        and SCALES==(0.,2.,8.,16.) and TOP_K==64,
        "original full49152 native BPE logits; real h320; four locked scales; frozen shadow")
    native=native_logits.detach().float()[0]
    ids=torch.topk(native,TOP_K,sorted=True).indices
    e=F.normalize(
        frozen_orig_native_embedding.detach().index_select(0,ids).float(),
        dim=-1,eps=1e-6)
    orig_scores=native.index_select(0,ids)
    output={
        "original_NATIVE_GREEDY_BPE2_ID":int(ids[0]),
        "original_TARGET_BLIND_TOP64_NATIVE_BPE_IDS":[int(i) for i in ids.tolist()],
        "baseline_GREEDY_TOP64_SCORE_SPREAD":
            float(orig_scores[0]-orig_scores[-1]),
        "EOS_preselected_original_TOP64":bool((ids==EOS_ID).any()),
        "states":{},
        "source_gold_NOT_USED_in_any_candidate_or_scale":True,
        "NO_training_or_extra_43M_forwards":True,
    }
    for state in STATES:
        pred=shadows[state](actual_final_h320).detach().float()
        check(pred.shape==(1,HIDDEN) and torch.isfinite(pred).all(),
              "real frozen Z22 predicted future native h320")
        sim=(e@pred[0])
        check(sim.shape==(TOP_K,)
              and bool(torch.isfinite(sim).all())
              and bool((sim.abs()<=1.00001).all()),
              "full original frozen 64 native embedding cosine")
        by_scale={}
        for scale in SCALES:
            scores=orig_scores + scale*sim
            selected=int(ids[scores.argmax()].item())
            by_scale[str(int(scale))]={
                "choice_native_BPE2_ID":selected,
                "changed_from_original_greedy":selected!=int(ids[0]),
                "chose_EOS":selected==EOS_ID,
                "native_original_logit_of_choice":float(native[selected]),
                "cosine_of_choice":float(sim[ids==selected][0]),
            }
        check(by_scale["0"]["choice_native_BPE2_ID"]==int(ids[0]),
              "zero scale exactly original greedy identity")
        output["states"][state]={
            "future_h320_l2":float(torch.linalg.vector_norm(pred)),
            "original_top64_future_cosines":[float(v) for v in sim.tolist()],
            "scale_choices":by_scale,
        }
    return output

def posthoc_human_gold(d,correct_first:bool,human_BPE2:int,
                       actual_native_logits):
    """ONLY after target-free all-scale predictions; no model calls."""
    check(d.get("source_gold_NOT_USED_in_any_candidate_or_scale") is True
          and type(correct_first) is bool
          and type(human_BPE2) is int and 0<=human_BPE2<VOCAB
          and len(d["original_TARGET_BLIND_TOP64_NATIVE_BPE_IDS"])==TOP_K
          and set(d["states"])==set(STATES)
          and isinstance(actual_native_logits,torch.Tensor)
          and actual_native_logits.shape==(1,VOCAB),
          "gold is allowed POSTHOC only for already-selected native proposals")
    ids=d["original_TARGET_BLIND_TOP64_NATIVE_BPE_IDS"]
    logits=actual_native_logits.detach().float()[0]
    orig_best=ids[0]
    reachable=bool(correct_first and human_BPE2 in ids)
    rank=1+int((logits>logits[human_BPE2]).sum().item()) if correct_first else None
    gap=float(logits[orig_best]-logits[human_BPE2]) if correct_first else None
    evidence={
        "correct_actual_OWN_FIRST":correct_first,
        "original_native49152_second_gold_rank_if_correct_OWN_FIRST":rank,
        "original_native_logit_gap_greedy_minus_gold_if_correct_OWN_FIRST":gap,
        "gold_BPE2_in_original_TOP64_after_correct_first":reachable,
        "original_greedy_BPE2_correct":bool(correct_first and orig_best==human_BPE2),
        "per_frozen_trained_state":{},
        "human_gold_only_POSTHOC_no_inference_or_training":True,
    }
    for state,s in d["states"].items():
        indexes={str(k):int(v["choice_native_BPE2_ID"])
                 for k,v in s["scale_choices"].items()}
        res={
          "second_gold_correct_at_fixed_scale":{k:bool(correct_first and choice==human_BPE2)
                                                 for k,choice in indexes.items()},
          "changed_from_original_greedy_by_scale":{k:choice!=orig_best
                                                    for k,choice in indexes.items()},
          "premature_EOS_by_scale":{k:choice==EOS_ID for k,choice in indexes.items()},
          "chosen_native_BPE_ID_by_scale":indexes,
          "max_cosine_gap_gold_minus_original_greedy_if_reachable":None,
          "theoretical_scale_to_beat_original_greedy_POSTHOC_NOT_a_parameter":None,
        }
        if reachable:
            gold_i=ids.index(human_BPE2)
            gold_s=s["original_top64_future_cosines"][gold_i]
            greedy_s=s["original_top64_future_cosines"][0]
            delta=float(gold_s-greedy_s)
            res["max_cosine_gap_gold_minus_original_greedy_if_reachable"]=delta
            # Only a necessary condition for second gold to outrank the
            # original greedy, NOT enough to beat ALL other 63 candidates.
            res["theoretical_scale_to_beat_original_greedy_POSTHOC_NOT_a_parameter"]=(
                max(0.,float(gap/delta)) if delta>1e-12 else None)
        evidence["per_frozen_trained_state"][state]=res
    return evidence
