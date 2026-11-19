"""NRA2 S81: ONE AND ONLY first-read novel-syntax stress scoring of
FROZEN S78 eight independent situation heads on REAL genuine Calibri43M GT
S80 first CPU H8, source SHA and constructed labels S79 preregistered.

The S79 48 SOURCE + separately authored labels were Git-blob and SHA
locked in #44507334853 BEFORE any original 43M / S78 novel input read.
S80 genuine original43M/GT original 616 baseline J7 replay then new
48 H8 source capture has no label/S78 imports. S78 physical original
trained 7496 FP16 model from run #44470550326 is immutable, SHA pinned.

Do not train, tune any threshold, retry a different model, repair data
based on scores, evaluate previous held/EVAL or activate any mechanism.
The one-time frozen S78 model outputs and source-only event ranks are
serialized BEFORE any S79 target labels are dynamically imported.
These authored synthetic labels are NOT independently human-adjudicated;
results test new *syntax patterns* but are NOT natural-language benchmark,
and teacher-anchored event F1 is NOT end-to-end factuality/recipient F1.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file,save_file

from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    S52QuoteAwareFrozenPaidGraphBridge,tensor_sha,
)
from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import (
    gold_blind_eventfirst_features,
)
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    semantic_features,
)
from research.nra3_s20.s60_native43m_local_triclass_model import (
    source_local_scope,
)
from research.nra3_s20.s73_fit_real43m_old344_microareas import (
    original_frozen_paid,
)
from research.nra3_s20.s77_real43m_gt_mentionhood_vs_asserted_candidate_audit import (
    mention_only_source_mask,
)
from research.nra3_s20.s78_train_true43m_quote_mention_8_specialists import (
    S78MentionSituationSpecialists,N_PAID,AREAS,
)
from research.nra3_s20.s79_NEW_NOVEL48_SOURCE_ONLY_locked_before_S78_firstread import (
    new_novel48_source_only,new_novel48_source_sha256,
)

STAGE="NRA2_S81_FIRST_READ_FROZEN_S78_TRUE43M_GT_NOVEL48_SYNTAX_SHIFT_STRESS"
S79_SOURCE_SHA="e554c004e40ba91bfa8cb2887a78ce7ec0fb6f32d03361e758edd1a9f0613bc0"
S78_PAID_SHA="7e642eb3e5a9eecef75c07d7ff16bc6065760f1d1fa93f0abaebe254a5ea5749"
S80_TENSORS="NRA2_S80_GENUINE43M_GT_NOVEL48_WORD339_NO_GOLD.safetensors"
S80_MANIFEST="NRA2_S80_GENUINE43M_GT_NOVEL48_GOLD_BLIND_SOURCE_PROVENANCE.json"
S78_TENSORS="NRA2_S78_TRUE43M_GT_MENTION8_SITUATION_SPECIALISTS_7496_FP16.safetensors"
BEFORE_GOLD="NRA2_S81_PRE_GOLD_FROZEN_S78_FIRST_READ_LOGITS_AND_SOURCE_RANKS.safetensors"
FINAL="NRA2_S81_FROZEN_S78_NOVEL48_FIRSTREAD_POSTHOC_REPORT.json"
N=48; K=8; PHYSICAL_S78_BATCH=120


def require(ok,why):
    if not bool(ok):raise ValueError("NRA2 S81 one-shot no refit: "+why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_SOURCE_AND_model_firstread(args):
    """SOURCE-ONLY original H8 + old physical source ranks, never gold."""
    source=new_novel48_source_only()
    require(len(source)==N and new_novel48_source_sha256()==S79_SOURCE_SHA
            and all(set(z)=={"row_id","text"} for z in source),
            "preregistered author source text source-only SHA locked")
    require(args.s80.name==S80_TENSORS
            and args.s80_report.name==S80_MANIFEST
            and args.s78.name==S78_TENSORS
            and sha(args.s78)==S78_PAID_SHA,
            "original S80 first physical H8 and pre-trained S78 SHA pinned")
    report=json.loads(args.s80_report.read_text(encoding="utf8"))
    require(report["stage"]=="NRA2_S80_TRUE43M_GT_FIRSTREAD_NOVEL48_GOLD_BLIND_H8_SOURCE"
            and report["S79_new_novel48_SOURCE_ONLY_SHA256"]==S79_SOURCE_SHA
            and report["S79_PRECOMMIT_SUCCESS_WORKFLOW_RUN_ID"]==44507334853
            and report["original_Calibri43M_protected_616J7_physically_replayed_BEFORE_firstread"]
            and report["S79_CONSTRUCTION_GOLD_NEVER_IMPORTED_BY_REAL_H8_CPU"]
            and report["S78_trained_weights_NEVER_LOADED_BY_SOURCE_H8_CAPTURE"]
            and report["genuine_original_12_layers_with_GT_S132_prefill_S93"]
            and report["genuine_source_tensor_file_SHA256"]==sha(args.s80)
            and report["n_novel_shift_input_rows"]==N
            and report["new_model_training_steps"]==0
            and report["output_gate"]==0 and not report["candidate_accepted"],
            "previous fully original 43M source only first 616 J7 + NOVEL48 physical trace")
    physical=load_file(str(args.s80),device="cpu")
    require(set(physical)=={
            "original_real43m_gt_word339_READONLY",
            "original_native_word_valid_mask",
            "native_BPE_causally_unsafe_word_mask",
            "source_row_indices_in_frozen_S79_NOVEL48",
            }, "source original first read has NO GOLD arrays")
    h=physical["original_real43m_gt_word339_READONLY"]
    valid=physical["original_native_word_valid_mask"]
    unsafe=physical["native_BPE_causally_unsafe_word_mask"]
    require(h.ndim==3 and h.shape[0]==N and h.shape[-1]==339
            and valid.shape==unsafe.shape==h.shape[:2]
            and h.dtype==torch.float32
            and bool(torch.isfinite(h).all())
            and not bool((unsafe&~valid).any())
            and physical["source_row_indices_in_frozen_S79_NOVEL48"].tolist()
                ==list(range(N)),
            "first frozen original real H8 BPE source physical")
    parents,sha_before=original_frozen_paid(args)
    old26,old57,old59,*_=parents
    texts=tuple(z["text"] for z in source)
    with torch.no_grad():
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(h,valid,unsafe)
        ev=gold_blind_eventfirst_features(h,obs,texts)
        sem=semantic_features(h,ev)
        fused=source_local_scope(sem,texts)
        old_logits=tuple(v.detach() for v in old57(ev))
        native_event_scores=old59(fused.base,old_logits)[0].detach()
    quote=obs.quote_word_mask
    asserted=ev.candidate_event.detach().clone()
    mention=mention_only_source_mask(
        valid,unsafe,obs.punctuation_mask,texts)
    require(native_event_scores.shape==valid.shape
            and mention.shape==asserted.shape==valid.shape
            and not bool((asserted&quote).any())
            and not bool((mention&unsafe).any())
            and not bool((mention&obs.punctuation_mask).any()),
            "unchanged old asserted quote-exclusion vs new isolated mention pool")
    def original_source_top8(mask):
        e=native_event_scores.masked_fill(~mask,float("-inf"))
        top=e.topk(K,dim=-1)
        return top.indices.detach().long(),torch.isfinite(top.values)
    mention_top,mention_valid=original_source_top8(mention)
    old_top,old_valid=original_source_top8(asserted)
    require(bool(mention_valid.any(-1).all())
            and bool(old_valid.any(-1).all())
            and mention_top.shape==old_top.shape==(N,K),
            "preregistered source ONLY original top8 ranks")
    saved=load_file(str(args.s78),device="cpu")
    prefix="NRA2_S78_NEW_MENTION_SPECIALISTS."
    require(all(k.startswith(prefix) for k in saved)
            and sum(z.numel() for z in saved.values())==N_PAID
            and all(z.dtype==torch.float16 for z in saved.values()),
            "exact true TRAIN90 S78 all eight trained original frozen FP16 weights")
    net=S78MentionSituationSpecialists().eval().requires_grad_(False)
    net.load_state_dict({
        k[len(prefix):]:v.float().clone() for k,v in saved.items()
    },strict=True)
    require(net.budget()["new_FP16_trainable_parameters"]==7496
            and net.output_gate==0 and not net.activation_allowed
            and not net.candidate_accepted,
            "frozen S78 8 specialist original training architecture unchanged")
    # S78 trained network hard-requires B=120 but all operations are
    # per-source (two linear projections, GELU, 8 independent heads)
    # with no cross-batch state. Deterministic padding/repeat to exactly
    # original 120 shape preserves model code/weights BIT-FOR-BIT.
    pad=torch.arange(PHYSICAL_S78_BATCH,dtype=torch.long)%N
    require(torch.equal(pad[:N],torch.arange(N))
            and PHYSICAL_S78_BATCH==120,"S78 original 120 batch semantics")
    with torch.inference_mode():
        expanded=net(h[pad],valid[pad],
                     mention_top[pad],mention_valid[pad])
        first=expanded[:N].detach().clone().float()
    require(first.shape==(N,K,len(AREAS))
            and bool(torch.isfinite(first).all()),
            "one and ONLY pretrained frozen original S78 forward complete")
    require(all(tensor_sha(*(z.detach() for z in m.parameters()))==h_before
                for m,h_before in zip(parents,sha_before)),
            "original physical Calibri-adjacent S26/S57/S59/S61/S62/S63/S65 remain immutable")
    require(sha(args.s78)==S78_PAID_SHA,
            "S78 real pretrained weights physically unchanged after firstread")
    # Persist actual autonomous source-only first predictions BEFORE
    # the gold module is imported anywhere (separate downstream step).
    args.out.mkdir(parents=True,exist_ok=False)
    pre=args.out/BEFORE_GOLD
    save_file({
        "original_S78_frozen_FIRSTREAD_factor8_logits":first.contiguous(),
        "original_S77_mention_top8_source_only":mention_top.int().contiguous(),
        "original_S77_mention_top8_allowed_source_only":mention_valid.contiguous(),
        "old_S65_ASSERTED_only_top8_source_only":old_top.int().contiguous(),
        "old_S65_ASSERTED_only_top8_allowed_source_only":old_valid.contiguous(),
        "original_S52_quote_word_mask_source_only":quote.contiguous(),
        "S77_quote_inclusive_mention_word_mask_source_only":mention.contiguous(),
        "original_word_valid_mask_source_only":valid.contiguous(),
    },str(pre))
    loaded=load_file(str(pre),device="cpu")
    require(torch.equal(loaded["original_S78_frozen_FIRSTREAD_factor8_logits"],first)
            and torch.equal(loaded["original_S77_mention_top8_source_only"],
                            mention_top.int()),
            "physical first-read logits serialized BEFORE opening target labels")
    result={
        "source_SHA256":S79_SOURCE_SHA,
        "original_true43m_GT_H8_source_SHA256":sha(args.s80),
        "original_S78_pretrained_7496FP16_SHA256":S78_PAID_SHA,
        "original_S78_frozen_firstread_logits_PHYSICALLY_SAVED_BEFORE_GOLD":True,
        "source_only_firstread_logits_SHA256":sha(pre),
        "no_optimizer_no_retraining_no_threshold_search":True,
        "no_constructed_targets_entered_any_original43M_or_S78_forward":True,
        "S77_source_only_quote_inclusive_MENTION_NOT_ASSERTION":True,
        "original_asserted_S65_quote_EXCLUSION_preserved":True,
        "n_mention_candidates_SOURCEONLY":int(mention.sum()),
        "n_asserted_candidates_SOURCEONLY":int(asserted.sum()),
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    return result


def _micro_and_exact(pred:torch.Tensor,target:torch.Tensor):
    require(pred.shape==target.shape and pred.ndim==2
            and pred.shape[-1]==len(AREAS),"S81 predeclared 8 factor comparison")
    p=pred.bool();y=target.bool()
    tp=int((p&y).sum());fp=int((p&~y).sum());fn=int((~p&y).sum())
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    return {
        "n_rows":len(pred),
        "micro_precision":precision,"micro_recall":recall,"micro_f1":f1,
        "exact_eight_factors_correct":int((p==y).all(-1).sum()),
        "tp":tp,"fp":fp,"fn":fn,
    }


def score_once_POSTHOC_only(out:Path,pre_source:dict):
    """Now, and ONLY now, import completely frozen separate constructed gold."""
    # GOLD imported only after genuine 48 H8 and S78 model FIRST forward
    # produced immutable source-only real prediction artifact.
    from research.nra3_s20.s79_NEW_NOVEL48_LABELS_POSTHOC_ONLY_never_in_forward import (
        frozen_novel48_labels_for_ONE_FINAL_READ_only,preregistered_manifest,
    )
    annotations=frozen_novel48_labels_for_ONE_FINAL_READ_only()
    prereg=preregistered_manifest()
    require(len(annotations)==N==48
            and prereg["raw_only_source_sha256"]==S79_SOURCE_SHA
            and prereg["one_time_score_no_new_S78_fitting_or_hparam_selection"]
            and not prereg["candidate_accepted"],
            "S79 48 constructed labels frozen before original H8 source read")
    t=load_file(str(out/BEFORE_GOLD),device="cpu")
    logits=t["original_S78_frozen_FIRSTREAD_factor8_logits"]
    mention=t["original_S77_mention_top8_source_only"].long()
    allowed=t["original_S77_mention_top8_allowed_source_only"]
    original=t["old_S65_ASSERTED_only_top8_source_only"].long()
    original_allowed=t["old_S65_ASSERTED_only_top8_allowed_source_only"]
    quoted=t["original_S52_quote_word_mask_source_only"]
    target=torch.tensor(
        [z["eight_area_target_POSTHOC"] for z in annotations],
        dtype=torch.bool)
    anch=torch.tensor(
        [z["constructed_anchor_INDEX_ONLY_FOR_POSTHOC"]
         for z in annotations],dtype=torch.long)
    require(logits.shape==(N,K,len(AREAS)) and
            target.shape==(N,len(AREAS)) and
            allowed.shape==original_allowed.shape==mention.shape==(N,K),
            "S81 no labels in original true43M S78 forward, posthoc 48 only")
    matches=(mention==anch[:,None]) & allowed
    has=matches.any(-1)
    idx=matches.long().argmax(-1)
    old_has=((original==anch[:,None])&original_allowed).any(-1)
    row_ids=torch.arange(N)
    # Covered cases: teacher-selected event slot ONLY for benchmark
    # diagnostic. The actual source-only MENTION top8 rankings and
    # all 8 logits were fully frozen BEFORE labels were read.
    pred_at_teacher=logits[row_ids,idx].sigmoid()>=0.5
    only_covered=_micro_and_exact(pred_at_teacher[has],target[has])
    # Whole-set valid coverage denominator: missing event proposals
    # cannot be counted as correct even if the eight factors are zero.
    with_coverage_correct=has & (pred_at_teacher==target).all(-1)
    top1=(mention[:,0]==anch)&allowed[:,0]
    joint_top1=top1 & (logits[:,0].sigmoid().ge(0.5)==target).all(-1)
    area={}
    for j,name in enumerate(AREAS):
        a=pred_at_teacher[has,j]
        y=target[has,j]
        tp=int((a&y).sum());fp=int((a&~y).sum());fn=int((~a&y).sum())
        precision=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else 0.
        area[name]={
            "constructed_support":int(y.sum()),
            "precision":precision,"recall":recall,
            "F1":2*precision*recall/(precision+recall) if precision+recall else 0.,
            "tp":tp,"fp":fp,"fn":fn,
        }
    classes={}
    for i in range(8):
        ix=torch.arange(i*6,(i+1)*6)
        actual=has[ix]
        classes[f"NOVEL_SYNTAX_GROUP_{i+1}"]={
            "n":6,
            "n_mentioned_event_in_sourceonly_top8":int(actual.sum()),
            "n_original_S65_asserted_beam8_anchor":int(old_has[ix].sum()),
            "n_exact_all_eight_covered":int(with_coverage_correct[ix].sum()),
            "n_mention_source_top1_matches_author_anchor":int(top1[ix].sum()),
        }
    result={
        "stage":STAGE,
        "PREREGISTRATION_S79_original_source_AND_separate_gold_locked_before_S78_read":True,
        "S79_precommit_run_44507334853_SUCCESS":True,
        "S80_first_real_original43M_GT_48_sourceonly_H8_after_616_J7":True,
        "sourceONLY_pred_logits_and_original_top8_SAVED_BEFORE_targets_opened":True,
        "same_unmodified_S78_7496_FP16_weights_sha256":S78_PAID_SHA,
        "original_43M_GT_S26_S57_S59_S61_S62_S63_S65_unchanged":True,
        "new_syntax_NOVEL48_not_seen_in_S78_TRAIN90_or_synthetic_diag30":True,
        "synthetic_author_targets_NOT_independently_human_adjudicated":True,
        "NO_S78_parameter_update_NO_threshold_change_NO_reselection":True,
        "no_new_candidate_acceptance_or_EVAL32_EVAL128_held32":True,
        "source_only_autoMENTION_top8_NOT_factual_assertion":True,
        "n_preregistered_novel_syntax_cases":N,
        "n_sourceonly_mention_gold_anchor_in_top8":int(has.sum()),
        "n_sourceonly_mention_gold_anchor_missing_from_top8":int((~has).sum()),
        "n_original_asserted_S65_anchor_in_top8":int(old_has.sum()),
        "n_original_asserted_S65_anchor_missing_from_top8":int((~old_has).sum()),
        "n_gold_anchor_inside_source_quote":int(
            quoted[row_ids,anch].sum()),
        "n_sourceonly_top1_mention_anchor_correct":int(top1.sum()),
        "n_sourceonly_top1_AND_8factors_correct":int(joint_top1.sum()),
        "n_mention_covered_AND_all8factors_exact":int(
            with_coverage_correct.sum()),
        "micro_and_exact_ONLY_teacher_anchored_on_sourceonly_top8_covered":only_covered,
        "per_miniarea_firstread_teacher_anchored":area,
        "per_novel_syntax_group_posthoc":classes,
        "S78_frozen_raw_unlabeled_firstread_predictions_SHA256":
            pre_source["source_only_firstread_logits_SHA256"],
        "original_source_BPE_SHA256":pre_source["original_true43m_GT_H8_source_SHA256"],
        "only_constructed_author_labels_NOT_independent_human_test":True,
        "no_autonomous_factual_transfer_or_recipient_quality_claim":True,
        "unanchored_mention_FALSE_POSITIVE_rate_NOT_measured":True,
        "original_616_J7_NEW_ACTIVE_candidate_NOT_tested":True,
        "all_in_bpp_le07_CPU_ge90pct_NOT_certified":True,
        "streaming_causal_safety_NOT_certified":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (out/FINAL).write_text(
        json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("NRA2_S81_FROZEN_S78_NEW_SYNTAX48_ONE_SHOT_FIRSTREAD",
          json.dumps({k:result[k] for k in (
              "n_preregistered_novel_syntax_cases",
              "n_sourceonly_mention_gold_anchor_in_top8",
              "n_original_asserted_S65_anchor_in_top8",
              "n_sourceonly_top1_mention_anchor_correct",
              "n_mention_covered_AND_all8factors_exact",
              "n_sourceonly_top1_AND_8factors_correct",
              "output_gate",
          )},sort_keys=True),flush=True)
    print("NRA2_S81_TEACHER_ANCHORED_NOVEL48_MICRO_F1",
          only_covered["micro_f1"],"FROM",int(has.sum()),"COVERED",
          "BUT_NEVER_END_TO_END_USER_ANSWER",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ("s80","s80_report","s78",
                 "s26","s57_paid","s59_paid","s61_paid","s62_paid",
                 "s63_paid","s65","s65_manifest","out"):
        parser.add_argument("--"+name.replace("_","-"),required=True,type=Path)
    args=parser.parse_args()
    require(not args.out.exists(),"one-time S81 no overwrite/repeat evaluations")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pre=immutable_SOURCE_AND_model_firstread(args)
    require(sha(args.out/BEFORE_GOLD)==pre["source_only_firstread_logits_SHA256"]
            and pre["no_constructed_targets_entered_any_original43M_or_S78_forward"]
            and pre["no_optimizer_no_retraining_no_threshold_search"],
            "S81 persisted source-only genuine model predictions before any target opens")
    result=score_once_POSTHOC_only(args.out,pre)
    require(result["output_gate"]==0
            and not result["candidate_accepted"]
            and not result["activation_allowed"],
            "S81 researcher must never enable learned candidates")


if __name__=="__main__":
    main()
