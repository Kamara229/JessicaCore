"""NRA2 S88: FIRST-READ-only S85 (8,264 FP16) on S86 NEW syntax48.

S85 trained on S83 56 authored synthetic TRAIN rows and S83 16
same-situation diagnostic; it has NEVER seen S86 text or constructed
S86 targets. The previous S79 NOVEL48 first-read was already spent
in S81 and is NEVER imported, retrained/tuned or rescored here.

S86 first-read source and separately authored 48 event+8 factor
labels were Git SHA locked at S86 #44536178198 BEFORE any S87 native
CPU source read or pretrained S85 inference. S87 physically replayed
original protected Calibri43M GT 616 J7 and then captured first 48
genuine BPE/H8 source representations, WITHOUT annotations or S85.

This module freezes original S26/S57/S59/S61/S62/S63/S65 physical
parents, recomputes separate source-only event MENTION top8 ranked
by OLD frozen S59 (not labels), loads EXACT SHA-pinned trained S85,
runs S82 eight-head model ONCE and physically saves raw logits/indices
BEFORE importing constructed S86 targets. No model optimizer, no
threshold search, no checkpoint selection, no baseline tuning.

These 48 author-constructed targets are NOT independent human gold,
not representative real user performance, and teacher-anchored event
F1 is NOT end-to-end detection, recipient/factuality or generated J7.
There is no online causal or all-in bpp/CPU proof. Gate always 0.
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
from research.nra3_s20.s82_predicate_centered_situation_fields import (
    S82PredicateCenteredMiniAreas,AREAS,
)
from research.nra3_s20.s86_newSYNTHETIC48_SOURCEONLY_never_train import (
    S86_novel48_source_only,S86_novel48_source_sha256,
)

STAGE="NRA2_S88_FROZEN_S85_NEW_S86_NOVEL48_REAL43M_GT_ONE_SHOT_FIRSTREAD"
SOURCE_SHA="60790a13127614af91cfd748bf503067fcaf0b37d2278ac3efef5e2942add867"
S85_PAID_SHA="45997dca99a91c820e285536d5b1ce90c3567db1a58e244c6d26072380e06eaf"
S87_PHYSICAL_SHA="7ae7954d229338a0c4bd14f700f217e5d9437fa1be3f4a4e1bbabca4240b1893"
S87_SOURCE_TENSOR="NRA2_S87_GENUINE43M_GT_NOVEL48_WORD339_NO_GOLD.safetensors"
S87_SOURCE_REPORT="NRA2_S87_GENUINE43M_GT_NOVEL48_GOLD_BLIND_SOURCE_PROVENANCE.json"
S85_CHECKPOINT="NRA2_S85_TRUE43M_GT_S82_EVENT_LOCAL8264_FP16.safetensors"
PRE="NRA2_S88_FROZEN_S85_NOVEL48_PHYSICAL_FIRST_LOGITS_BEFORE_GOLD.safetensors"
REPORT="NRA2_S88_ONE_SHOT_S86_NOVEL48_POSTHOC_FROZEN_S85_REPORT.json"
N=48
K=8
PAID=8264


def require(ok,why):
    if not bool(ok):
        raise ValueError("NRA2 S88 43M frozen S85 novel syntax firstread: "+why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def FIRST_source_only_frozen_S85_logits_BEFORE_any_target_read(args):
    """No S86 label module or correct event position can enter this call."""
    texts=S86_novel48_source_only()
    require(len(texts)==N and S86_novel48_source_sha256()==SOURCE_SHA
            and all(set(x)=={"row_id","text"} for x in texts),
            "previously preregistered never-trained S86 48 source SHA")
    require(args.s87.name==S87_SOURCE_TENSOR
            and args.s87_report.name==S87_SOURCE_REPORT
            and args.s85.name==S85_CHECKPOINT
            and sha(args.s87)==S87_PHYSICAL_SHA
            and sha(args.s85)==S85_PAID_SHA,
            "source genuine physical SHA and exact frozen previously-trained S85")
    r=json.loads(args.s87_report.read_text(encoding="utf8"))
    require(r["stage"]=="NRA2_S87_TRUE43M_GT_FIRSTREAD_NOVEL48_GOLD_BLIND_H8_SOURCE"
            and r["S86_NEW_NOVEL48_SOURCE_ONLY_SHA256"]==SOURCE_SHA
            and r["S86_PRECOMMIT_SUCCESS_WORKFLOW_RUN_ID"]==44536178198
            and r["original_Calibri43M_protected_616J7_physically_replayed_BEFORE_firstread"]
            and r["S86_CONSTRUCTION_GOLD_NEVER_IMPORTED_BY_REAL_H8_CPU"]
            and r["S85_frozen_trained_weights_NEVER_LOADED_BY_SOURCE_H8_CAPTURE"]
            and r["original_43M_real_frozen_BPE_IDs_AND_offset_mapping_unchanged"]
            and r["genuine_original_12_layers_with_GT_S132_prefill_S93"]
            and r["original_S26_S57_S59_S61_S62_S63_S65_and_GT_weights_unchanged"]
            and r["genuine_source_tensor_file_SHA256"]==S87_PHYSICAL_SHA
            and r["n_novel_syntax_firstread_input_rows"]==N
            and r["new_model_training_steps"]==0
            and r["output_gate"]==0 and not r["candidate_accepted"],
            "S87 genuine source firstread after protected original 616 J7, no labels")
    original=load_file(str(args.s87),device="cpu")
    require(set(original)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S86_NOVEL48",
    } and original["source_row_indices_in_frozen_S86_NOVEL48"].tolist()==list(range(N)),
            "no S86 constructed labels entered original BPE physical source")
    h=original["original_real43m_gt_word339_READONLY"]
    valid=original["original_native_word_valid_mask"]
    unsafe=original["native_BPE_causally_unsafe_word_mask"]
    require(h.ndim==3 and h.shape[0]==N and h.shape[-1]==339
            and h.dtype==torch.float32
            and valid.shape==unsafe.shape==h.shape[:2]
            and bool(torch.isfinite(h).all())
            and not bool((unsafe&~valid).any()),
            "original 320 H8+18 raw+quote1 native source features original BPE")
    parents,before=original_frozen_paid(args)
    old26,old57,old59,*_=parents
    plain=tuple(z["text"] for z in texts)
    with torch.inference_mode():
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(h,valid,unsafe)
        ev=gold_blind_eventfirst_features(h,obs,plain)
        sem=semantic_features(h,ev)
        fused=source_local_scope(sem,plain)
        oldlogits=tuple(v.detach() for v in old57(ev))
        event_scores=old59(fused.base,oldlogits)[0].detach()
    quote=obs.quote_word_mask
    asserted=ev.candidate_event.detach().clone()
    mentioned=mention_only_source_mask(valid,unsafe,obs.punctuation_mask,plain)
    require(event_scores.shape==valid.shape==mentioned.shape==asserted.shape
            and not bool((asserted&quote).any())
            and not bool((mentioned&unsafe).any())
            and not bool((mentioned&obs.punctuation_mask).any()),
            "original S65 asserted quote mask intact, broader nonasserting mention source")
    def top8(source_mask):
        masked=event_scores.masked_fill(~source_mask,float("-inf"))
        top=masked.topk(K,dim=-1)
        return top.indices.detach().long(),torch.isfinite(top.values)
    mention_idx,mention_allowed=top8(mentioned)
    old_idx,old_allowed=top8(asserted)
    require(mention_idx.shape==mention_allowed.shape==(N,K)
            and old_idx.shape==old_allowed.shape==(N,K)
            and bool(mention_allowed.any(-1).all()),
            "S88 source-only original S59 candidates independently of S86 GOLD")
    saved=load_file(str(args.s85),device="cpu")
    prefix="NRA2_S85_S82_EVENT_LOCAL."
    require(len(saved)>0 and all(k.startswith(prefix) for k in saved)
            and sum(t.numel() for t in saved.values())==PAID
            and all(t.dtype==torch.float16 for t in saved.values()),
            "exact S85 real43M GT 8264 trained FP16 weights from #44517161307")
    net=S82PredicateCenteredMiniAreas().eval().requires_grad_(False)
    net.load_state_dict({
        k[len(prefix):]:v.detach().float().clone()
        for k,v in saved.items()
    },strict=True)
    net.is_trained=True
    require(net.budget()["new_neural_parameters"]==PAID
            and net.output_gate==0 and not net.activation_allowed
            and not net.candidate_accepted,
            "original trained S82 architecture, ZERO output gate and no fine tuning")
    with torch.inference_mode():
        logits=net(h[:,:,:320].detach(),valid,unsafe,
                   mention_idx,mention_allowed,plain).detach().clone()
    require(logits.shape==(N,K,len(AREAS))
            and bool(torch.isfinite(logits).all()),
            "one and only exact frozen S85 first 48 novel structure factors")
    require(all(tensor_sha(*(p.detach() for p in m.parameters()))==check
                for m,check in zip(parents,before)),
            "original S26/57/59/61/62/63/65 frozen physical SHA unchanged")
    require(sha(args.s85)==S85_PAID_SHA,
            "pretrained 8264 FP16 physical checkpoint bit-identical after firstread")
    # Source-only first prediction serialized BEFORE importing S86 labels.
    args.out.mkdir(parents=True,exist_ok=False)
    pre=args.out/PRE
    save_file({
        "S85_firstread_frozen_eight_factor_logits_48x8x8":logits.contiguous(),
        "source_only_S59_MENTION_top8_word_ids":mention_idx.int().contiguous(),
        "source_only_S59_MENTION_top8_allowed":mention_allowed.contiguous(),
        "old_S65_ASSERTED_top8_word_ids":old_idx.int().contiguous(),
        "old_S65_ASSERTED_top8_allowed":old_allowed.contiguous(),
        "source_only_quote_word_mask":quote.contiguous(),
        "source_only_broad_MENTION_mask":mentioned.contiguous(),
        "source_only_original_BPE_word_valid":valid.contiguous(),
    },str(pre))
    check=load_file(str(pre),device="cpu")
    require(torch.equal(check["S85_firstread_frozen_eight_factor_logits_48x8x8"],logits)
            and torch.equal(check["source_only_S59_MENTION_top8_word_ids"],mention_idx.int()),
            "first-read autonomous logits physically fixed before constructed S86 labels")
    return {
        "source_SHA256":SOURCE_SHA,
        "original_true43m_GT_H8_source_SHA256":S87_PHYSICAL_SHA,
        "original_S85_pretrained_8264FP16_SHA256":S85_PAID_SHA,
        "model_sourceonly_logits_SAVED_BEFORE_author_gold_module_import":True,
        "model_sourceonly_firstread_logits_SHA256":sha(pre),
        "no_S86_labels_in_original43M_GT_S65_or_S82_forward":True,
        "no_optimizer_no_threshold_search_no_checkpoint_selection":True,
        "S77_broad_MENTION_not_factual_assertion":True,
        "old_S65_quote_exclusion_remains_intact":True,
        "n_sourceonly_MENTION_candidate_words":int(mentioned.sum()),
        "n_old_S65_asserted_candidate_words":int(asserted.sum()),
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }


def micro(pred:torch.Tensor,gold:torch.Tensor):
    require(pred.shape==gold.shape and pred.ndim==2
            and pred.shape[1]==len(AREAS),
            "S88 only 8 factor fixed threshold 0.5")
    a=pred.bool();b=gold.bool()
    tp=int((a&b).sum());fp=int((a&~b).sum());fn=int((~a&b).sum())
    p=tp/(tp+fp) if tp+fp else 0.
    r=tp/(tp+fn) if tp+fn else 0.
    f=2*p*r/(p+r) if p+r else 0.
    return {
        "n_covered_event_rows":len(a),
        "precision":p,"recall":r,"micro_F1":f,
        "all8_factor_exact":int(a.eq(b).all(-1).sum()),
        "tp":tp,"fp":fp,"fn":fn,
    }


def posthoc_ONE_SHOT_gold_after_logits_saved(out:Path,pre:dict):
    """No model/weights update; constructed S86 labels opened only NOW."""
    from research.nra3_s20.s86_NOVEL48_GOLD_SEPARATE_POSTHOC_ONLY import (
        S86_frozen_authored_gold_ONE_SHOT_POSTHOC_only,
        preregistered_S86_firstread_manifest,
    )
    gold=S86_frozen_authored_gold_ONE_SHOT_POSTHOC_only()
    m=preregistered_S86_firstread_manifest()
    require(len(gold)==N
            and m["new_source_only_SHA256"]==SOURCE_SHA
            and m["S85_source_only_logits_PHYSICALLY_SAVED_BEFORE_gold_open"]
            and m["no_checkpoint_selection_no_threshold_search_no_training_after_firstread"]
            and m["synthetic_author_labels_not_independent_human_truth"]
            and not m["candidate_accepted"],
            "original S86 author annotations frozen BEFORE any original pretrained read")
    immutable=load_file(str(out/PRE),device="cpu")
    logits=immutable["S85_firstread_frozen_eight_factor_logits_48x8x8"]
    top=immutable["source_only_S59_MENTION_top8_word_ids"].long()
    allowed=immutable["source_only_S59_MENTION_top8_allowed"]
    asserted=immutable["old_S65_ASSERTED_top8_word_ids"].long()
    asserted_allowed=immutable["old_S65_ASSERTED_top8_allowed"]
    quote=immutable["source_only_quote_word_mask"]
    anchor=torch.tensor(
        [x["original_event_word_index_POSTHOC_ONLY"] for x in gold],
        dtype=torch.long)
    target=torch.tensor(
        [x["event_situation_factor8_POSTHOC_ONLY"] for x in gold],
        dtype=torch.bool)
    require(logits.shape==(N,K,len(AREAS))
            and top.shape==allowed.shape==asserted.shape==asserted_allowed.shape==(N,K)
            and target.shape==(N,8)
            and all(not z["independently_human_adjudicated"] for z in gold),
            "immutable 48 source-only logits before 48 constructed event gold opened")
    found=(top==anchor[:,None])&allowed
    has=found.any(-1)
    slot=found.long().argmax(-1)
    old_has=((asserted==anchor[:,None])&asserted_allowed).any(-1)
    idx=torch.arange(N)
    teacher=logits[idx,slot].sigmoid()>=.5
    covered=micro(teacher[has],target[has])
    correct=has&teacher.eq(target).all(-1)
    top1=(top[:,0]==anchor)&allowed[:,0]
    autonomous_top1_AND_all8=top1 & ((logits[:,0].sigmoid()>=.5)==target).all(-1)
    per={}
    for k,name in enumerate(AREAS):
        p=teacher[has,k]
        g=target[has,k]
        tp=int((p&g).sum());fp=int((p&~g).sum());fn=int((~p&g).sum())
        precision=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else 0.
        per[name]={
            "support_constructed":int(g.sum()),
            "tp":tp,"fp":fp,"fn":fn,
            "precision":precision,"recall":recall,
            "F1":2*precision*recall/(precision+recall) if precision+recall else 0.,
        }
    groups={}
    for k in range(8):
        rows=torch.arange(k*6,(k+1)*6)
        groups[f"UNSEEN_SYNTAX_GROUP_{k+1}"]={
            "n_source_rows":6,
            "n_source_only_MENTION_top8_contains_constructed_event":int(has[rows].sum()),
            "n_old_asserted_S65_top8_contains_constructed_event":int(old_has[rows].sum()),
            "n_source_only_MENTION_top1_hits_constructed_event":int(top1[rows].sum()),
            "n_ALL8_event_factors_exact_among_all_group_rows":int(correct[rows].sum()),
        }
    out_report={
        "stage":STAGE,
        "S86_SOURCE_AND_SEPARATE_CONSTRUCTED_GOLD_PREREGISTERED_BEFORE_MODEL_READ":True,
        "S86_PRECOMMIT_RUN_ID":44536178198,
        "S87_original43M_GT_CPU_H8_after_protected_original616J7":True,
        "new_S86_48_lexical_syntax_text_NEVER_seen_during_S85_TRAIN56":True,
        "same_frozen_S85_FP16_SHA256":S85_PAID_SHA,
        "gold_opened_ONLY_after_physical_logits_source_only_serialized":True,
        "model_source_only_FIRSTREAD_logits_SHA256":
            pre["model_sourceonly_firstread_logits_SHA256"],
        "no_S85_retraining_or_threshold_change_or_model_reselection":True,
        "original_S26_S57_S59_S61_S62_S63_S65_and_Calibri43M_GT_unchanged":True,
        "synthetic_author_anchors_AND_factors_not_independently_human_verified":True,
        "original_asserted_S65_quote_exclusion_preserved":True,
        "n_precommitted_synthetic_novel48":N,
        "n_author_gold_event_in_source_only_mention_top8":int(has.sum()),
        "n_author_gold_event_NOT_in_source_only_mention_top8":int((~has).sum()),
        "n_author_gold_event_in_old_asserted_S65_top8":int(old_has.sum()),
        "n_author_gold_event_inside_literal_quote_mask":int(quote[idx,anchor].sum()),
        "n_source_only_mention_top1_gold_event":int(top1.sum()),
        "n_source_only_top1_AND_all8_exact":int(autonomous_top1_AND_all8.sum()),
        "n_teacher_anchored_top8_covered_AND_all8_exact":int(correct.sum()),
        "micro_and_exact_TEACHER_ANCHORED_ON_COVERED_only":covered,
        "per_area_teacher_anchored_firstread_synthetic":per,
        "per_syntax_group_firstread_synthetic":groups,
        "original_source_S87_native_H8_SHA256":
            pre["original_true43m_GT_H8_source_SHA256"],
        "pretrain_previous_S85_source_only_checkpoint_SHA256":
            pre["original_S85_pretrained_8264FP16_SHA256"],
        "not_autonomous_event_discovery_quality_or_recipient_or_factuality":True,
        "unanchored_mention_false_positive_precision_NOT_measured":True,
        "not_independent_natural_language_human_benchmark":True,
        "original616J7_new_CANDIDATE_not_run":True,
        "all_in_bpp_and_cpu_NOT_measured":True,
        "online_decoding_causal_safety_NOT_certified":True,
        "no_old_spent_S79_gold_retrained_or_rescored":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (out/REPORT).write_text(json.dumps(
        out_report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("NRA2_S88_FROZEN_S85_NEW_NOVEL48_SYNTAX_FIRSTREAD",
          json.dumps({key:out_report[key] for key in (
              "n_precommitted_synthetic_novel48",
              "n_author_gold_event_in_source_only_mention_top8",
              "n_author_gold_event_in_old_asserted_S65_top8",
              "n_source_only_mention_top1_gold_event",
              "n_source_only_top1_AND_all8_exact",
              "n_teacher_anchored_top8_covered_AND_all8_exact",
              "output_gate",
          )},sort_keys=True),flush=True)
    print("NRA2_S88_NEW_SYNTAX48_TEACHER_ANCHORED_MICRO_F1",
          covered["micro_F1"],"OVER",covered["n_covered_event_rows"],
          "COVERED_AUTHOR_EVENT_ANCHORS_NOT_END_TO_END",flush=True)
    return out_report


def main():
    parser=argparse.ArgumentParser()
    for n in ("s87","s87_report","s85","s26","s57_paid","s59_paid",
              "s61_paid","s62_paid","s63_paid","s65","s65_manifest","out"):
        parser.add_argument("--"+n.replace("_","-"),required=True,type=Path)
    args=parser.parse_args()
    require(not args.out.exists(),"never overwrite previous frozen firstread output")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source=FIRST_source_only_frozen_S85_logits_BEFORE_any_target_read(args)
    require(sha(args.out/PRE)==source["model_sourceonly_firstread_logits_SHA256"]
            and source["no_S86_labels_in_original43M_GT_S65_or_S82_forward"]
            and source["no_optimizer_no_threshold_search_no_checkpoint_selection"],
            "S88 first model predicted before any annotation import")
    result=posthoc_ONE_SHOT_gold_after_logits_saved(args.out,source)
    require(result["output_gate"]==0
            and not result["candidate_accepted"]
            and not result["activation_allowed"],
            "never accept S85 based on synthetic user-held firstread")


if __name__=="__main__":
    main()
