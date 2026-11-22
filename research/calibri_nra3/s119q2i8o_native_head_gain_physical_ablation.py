"""S119q2i8o: PHYSICAL read-only learned native-head gain ablation.

Identical SHA-verified completed Q2I8k fitted original43M/GT12/Core/NRA3/NPA1
49,152-way physical FP16 child and original donor weights; only scale
preexisting rank4 NativeHeadAlignment correction at gains [0, 0.25, 1].
Reference 0.25 must reproduce EXACT previous Q2I8k own-native DEV BPE.
Gold-previous prefix (only second/third BPE) is posthoc forensic only,
separate from honest OWN native 40-token greedy and strict EOS+UTF8 scores.
14 tiny ALREADY SPENT synthetic internal DEV; no independent blind claim,
no training/optimizer/backprop, no production, never select/publish model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import math
import statistics
from collections import Counter
from pathlib import Path
import torch

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import old_original_param_digest
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    CONTEXT_CKPT as Q2D_CTX,LOCAL_CKPT as Q2D_LOCAL,REPORT as Q2D_REPORT,OLD_PREFIX,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import load_q2d_proof
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import load_q2d_trained_child
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    CONTEXT_CKPT as Q2F_CTX,REPORT as Q2F_REPORT,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import load_physical_q2f_child
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    CONTEXT_CKPT as Q2I1_CTX,REPORT as Q2I1_REPORT,
)
from research.calibri_nra3.s119q2i2_gt12_recurrence_connectivity import (
    load_q2i1_physical_context,native_full_gt_gate_contract,set_gt_route,
)
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    REPORT as Q2I3_REPORT,RECURRENT_CKPT,
)
from research.calibri_nra3.s119q2i4_native_byte_head_component_forensics import check_trained_joint
from research.calibri_nra3.s119q2i7_real43m_utf8_weight_causal_ablation import REPORT as Q2I7_REPORT
from research.calibri_nra3.s119q2i8a_real43m_broader_russian_joint import (
    REPORT as Q2I8A_REPORT,RECURRENT_FP16 as Q2I8A_REC,
    RECURRENT_GT as Q2I8A_GT,verify_physical_1x_parent,
)
from research.calibri_nra3.s119q2i8b_source135m_vs_original43m_native import physical_q2i8a_fitted_child
from research.calibri_nra3.s119q2i8k_real43m_secondthird_joint_train_ab import (
    STAGE as TRAIN_STAGE,REPORT as TRAIN_REPORT,
    CE_FP16,KL_FP16,CE_GATES,KL_GATES,pack_new_author_dev,own_token_dev_long,
)
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import real_joint_native
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    HEAD_GAIN,scoped_reduced_native_head_gain,
)
from research.calibri_nra3.s119q2i5_real43m_own_vs_gold_utf8_sequence import (
    native_bytelevel_full_utf8_status,first_divergence,
)
from research.calibri_nra3.s119q2i8g_real43m_multiword_native_joint_ab import (
    STAGE as Q2I8G_PREV_STAGE,REPORT as Q2I8G_PREV_REPORT,
    CE_FP16 as Q2I8G_PREV_CE_FP16,CE_GATES as Q2I8G_PREV_CE_GATES,
    PREV_REPORT,physical_q2i8d_CE_warmstart,
)
from research.calibri_nra3.s119q2i8k_new_ru_multiword_dev_source import (
    STAGE as DEV_STAGE,rows as spent_dev,verify as dev_verify,
)
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import (
    verify as train_verify,
)
from safetensors.torch import load_file
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA

STAGE="CALIBRI_S119Q2I8O_PHYSICAL_FROZEN_NATIVEHEAD_GAIN_0_025_1_KL_VS_CE_FORENSICS"
REPORT="S119Q2I8O_TRUE_43M_NATIVE_HEAD_GAIN_0_025_1_G0.json"
POSITIONS=(1,2) # zero-based gold answer native BPE: second and third
NATIVE_VOCAB=49152
CE_LABEL="Q2I8K_CE_ONLY"
KL_LABEL="Q2I8K_SOURCE135M_EARLY23_KL"
ARMS=(CE_LABEL,KL_LABEL)
GAINS=(0.0,0.25,1.0)
LABELS=tuple(f"{arm}__gain_{gain:g}" for arm in ARMS for gain in GAINS)
GATES_KEY="research.s119q2i2.fullGT12_gates_fp16"
SEED=119155


def real_native_rank(logits,gold):
    """True 49152-way rank and two discriminative *logit* margins."""
    x=logits.detach().float().cpu().reshape(-1)
    check(x.shape==(NATIVE_VOCAB,)
          and bool(torch.isfinite(x).all())
          and type(gold) is int and 0<=gold<NATIVE_VOCAB,
          "full original 49152 native finite logits, no vocabulary masking")
    top_ids=torch.topk(x,5,largest=True,sorted=True).indices
    pred=int(top_ids[0].item())
    rank=1+int((x>x[gold]).sum().item())
    logp=float((x[gold]-torch.logsumexp(x,dim=0)).item())
    return {
        "correct_gold_native_rank_full49152":rank,
        "correct_gold_rank_top5":rank<=5,
        "gold_native_top1":pred==gold,
        "gold_log_probability_native_49152":round(logp,7),
        "wrong_top1_minus_gold_logit_gap":round(float(x[pred]-x[gold]),7),
        "greedy_original_native_id":pred,
        "gold_native_id":gold,
        "top5_native_ids":[int(i) for i in top_ids.tolist()],
    }


def full_saved_child(path,metadata,parent):
    check(path.exists() and sha(path)==metadata["sha256"]
          and path.stat().st_size==metadata["file_bytes"],
          "SHA-pinned original physical fitted FP16 child")
    data=load_file(str(path),device="cpu")
    check(data and sum(v.numel() for v in data.values())==
          metadata["saved_total_child_FP16_params"]==421793
          and all(k.startswith(OLD_PREFIX) and v.dtype==torch.float16
                  for k,v in data.items()),
          "all full 421793 child parameters, original 49k head included")
    actual={k[len(OLD_PREFIX):]:v.float()
            for k,v in data.items()}
    child=copy.deepcopy(parent)
    check(set(actual)==set(child.state_dict()),"exact matching physical architecture")
    child.load_state_dict(actual,strict=True)
    child.eval()
    check(child.output_gate==0 and child.candidate_accepted is False
          and child.activation_allowed is False,"research-only gate0")
    return child


def full_saved_gt(path,metadata):
    check(path.exists() and sha(path)==metadata["sha256"]
          and path.stat().st_size==metadata["file_bytes"]==128
          and metadata["12_GT12_trainable_only"]
          and metadata["condition"]=="GT_RECURRENT",
          "original physical GT12 gates metadata and SHA")
    data=load_file(str(path),device="cpu")
    check(set(data)=={GATES_KEY}
          and data[GATES_KEY].shape==(12,)
          and data[GATES_KEY].dtype==torch.float16,
          "original 12 layer physically trained scalar gates sidecar")
    return data[GATES_KEY].float()


@contextmanager
def scoped_exact_native_gain(head,gain):
    """One reversible downstream native rank4 readout factor ONLY."""
    check(type(gain) is float and gain in GAINS
          and HEAD_GAIN==0.25
          and head.output_gate==0
          and head.candidate_accepted is False
          and head.activation_allowed is False,
          "only preregistered research native head gain 0/0.25/1.0")
    before=set(head._forward_hooks)
    count=[0]
    def apply(_module,_args,delta):
        check(delta.ndim==3 and delta.shape[-1]==NATIVE_VOCAB
              and bool(torch.isfinite(delta).all()),
              "full original native 49152 learned rank4 head correction")
        count[0]+=1
        return delta*gain
    hook=head.register_forward_hook(apply)
    try:
        yield count
    finally:
        hook.remove()
        check(set(head._forward_hooks)==before,
              "native gain MUST be fully restored after every probe")


def gain_native_step(model,sources,bridge,full,expansion,child,prefix,features,gain):
    with scoped_exact_native_gain(child.language_head,gain) as count:
        logits=real_joint_native(model,sources,bridge,full,expansion,child,
                                  prefix,features,contextual=True)
    check(count[0]==1 and logits.shape==(1,NATIVE_VOCAB)
          and full.last_forward_coverage==tuple(range(12))
          and not full._active and full._carry is None,
          "ONE original physical 43M Core12+GT12+NRA3+NPA1/native forward")
    return logits.detach().float().cpu()


def score_by_position(records):
    out={}
    for position in POSITIONS:
        rows=[r for r in records if r["native_gold_answer_index_0based"]==position]
        check(len(rows)==14,"all 14 previously spent source-disjoint author DEV")
        x={}
        for label in LABELS:
            rs=[r["genuine_49152_head_gain_results"][label] for r in rows]
            ranks=[r["correct_gold_native_rank_full49152"] for r in rs]
            x[label]={
                "source_count":len(rs),
                "gold_native_top1":sum(r["gold_native_top1"] for r in rs),
                "gold_native_top5":sum(r["correct_gold_rank_top5"] for r in rs),
                "mean_gold_native_rank":round(statistics.mean(ranks),5),
                "median_gold_native_rank":statistics.median(ranks),
                "mean_greedy_minus_gold_logit_gap":round(statistics.mean(
                    r["wrong_top1_minus_gold_logit_gap"] for r in rs),6),
                "mean_gold_native_NLL":round(-statistics.mean(
                    r["gold_log_probability_native_49152"] for r in rs),6),
            }
        pairs={}
        for arm in ARMS:
            a=[r["genuine_49152_head_gain_results"][f"{arm}__gain_0.25"][
                "correct_gold_native_rank_full49152"] for r in rows]
            for gain in (0.0,1.0):
                b=[r["genuine_49152_head_gain_results"][f"{arm}__gain_{gain:g}"][
                    "correct_gold_native_rank_full49152"] for r in rows]
                pairs[f"{arm}_gain{gain:g}_vs_0.25"]={
                    "rank_better":sum(v<u for u,v in zip(a,b)),
                    "rank_worse":sum(v>u for u,v in zip(a,b)),
                    "rank_tied":sum(v==u for u,v in zip(a,b)),
                }
        x["descriptive_pair_rank_changes_NOT_MODEL_SELECTION"]=pairs
        out[f"gold_position_{position+1}"]=x
    return out


def own_native_score_for_gain(tok,rows):
    check(len(rows)==14,"all 14 spent DEV, no partial cherry pick")
    return {
        "source_count":len(rows),
        "full_answer_exact_native_EOS":sum(x["strict_finished_full_answer_exact"] for x in rows),
        "valid_complete_native_UTF8":sum(x["valid_complete_native_UTF8"] for x in rows),
        "native_EOS":sum(x["native_EOS"] for x in rows),
        "correct_first_gold_native_BPE":sum(x["correct_first_gold_BPE"] for x in rows),
        "real_full_original43M_decoder_forwards":sum(x["original_full_decoder_calls"] for x in rows),
    }



def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,q2i3_dir,
        q2i7_dir,q2i8a_dir,q2i8d_dir,q2i8g_dir,q2i8k_dir,out):
    check(not out.exists(),"research report never overwrite")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    old=dev_verify()
    train=train_verify()
    check(old["stage"]==DEV_STAGE
          and old["new_DEV"]==14
          and train["new_TRAIN_only"]==70,
          "14 previously scored author DEV only; no TRAIN/EVAL access")
    original_report=json.loads((q2i8k_dir/TRAIN_REPORT).read_text(encoding="utf8"))
    check(original_report["stage"]==TRAIN_STAGE
          and original_report["candidate_accepted"] is False
          and original_report["output_gate"]==0
          and original_report["registered_new_14_original_author_DEV_SHA"]==
              old["sha256"]
          and original_report["registered_70_new_multisentence_TRAIN_source_sha"]==
              train["TRAIN_sha256"]
          and original_report["original_S93_GT_sha256"]==sha(s93)
          and original_report["original_S142_43M_sha256"]==sha(s142)
          and original_report["all_trained_rows_are_SOURCE_UNUSED_NEXT21_NEW_TRAIN_plus_6_old_TRAIN"],
          "reused completed 21-new matched real 43M native A/B only")
    evidence=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CTX,q2d_dir/Q2D_LOCAL,s93,s142)
    q2f=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    q2i1=json.loads((q2i1_dir/Q2I1_REPORT).read_text(encoding="utf8"))
    q2i3=json.loads((q2i3_dir/Q2I3_REPORT).read_text(encoding="utf8"))
    q2i7=json.loads((q2i7_dir/Q2I7_REPORT).read_text(encoding="utf8"))
    q2i8a=json.loads((q2i8a_dir/Q2I8A_REPORT).read_text(encoding="utf8"))
    p7=q2i7_dir/Q2I7_REPORT
    p8=q2i8a_dir/Q2I8A_REPORT
    check(q2i8a["original_S93_sha256"]==sha(s93)
          and q2i8a["original_S142_real43M_GT_sha256"]==sha(s142)
          and q2i8a["prior_true_Q2I7_1X_control_physical_run_SHA256"]==sha(p7)
          and q2i8a["output_gate"]==0 and not q2i8a["candidate_accepted"],
          "original 43M provenance")
    d,_=load_q2d_trained_child(
        q2d_dir/Q2D_CTX,evidence["new_research_checkpoint_contextual"])
    f=load_physical_q2f_child(
        q2f_dir/Q2F_CTX,q2f["projector_contextual_child_checkpoint"],d)
    i1=load_q2i1_physical_context(
        q2i1_dir/Q2I1_CTX,
        q2i1["learned_checkpoint_fp16"]["joint_contextual"],f)
    i3=check_trained_joint(
        q2i3_dir/RECURRENT_CKPT,
        q2i3["research_only_updated_FULL_neural_child_FP16"][
            "GT_RECURRENT"],i1)
    i7=verify_physical_1x_parent(
        q2i7_dir/"S119Q2I7_UNIFORM1X_GT12_NRA3_NPA1_NATIVEHEAD_FP16_G0.safetensors",
        q2i7["new_1x_physical_FP16_checkpoint_and_GT12_sidecar"][
            "trained_1x_neural_fp16"],i3)
    i8=physical_q2i8a_fitted_child(
        q2i8a_dir/Q2I8A_REC,
        q2i8a["reversible_research_only_FP16_neural_A_B_and_GT12_gates"][
            "GT_RECURRENT_neural"],i7)
    prev=json.loads((q2i8d_dir/PREV_REPORT).read_text(encoding="utf8"))
    check(original_report["previous_physical_Q2I8d_parent_report_SHA256"]==
          sha(q2i8d_dir/PREV_REPORT)
          and original_report["previous_physical_Q2I8g_parent_report_SHA256"]==
          sha(q2i8g_dir/Q2I8G_PREV_REPORT)
          and prev["original_S142_43M_sha256"]==sha(s142),
          "exact completed parent SHA")
    parent_old,gates_old=physical_q2i8d_CE_warmstart(i8,q2i8d_dir,prev)
    g_report=json.loads((q2i8g_dir/Q2I8G_PREV_REPORT).read_text(encoding="utf8"))
    check(g_report["stage"]==Q2I8G_PREV_STAGE
          and g_report["candidate_accepted"] is False
          and g_report["output_gate"]==0
          and g_report["original_S93_GT_sha256"]==sha(s93)
          and g_report["original_S142_43M_sha256"]==sha(s142)
          and g_report["registered_70_new_multisentence_TRAIN_source_sha"]==
              train["TRAIN_sha256"],
          "exact immutable previous G CE trained parent")
    gmeta=g_report["saved_full_joints_and_native_gt_gate_sidecars_FP16_gate0"]
    parent=full_saved_child(q2i8g_dir/Q2I8G_PREV_CE_FP16,
        gmeta["CE_ONLY_neural"],parent_old)
    parent_gates=full_saved_gt(q2i8g_dir/Q2I8G_PREV_CE_GATES,
        gmeta["CE_ONLY_GT12"])
    check(gates_old.shape==parent_gates.shape==(12,),
          "original GT12 shape unchanged from old trained research state")
    meta=original_report["saved_full_joints_and_native_gt_gate_sidecars_FP16_gate0"]
    check(set(meta)=={"CE_ONLY_neural","SOURCE135M_KL_neural",
                      "CE_ONLY_GT12","SOURCE135M_KL_GT12"},
          "exact two matched fitted physical children and GT gates")
    ce=full_saved_child(q2i8k_dir/CE_FP16,meta["CE_ONLY_neural"],parent)
    kl=full_saved_child(q2i8k_dir/KL_FP16,meta["SOURCE135M_KL_neural"],parent)
    gt_ce=full_saved_gt(q2i8k_dir/CE_GATES,meta["CE_ONLY_GT12"])
    gt_kl=full_saved_gt(q2i8k_dir/KL_GATES,meta["SOURCE135M_KL_GT12"])
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==NATIVE_VOCAB and tok.eos_token_id==2,
          "exact same pinned 49152 original BPE and native EOS")
    dev=[pack_new_author_dev(tok,row) for row in spent_dev()]
    check(len(dev)==14 and
          len({x["prompt_sha256"] for x in dev})==14
          and all(x["split"]=="DEV" and
                  len(x["answer_native_with_eos"])>=4
                  for x in dev),
          "all 14 already spent author DEV two- and three-native targets")
    check(Counter(x["kind"] for x in dev)==
          {k:2 for k in {x["kind"] for x in dev}},
          "two original already spent internal DEV per category")
    model,sources,bridge,full,expansion=load_actual_original(s93,side,s142)
    model.model.add_module("nra3_s119m_onpath",ce)
    model.eval()
    frozen_orig=old_original_param_digest(model)
    frozen_full={k:v.detach().clone() for k,v in full.state_dict().items()}
    native_full_gt_gate_contract(full,bridge)
    all_children={CE_LABEL:ce,KL_LABEL:kl}
    all_gates={CE_LABEL:gt_ce,KL_LABEL:gt_kl}
    unchanged={
        name:{k:v.detach().clone() for k,v in child.state_dict().items()}
        for name,child in all_children.items()
    }
    check(parent.output_gate==0 and parent.candidate_accepted is False,
          "previous actual fitted Q2I8g parent remains unaccepted")
    rows=[];physical43m=0;own_calls=0;own={}
    baseline=original_report["physical_OWN_sixdomain_new_and_old_DEV"][
        "SOURCE135M_SECONDTHIRD_BPE_KL"]["NEW_DEV"]
    baseline_by_id={r["source_id"]:r for r in baseline}
    check(len(baseline_by_id)==14,"prior physical original 0.25 own DEV provenance")
    try:
        # Gold-prefix rank diagnostics are scored post hoc. No gold DEV enters
        # any optimizer, and separately all own sequences use prompt+OWN only.
        for ex in dev:
            gold=ex["answer_native_with_eos"]
            check(ex["id"] in baseline_by_id and ex["split"]=="DEV",
                  "exact previously spent physical source, no new DEV input")
            for index in POSITIONS:
                check(index<len(gold)-1 and len(ex["prefix"])+index<160,
                      "only gold previous BPE at 2nd/3rd, no EOS target")
                prefix=ex["prefix"]+gold[:index]
                check(prefix[:len(ex["prefix"])]==ex["prefix"],
                      "never future token in diagnostic prompt")
                target=int(gold[index])
                entry={"source_id":ex["id"],"domain":ex["kind"],
                       "source_prompt_SHA256":ex["prompt_sha256"],
                       "native_gold_answer_index_0based":index,
                       "gold_previous_BPE_teacher_forced_FORENSIC_ONLY":True,
                       "genuine_49152_head_gain_results":{}}
                for arm,child in all_children.items():
                    set_gt_route(full,"GT_RECURRENT",all_gates[arm])
                    for gain in GAINS:
                        logits=gain_native_step(
                            model,sources,bridge,full,expansion,
                            child,prefix,ex["prompt_features"],gain)
                        entry["genuine_49152_head_gain_results"][
                            f"{arm}__gain_{gain:g}"]=real_native_rank(logits,target)
                        physical43m+=1
                rows.append(entry)
        # One-arm (actual physically trained KL) full 14-own-token ablation.
        # Each run is an UNCHANGED original 43M autoregressive decoder,
        # not a teacher-forced gold-prefix score.
        for gain in GAINS:
            arm=KL_LABEL;child=all_children[arm]
            set_gt_route(full,"GT_RECURRENT",all_gates[arm])
            scored=[]
            for ex in dev:
                with scoped_exact_native_gain(child.language_head,gain) as calls:
                    got=own_token_dev_long(
                        model,sources,bridge,full,expansion,child,ex,
                        condition="GT_RECURRENT",max_generate=40)
                check(calls[0]==got["original_full_decoder_calls"],
                      "exact one physical 43M original forward per SELF native token")
                raw=got["generated_own_native_BPE_ids"]
                status=native_bytelevel_full_utf8_status(
                    tok,raw)["entire_original_native_BPE_prefix_UTF8_status"]
                item={
                    "source_id":ex["id"],"domain":ex["kind"],
                    "real_native_gain":gain,
                    "generated_own_native_BPE_ids":raw,
                    "first_differing_from_original_gold_BPE":
                        first_divergence(ex["answer_native_with_eos"],raw)[
                            "first_wrong_or_missing_native_BPE_position_1based"],
                    "strict_finished_full_answer_exact":
                        bool(got["strict_finished_full_answer_exact"]),
                    "correct_first_gold_BPE":
                        bool(raw and raw[0]==ex["answer_native_with_eos"][0]),
                    "valid_complete_native_UTF8":status=="VALID_COMPLETE_UTF8",
                    "native_EOS":bool(got["native_EOS"]),
                    "original_full_decoder_calls":int(got["original_full_decoder_calls"]),
                }
                if gain==0.25:
                    check(raw==baseline_by_id[ex["id"]]["generated_own_native_BPE_ids"]
                          and got["native_EOS"]==baseline_by_id[ex["id"]]["native_EOS"],
                          "new reversible 0.25 hook MUST match physical previous OWN BPE")
                own_calls+=len(raw)
                scored.append(item)
            own[f"gain_{gain:g}"]={
                "strict_own_score":own_native_score_for_gain(tok,scored),
                "all_14_own_native_generated_cases":scored,
            }
    finally:
        with torch.no_grad():
            for k,v in frozen_full.items():
                full.state_dict()[k].copy_(v)
        for p in full.parameters():
            p.requires_grad_(False)
    check(old_original_param_digest(model)==frozen_orig
          and all(torch.equal(full.state_dict()[k],v)
                  for k,v in frozen_full.items())
          and all(
              torch.equal(child.state_dict()[kk],value)
              for label,child in all_children.items()
              for kk,value in unchanged[label].items()),
          "original 43M + GT12 and all physical children FROZEN no optimizer")
    check(physical43m==14*len(POSITIONS)*len(all_children)*len(GAINS)==168
          and 14<=own_calls<=14*40*len(GAINS),
          "exact 168 full original43M 49k rank forwards + real own-native generation")
    agg=score_by_position(rows)
    baselineOwn=own["gain_0.25"]["strict_own_score"]
    check(baselineOwn["full_answer_exact_native_EOS"]==0
          and baselineOwn["valid_complete_native_UTF8"]==6
          and baselineOwn["native_EOS"]==11
          and baselineOwn["correct_first_gold_native_BPE"]==14,
          "exact S119q2i8k true physical baseline reproduced")
    ref={r["source_id"]:r for r in own["gain_0.25"]["all_14_own_native_generated_cases"]}
    div={}
    for gain in (0.0,1.0):
        target=own[f"gain_{gain:g}"]["all_14_own_native_generated_cases"]
        ch=[r for r in target if
            r["generated_own_native_BPE_ids"]!=ref[r["source_id"]][
                "generated_own_native_BPE_ids"]]
        div[f"gain_{gain:g}_vs_0.25"]={
            "different_own_native_49152_BPE_sequences":len(ch),
            "identical_own_native_sequences":14-len(ch),
            "changed_source_ids":[x["source_id"] for x in ch],
            "original_14_spent_synthetic_DEV_only_NOT_NEW_BENCHMARK":True,
        }
    result={
        "stage":STAGE,
        "reused_Q2I8k_completed_real_training_report_SHA256":
            sha(q2i8k_dir/TRAIN_REPORT),
        "original_S142_43M_sha256":sha(s142),
        "original_S93_GT_sha256":sha(s93),
        "native_vocab":NATIVE_VOCAB,
        "physical_existing_native_lowrank":4,
        "original_frozen_head_gain_before_ablation":HEAD_GAIN,
        "readout_gains_predeclared":[0.,0.25,1.],
        "two_physical_43M_fitted_FP16_children":["CE_ONLY","SOURCE135M_EARLY23_KL"],
        "original43M_full_native_rank_forwards":physical43m,
        "original43M_full_native_self_greedy_forwards":own_calls,
        "gold_previous_BPE_rank_diagnostics_NOT_OWN":agg,
        "new_DEV_14_spent_internal_explanatory_per_sample_rank":rows,
        "fresh_OWN_native_KL_arm_no_gold_prefix_by_gain":own,
        "paired_OWN_sequence_differences":div,
        "original0p25_OWN_native_baseline_EXACT_prev_S119q2i8k":True,
        "all_original_43M_GT12_421793_neural_FP16_unchanged":True,
        "no_backward_no_optimizer_no_weights_saved":True,
        "no_independent_blind_eval_or_protected_616J7_or_bpp_CPU":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"NO_MODEL_PROMOTION_HEAD_GAIN_POSTHOC_VISIBLE_DEV_FORENSIC_ONLY",
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(
        json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2I8O_PHYSICAL_NATIVE_HEAD_GAIN_ABLATION "+json.dumps({
        "second_gold_BPE":agg["gold_position_2"],
        "third_gold_BPE":agg["gold_position_3"],
        "own_14_gain_scores":{k:v["strict_own_score"] for k,v in own.items()},
        "paired_own_sequences":div,
        "original43M_full_native_gold_prefix_calls":physical43m,
        "original43M_full_native_OWN_calls":own_calls,
        "gate0":True,
    },ensure_ascii=False,sort_keys=True),flush=True)

def main():
    parser=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir",
                "q2i3_dir","q2i7_dir","q2i8a_dir","q2i8d_dir",
                "q2i8g_dir","q2i8k_dir","out"):
        parser.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=parser.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.q2i1_dir,
        a.q2i3_dir,a.q2i7_dir,a.q2i8a_dir,a.q2i8d_dir,a.q2i8g_dir,a.q2i8k_dir,a.out)

if __name__=="__main__":
    main()
