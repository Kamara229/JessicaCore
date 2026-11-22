"""S119q2i8q: actual original43M first-token EOS collapse learned-head causal decomposition.

Prior #951 genuine self-generation on 14 already spent synthetic DEV:
rank4 correction gain 0 never emits EOS, gain 0.25 starts right token
14/14, gain 1 produces EOS immediately 14/14. The #952 SHA-verified
physical output_bias EOS is only -0.0534, so global bias alone cannot
explain this. Probe one IDENTICAL original native 49,152 logits prefix
per 14 previously spent prompts for gain 0/0.25/1 with ZERO training.
By additive original lm_head hook, observed (logits_gain1 - logits_gain0)
is the actual full rank4 readout correction. Subtract physical learned
output_bias EOS-vs-gold difference to ATTRIBUTE remaining context-dependent
lowrank U gelu(D layernorm(hidden)) contribution exactly. No new model,
no DEV tuning, no sealed J7/EVAL128, no production, no acceptance.
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

STAGE="CALIBRI_S119Q2I8Q_REAL43M_EOS_GOLD_NATIVE_HEAD_BIAS_VS_LOWRANK_CONTEXT_CAUSAL"
REPORT="S119Q2I8Q_REAL43M_FIRST_BPE_EOS_HEAD_COMPONENT_DECOMPOSITION_G0.json"
POSITIONS=(1,2) # zero-based gold answer native BPE: second and third
NATIVE_VOCAB=49152
CE_LABEL="Q2I8K_CE_ONLY"
KL_LABEL="Q2I8K_SOURCE135M_EARLY23_KL"
ARMS=(CE_LABEL,KL_LABEL)
GAINS=(0.0,0.25,1.0)
LABELS=tuple(f"{arm}__gain_{gain:g}" for arm in ARMS for gain in GAINS)
GATES_KEY="research.s119q2i2.fullGT12_gates_fp16"
SEED=119156


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
    baseline=original_report["physical_OWN_sixdomain_new_and_old_DEV"][
        "SOURCE135M_SECONDTHIRD_BPE_KL"]["NEW_DEV"]
    base={r["source_id"]:r for r in baseline}
    check(len(base)==14,"14 exact earlier original KL own native samples")
    # Only the prior KL-trained child, with all original FP16 weights frozen.
    child=all_children[KL_LABEL]
    set_gt_route(full,"GT_RECURRENT",all_gates[KL_LABEL])
    trained_bias=child.language_head.output_bias.detach().float().cpu()
    check(trained_bias.shape==(NATIVE_VOCAB,)
          and len(trained_bias)==49152
          and child.language_head.down.weight.shape==(4,320)
          and child.language_head.up.weight.shape==(NATIVE_VOCAB,4)
          and bool(torch.isfinite(trained_bias).all()),
          "genuine previously saved full 49152 native rank4 FP16 head")
    samples=[];nforwards=0
    try:
        for ex in dev:
            check(ex["id"] in base
                  and ex["split"]=="DEV"
                  and len(ex["prefix"])<160,
                  "EXACT 14 already spent DEV prompt only, original native context")
            first_gold=int(ex["answer_native_with_eos"][0])
            check(first_gold!=tok.eos_token_id,
                  "never leak native EOS as first gold answer")
            logits={}
            predictions={}
            for gain in GAINS:
                x=gain_native_step(model,sources,bridge,full,expansion,
                                   child,ex["prefix"],ex["prompt_features"],gain)
                check(x.shape==(1,NATIVE_VOCAB),
                      "genuine full original 49152 logits at first answer token")
                logits[gain]=x[0]
                predictions[gain]=int(torch.argmax(x))
                nforwards+=1
            check(predictions[.25]==first_gold
                  and predictions[.25]==base[ex["id"]][
                      "generated_own_native_BPE_ids"][0]
                  and predictions[1.]==tok.eos_token_id
                  and predictions[0.]!=tok.eos_token_id,
                  "must exactly reproduce previous physical 14/14 0.25 correct, 1.0 EOS, 0 noEOS")
            a=logits[0.]
            b=logits[.25]
            c=logits[1.]
            linear_residual=float((b-(.75*a+.25*c)).abs().max().item())
            check(linear_residual<.002,
                  "actual native logits must depend linearly on gain, nothing else changed")
            eos=int(tok.eos_token_id)
            # Full learned head contextual U gelu(D layernorm(h)) + output_bias:
            # x(1)-x(0) = nativeHeadDelta exactly. We compare EOS v gold.
            delta=c-a
            delta_eos=float(delta[eos])
            delta_gold=float(delta[first_gold])
            head_delta_eos_vs_gold=delta_eos-delta_gold
            bias_eos_vs_gold=float(trained_bias[eos]-trained_bias[first_gold])
            contextual_rank4_eos_vs_gold=head_delta_eos_vs_gold-bias_eos_vs_gold
            original_gold_minus_eos=float(a[first_gold]-a[eos])
            current25_gold_minus_eos=float(b[first_gold]-b[eos])
            full1_gold_minus_eos=float(c[first_gold]-c[eos])
            check(current25_gold_minus_eos>0 and full1_gold_minus_eos<0
                  and head_delta_eos_vs_gold>0,
                  "x0.25 must favor first gold BPE over EOS while x1 must prefer EOS")
            # Exact crossover between .25 and 1, in unchanged true full native
            # logits at this particular previously scored source prompt.
            crossing=original_gold_minus_eos/head_delta_eos_vs_gold
            check(.25<crossing<1.,
                  "observed gold-vs-EOS crossing must match additive rank4 head")
            reconstruct=head_delta_eos_vs_gold-(
                bias_eos_vs_gold+contextual_rank4_eos_vs_gold)
            check(abs(reconstruct)<.0002,"additive true FP16 bias and contextual decomposition")
            samples.append({
                "source_id":ex["id"],"domain":ex["kind"],
                "prompt_SHA256":ex["prompt_sha256"],
                "source_split":"ALREADY_SPENT_INTERNAL_DEV",
                "original_native_BPE_gold_first":first_gold,
                "zero_head_gain_predicted_native_first_BPE":predictions[0.],
                "historical_point25_gain_predicted_native_first_BPE":predictions[.25],
                "head_gain1_native_predicted_first_EOS_ID":predictions[1.],
                "exact_readout_linear_interpolation_residual_max":round(linear_residual,8),
                "unchanged_original_gain0_gold_minus_EOS_logit":round(original_gold_minus_eos,7),
                "original_point25_gold_minus_EOS_logit":round(current25_gold_minus_eos,7),
                "original_gain1_gold_minus_EOS_logit":round(full1_gold_minus_eos,7),
                "learned_head_full_gain1_EOS_logit_delta":round(delta_eos,7),
                "learned_head_full_gain1_gold_logit_delta":round(delta_gold,7),
                "learned_head_total_EOS_advantage_over_gold":round(head_delta_eos_vs_gold,7),
                "learned_static_FP16_output_bias_EOS_advantage_over_gold":round(bias_eos_vs_gold,7),
                "learned_CONTEXTUAL_lowrank4_EOS_advantage_over_gold":round(contextual_rank4_eos_vs_gold,7),
                "gain_threshold_gold_to_EOS_crossing":round(crossing,7),
                "physical_true_full_native_original43M_forward_calls":3,
            })
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
              torch.equal(other.state_dict()[kk],value)
              for label,other in all_children.items()
              for kk,value in unchanged[label].items()),
          "full original 43M/GT12 and 421793 previous research FP16 weights strictly unchanged")
    check(nforwards==42 and len(samples)==14 and
          len({x["source_id"] for x in samples})==14,
          "exact genuine 42 full original 43M 49152 logits, no test set filtering")
    ctx=[x["learned_CONTEXTUAL_lowrank4_EOS_advantage_over_gold"] for x in samples]
    bias=[x["learned_static_FP16_output_bias_EOS_advantage_over_gold"] for x in samples]
    head=[x["learned_head_total_EOS_advantage_over_gold"] for x in samples]
    result={
        "stage":STAGE,
        "previous_SHA_verified_real_S119q2i8k_report":sha(q2i8k_dir/TRAIN_REPORT),
        "original_S142_43M_sha256":sha(s142),
        "original_S93_GT_sha256":sha(s93),
        "exact_old_S119q2i8k_421793_FP16_neural_and_12GT_only":True,
        "original_native_vocab":NATIVE_VOCAB,
        "existing_learned_lowrank_readout_rank":4,
        "physical_original43M_full_native_first_token_forwards":nforwards,
        "same_14_spent_internal_DEV_first_BPE_source":old["sha256"],
        "same_14_past_DEV_first_gold_at_gainpoint25":14,
        "same_14_immediate_native_EOS_first_at_gain1":14,
        "same_14_no_EOS_first_at_gain0":14,
        "learned_contextual_rank4_EOS_vs_gold_advantage_positive":sum(x>0 for x in ctx),
        "learned_bias_EOS_vs_gold_advantage_positive":sum(x>0 for x in bias),
        "mean_CONTEXTUAL_lowrank4_EOS_vs_gold_advantage":round(statistics.mean(ctx),7),
        "mean_static_output_bias_EOS_vs_gold_advantage":round(statistics.mean(bias),7),
        "mean_total_trained_head_EOS_vs_gold_advantage":round(statistics.mean(head),7),
        "median_contextual_lowrank4_EOS_vs_gold_advantage":round(statistics.median(ctx),7),
        "all_sample_gain0_0p25_1_eos_gold_causal_decomposition":samples,
        "NO_model_select_from_spent_synthetic_author_DEV":True,
        "no_optimizer_no_backward_no_new_training_no_model_tensors_written":True,
        "no_independent_blind_or_protected_616J7_bpp_CPU":True,
        "original_pretrained_and_research_safetensors_unchanged":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"CONTINUE_RESEARCH_ONLY_COMPONENT_CAUSAL_EOS_ATTRIBUTION_NO_PROMOTION",
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(
        json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2I8Q_TRUE43M_RANK4_EOS_FIRST_CAUSAL_DECOMP "+json.dumps({
        "43M_native_forwards":nforwards,
        "context_favors_EOS_count":result[
            "learned_contextual_rank4_EOS_vs_gold_advantage_positive"],
        "output_bias_favors_EOS_count":result[
            "learned_bias_EOS_vs_gold_advantage_positive"],
        "mean_context_EOS_vs_gold":result[
            "mean_CONTEXTUAL_lowrank4_EOS_vs_gold_advantage"],
        "mean_bias_EOS_vs_gold":result[
            "mean_static_output_bias_EOS_vs_gold_advantage"],
        "mean_entire_head_EOS_vs_gold":result[
            "mean_total_trained_head_EOS_vs_gold_advantage"],
        "gain_crossings":[x["gain_threshold_gold_to_EOS_crossing"] for x in samples],
        "candidate_accepted":False
    },sort_keys=True),flush=True)

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
