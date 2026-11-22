"""S119q2i4: physical original43M native partial-byte collapse causal audit.

Physical q2i3 first-token language-head 4-way ablation on SAME final FP16
learned Core/NRA3/NPA1+GT12+native 49152 head weights:
  FULL         frozen original 49k head + 0.25*(lowrank + learned bias)
  NO_BIAS      frozen original 49k head + 0.25*lowrank
  NO_LOWRANK   frozen original 49k head + 0.25*learned bias
  HEAD_OFF     frozen original 49k head + 0.0*(lowrank + learned bias)
Three exact same-source physical models, each on the 12 ALREADY SCORED
q2i1 DEV examples: Q2I1 original 0-gate baseline; trained q2i3
recurrent-GT12 full joint FP16; trained q2i3 local/no-carry-GT12 full
joint FP16. 12x3x4=144 ACTUAL original S142 12layer 43M+GT decoder native
LM-head forwards, no optimizer or changed parameter. Each FULL argmax
MUST reproduce corresponding q2i3 physically reported own-BPE ID0.
Exactly 0.25 head scale in ALL arms; only head bias/lowrank is altered.

The original learned language-head output has a 49,152-dimensional bias
and low-rank projection, and q2i3 own-BPE outputs are dominated by token
30 ('.') and 155 (replacement byte piece). Quantify which component
mathematically causes wrong first-token rank; do not infer whole word's
language or "semantic" advantage from decoded individual BPE pieces.

Original S142 Core12/GT/LM-head and 53751 old FP16 NRA3 unchanged.
All Q2I1 and Q2I3 learned checkpoint tensors SHA verified, never mutated,
all Green Tunnel original states restored after experimental forward.
Used-old DEV: NOT an independent blind benchmark. No S118/GOLD/EVAL128/J7
answers, no LIVE/Render/Supabase/main/Jessica chat/APK, no promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    CONTEXT_CKPT as Q2D_CTX, LOCAL_CKPT as Q2D_LOCAL,
    REPORT as Q2D_REPORT, VOCAB_SIZE,OLD_PREFIX,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    load_q2d_proof,
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    load_q2d_trained_child,
)
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    CONTEXT_CKPT as Q2F_CTX, REPORT as Q2F_REPORT,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import (
    load_physical_q2f_child,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as authored_rows,provenance as author_provenance,KINDS,
)
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    STAGE as Q2I1_STAGE,CONTEXT_CKPT as Q2I1_CTX,
    REPORT as Q2I1_REPORT,verified_native_answer,real_joint_native,
)
from research.calibri_nra3.s119q2i2_gt12_recurrence_connectivity import (
    load_q2i1_physical_context,native_full_gt_gate_contract,set_gt_route,
)
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    STAGE as Q2I3_STAGE, REPORT as Q2I3_REPORT,
    RECURRENT_CKPT, LOCAL_CKPT, RECURRENT_GATES, LOCAL_GATES,
    HEAD_GAIN,
)

STAGE="CALIBRI_S119Q2I4_ORIGINAL43M_GT12_NATIVE_BYTE_COLLAPSE_CAUSAL_HEAD_ATTRIBUTION"
REPORT="S119Q2I4_NATIVE49K_BYTE155_DOT30_READOUT_BIAS_LOWRANK_FORENSICS.json"
SEED=119143
MASKS=("FULL","NO_BIAS","NO_LOWRANK","HEAD_OFF")
VARIANTS=("GT_ZERO_BASELINE","GT_RECURRENT","GT_NOCARRY_LOCAL")
N_SOURCE=12
N_PHYSICAL_DECODERS=N_SOURCE*len(MASKS)*len(VARIANTS)
COLLAPSE_IDS=(30,155)
TOP_K=12
ADDITIVE_PARITY_TOL=0.00025


@contextmanager
def temporary_head_component_mask(head,mode):
    """Scoped counterfactual: readout components only, no token mask."""
    check(mode in MASKS and type(mode) is str
          and HEAD_GAIN==0.25
          and head.output_gate==0 and not head.candidate_accepted
          and not head.activation_allowed
          and not head._forward_hooks,
          "exact research-only fixed 0.25 neural 49152 head ablation")
    past=dict(head._forward_hooks)
    witness={"calls":0,"mode":mode}
    def mask(_module,_inputs,delta):
        check(delta.ndim==3 and delta.shape[0]==1
              and delta.shape[-1]==VOCAB_SIZE
              and bool(torch.isfinite(delta).all()),
              "original 49k learned native delta only")
        witness["calls"]+=1
        bias=head.output_bias.reshape(1,1,-1)
        if mode=="FULL":
            masked=delta
        elif mode=="NO_BIAS":
            masked=delta-bias
        elif mode=="NO_LOWRANK":
            masked=bias.expand_as(delta)
        else:
            masked=delta*0.0
        check(masked.shape==delta.shape
              and bool(torch.isfinite(masked).all()),
              "no proxy re-ranking, no hard-coded Russian token boosts")
        return HEAD_GAIN*masked
    h=head.register_forward_hook(mask)
    try:
        yield witness
    finally:
        h.remove()
        check(dict(head._forward_hooks)==past,
              "ALWAYS remove research head-only counterfactual hook")


def inspect_native(tok,logits,answer_first_id):
    check(logits.shape==(1,VOCAB_SIZE) and
          0<=answer_first_id<VOCAB_SIZE
          and bool(torch.isfinite(logits).all()),
          "ORIGINAL native full49152 physical logits with finite values")
    raw=logits.detach().cpu().float()[0]
    top=torch.topk(raw,k=TOP_K).indices.tolist()
    top1=int(top[0]); g=float(raw[answer_first_id])
    p=tok.decode([top1],skip_special_tokens=False)
    return {
        "native_top1_id":top1,
        "native_top1_piece":p,
        "native_top1_tokenizer_raw":str(tok.convert_ids_to_tokens(top1)),
        "native_top12_ids":top,
        "native_top12_pieces":[tok.decode([i],skip_special_tokens=False)
                               for i in top],
        "correct_short_answer_first_native_BPE_id":answer_first_id,
        "correct_first_native_rank_49152":
            int((raw>raw[answer_first_id]).sum())+1,
        "correct_first_native_in_top12":answer_first_id in top,
        "top1_minus_correct_first_logit_margin":
            float(raw[top1])-g,
        "collapsed_30_raw_native_logit":float(raw[30]),
        "collapsed_155_raw_native_logit":float(raw[155]),
        "correct_first_raw_native_logit":g,
        "native_selected_piece_may_be_partial_UTF8_byte_NOT_whole_word":True,
    }


def additive_readout_identity(full,nobias,norank,off):
    """Prove bias/rank4 causal additivity on SAME original physical model."""
    check(all(x.shape==(1,VOCAB_SIZE) and
              bool(torch.isfinite(x).all()) for x in
              (full,nobias,norank,off)),
          "same original full native 49152 vocab and model across masks")
    error=float(torch.max(torch.abs(
        (full+off)-(nobias+norank))).item())
    check(error<ADDITIVE_PARITY_TOL,
          "ACTUAL original native logit head decomposition must add up")
    return {
        "native_49k_additive_physical_parity_Linf":error,
        "native_original_full_vocab_exact_alignment":True,
        "native_head_off_removes_BOTH_only_READOUT_components":True,
    }


def check_trained_joint(file,metadata,baseline):
    check(file.name in (RECURRENT_CKPT,LOCAL_CKPT)
          and sha(file)==metadata["sha256"]
          and file.stat().st_size==metadata["file_bytes"],
          "exact physically completed q2i3 FP16 joint child input")
    stored=load_file(str(file),device="cpu")
    check(stored and sum(v.numel() for v in stored.values())==
          metadata["saved_total_child_FP16_params"]
          and all(k.startswith(OLD_PREFIX) and
                  v.dtype==torch.float16 for k,v in stored.items()),
          "all learned 421793 q2i3 FP16 tensors, not partial model")
    candidate=copy.deepcopy(baseline)
    full_state={k[len(OLD_PREFIX):]:v.float() for k,v in stored.items()}
    check(set(candidate.state_dict())==set(full_state),
          "complete q2i3 Core/NRA3/NPA1/rank8/readout neural child")
    candidate.load_state_dict(full_state,strict=True)
    check(candidate.output_gate==0
          and not candidate.candidate_accepted
          and not candidate.activation_allowed
          and candidate.child.output_gate==0,
          "neural q2i3 must stay research gate0/unaccepted")
    return candidate


def verified_gt12_gates(file,metadata):
    check(file.name in (RECURRENT_GATES,LOCAL_GATES)
          and sha(file)==metadata["sha256"]
          and file.stat().st_size==metadata["file_bytes"]
          and metadata["12_GT12_trainable_only"],
          "original q2i3 12gate trained GT12-only sidecar SHA")
    v=load_file(str(file),device="cpu")
    check(len(v)==1
          and set(v)=={"research.s119q2i2.fullGT12_gates_fp16"}
          and v["research.s119q2i2.fullGT12_gates_fp16"].numel()==12
          and v["research.s119q2i2.fullGT12_gates_fp16"].dtype==
              torch.float16,
          "physical only 12 actual per-layer original full GT gains")
    return v["research.s119q2i2.fullGT12_gates_fp16"].float()


def summarize_masks(rows):
    check(len(rows)==N_SOURCE and all(set(x["masks"])==set(MASKS)
                                  for x in rows),
          "12 previously scored source prompts each four native masks")
    out={}
    for mode in MASKS:
        m=[x["masks"][mode]["metrics"] for x in rows]
        out[mode]={
            "first_native_token_30_dot_greedy":
                sum(x["native_top1_id"]==30 for x in m),
            "first_native_token_155_partial_byte_greedy":
                sum(x["native_top1_id"]==155 for x in m),
            "mean_correct_first_native_gold_rank":
                sum(x["correct_first_native_rank_49152"] for x in m)/len(m),
            "correct_first_gold_in_top12":
                sum(x["correct_first_native_in_top12"] for x in m),
            "mean_wrong_top1_minus_gold_margin":
                sum(x["top1_minus_correct_first_logit_margin"]
                    for x in m)/len(m),
            "mean_raw_logits_for_collapsed_native_ids":{
                str(k):sum(x[f"collapsed_{k}_raw_native_logit"]
                           for x in m)/len(m)
                for k in COLLAPSE_IDS},
        }
    out["FULL_to_HEAD_OFF_first_native_greedy_changed"]=sum(
        r["masks"]["FULL"]["metrics"]["native_top1_id"]!=
        r["masks"]["HEAD_OFF"]["metrics"]["native_top1_id"]
        for r in rows)
    out["FULL_to_NO_BIAS_first_native_greedy_changed"]=sum(
        r["masks"]["FULL"]["metrics"]["native_top1_id"]!=
        r["masks"]["NO_BIAS"]["metrics"]["native_top1_id"]
        for r in rows)
    out["FULL_to_NO_LOWRANK_first_native_greedy_changed"]=sum(
        r["masks"]["FULL"]["metrics"]["native_top1_id"]!=
        r["masks"]["NO_LOWRANK"]["metrics"]["native_top1_id"]
        for r in rows)
    return out


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,q2i3_dir,out):
    check(not out.exists(),"new read-only diagnosis must never overwrite")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    author=author_provenance()
    old_dev=authored_rows("DEV")
    check(len(old_dev)==N_SOURCE and
          author["DEV_rows"]==N_SOURCE and
          Counter(x["kind"] for x in old_dev)=={k:2 for k in KINDS},
          "only previously SCORED 12 q2i1 DEV; NEVER reserved EVAL")
    q2d=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CTX,q2d_dir/Q2D_LOCAL,s93,s142)
    q2f=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    q2i1=json.loads((q2i1_dir/Q2I1_REPORT).read_text(encoding="utf8"))
    path=q2i3_dir/Q2I3_REPORT
    q2i3=json.loads(path.read_text(encoding="utf8"))
    check(q2i1["stage"]==Q2I1_STAGE
          and q2i1["original_author_new_Russian_source_provenance"]==author
          and q2i1["original_S142_43M_GT_SHA256"]==sha(s142)
          and q2i1["original_S93_SHA256"]==sha(s93)
          and q2i3["stage"]==Q2I3_STAGE
          and q2i3["prior_q2i1_real_joint_report_sha256"]==
              sha(q2i1_dir/Q2I1_REPORT)
          and q2i3["original_S142_43M_GT_sha256"]==sha(s142)
          and q2i3["original_S93_sha256"]==sha(s93)
          and q2i3["source_original_ru_TRAIN_sha256"]==author["TRAIN_sha256"]
          and q2i3["source_prior_scored_ru_DEV_sha256"]==author["DEV_sha256"]
          and q2i3["total_real_original_decoder_forwards"]==988
          and q2i3["same_FULL_neural_param_budget_A_B"]==368051
          and q2i3["identical_HEAD_GAIN_baseline_recurrent_nocarry"]
               ==HEAD_GAIN
          and q2i3["candidate_accepted"] is False
          and q2i3["output_gate"]==0,
          "exact physically completed 988-forward q2i3 parent and data")
    parent_q2d,_=load_q2d_trained_child(
        q2d_dir/Q2D_CTX,
        q2d["new_research_checkpoint_contextual"])
    parent_q2f=load_physical_q2f_child(
        q2f_dir/Q2F_CTX,
        q2f["projector_contextual_child_checkpoint"],parent_q2d)
    baseline=load_q2i1_physical_context(
        q2i1_dir/Q2I1_CTX,
        q2i1["learned_checkpoint_fp16"]["joint_contextual"],
        parent_q2f)
    candidates={
        "GT_ZERO_BASELINE":baseline,
        "GT_RECURRENT":check_trained_joint(
            q2i3_dir/RECURRENT_CKPT,
            q2i3["research_only_updated_FULL_neural_child_FP16"][
                "GT_RECURRENT"],baseline),
        "GT_NOCARRY_LOCAL":check_trained_joint(
            q2i3_dir/LOCAL_CKPT,
            q2i3["research_only_updated_FULL_neural_child_FP16"][
                "GT_NOCARRY_LOCAL"],baseline),
    }
    gates={
        "GT_RECURRENT":verified_gt12_gates(
            q2i3_dir/RECURRENT_GATES,
            q2i3["research_only_updated_GT12_gate_FP16_sidecar"][
                "GT_RECURRENT"]),
        "GT_NOCARRY_LOCAL":verified_gt12_gates(
            q2i3_dir/LOCAL_GATES,
            q2i3["research_only_updated_GT12_gate_FP16_sidecar"][
                "GT_NOCARRY_LOCAL"]),
    }
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE and tok.eos_token_id is not None,
          "true S142 original 49,152 native BPE real EOS only")
    dev=[verified_native_answer(tok,r) for r in old_dev]
    check(len(dev)==N_SOURCE and
          all(v["split"]=="DEV" for v in dev),
          "no TRAIN input; correct first BPE used only AFTER model native forward")
    model,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    model.model.add_module("nra3_s119m_onpath",baseline)
    model.eval()
    original_digest=old_original_param_digest(model)
    gt_old={k:v.detach().cpu().clone()
            for k,v in full.state_dict().items()}
    zero=full.gates.detach().clone()
    parent_state={
        name:{k:v.detach().cpu().clone()
              for k,v in mdl.state_dict().items()}
        for name,mdl in candidates.items()}
    gt_source_provenance=native_full_gt_gate_contract(full,bridge)
    check(all(not v.requires_grad for name,v in model.named_parameters()
              if "nra3_s119m_onpath" not in name),
          "original S142 43M, original S93/GT3/9/native head must remain frozen")
    observations={};forwards=0
    try:
        for variant,child in candidates.items():
            set_gt_route(full,variant,zero if variant=="GT_ZERO_BASELINE"
                         else gates[variant])
            cases=[]
            for e in dev:
                gold=int(e["answer_native_without_eos"][0])
                old_case=next(v for v in
                    q2i3["oldDEV_own_native_complete_text"][variant]
                    if v["source_id"]==e["id"])
                outputs={};records={}
                for mode in MASKS:
                    with temporary_head_component_mask(
                        child.language_head,mode) as count:
                        native=real_joint_native(
                            model,sources,bridge,full,expansion,child,
                            e["prefix"],e["prompt_features"],
                            contextual=True).detach().cpu().float()
                    check(count["calls"]==1 and
                          full.last_forward_coverage==tuple(range(12))
                          and not full._active and full._carry is None,
                          "one real original GT12 Native 49k forward per head mode")
                    metrics=inspect_native(tok,native,gold)
                    outputs[mode]=native
                    records[mode]={"metrics":metrics,
                        "physical_original_decoder_forwards":1,
                        "readout_mask_applied_inside_original_lm_head":True}
                    forwards+=1
                check(records["FULL"]["metrics"]["native_top1_id"]==
                      old_case["generated_own_native_BPE_ids"][0],
                      "FULL native first greedy MUST reproduce previous Q2I3 "
                      "own-token genuine decoder prediction, exact same input")
                identity=additive_readout_identity(
                    outputs["FULL"],outputs["NO_BIAS"],
                    outputs["NO_LOWRANK"],outputs["HEAD_OFF"])
                # Additive native logit effect directly measured on the
                # SAME physical source, without any re-training/tuning.
                effects={}
                for idx in (30,155,gold):
                    effects[str(idx)]={
                        "bias_native_delta_at_FIXED_GAIN":
                            float(outputs["FULL"][0,idx]-
                                  outputs["NO_BIAS"][0,idx]),
                        "lowrank_native_delta_at_FIXED_GAIN":
                            float(outputs["FULL"][0,idx]-
                                  outputs["NO_LOWRANK"][0,idx]),
                        "base_native_logit_WITHOUT_new_readout":
                            float(outputs["HEAD_OFF"][0,idx]),
                    }
                cases.append({
                    "source_id":e["id"],"kind":e["kind"],
                    "question_sha256":e["prompt_sha256"],
                    "correct_first_native_target_BPE_id":gold,
                    "previous_q2i3_physical_OWN_first_token":
                        old_case["generated_own_native_BPE_ids"][0],
                    "real_full_native_readout_four_mask_parity":identity,
                    "readout_attribution_native_logits":effects,
                    "masks":records,
                    "postforward_gold_never_used_as_native_input":True,
                })
                del outputs
            observations[variant]=cases
    finally:
        with torch.no_grad():
            full.gates.copy_(zero)
            full.mix_logits.zero_()
    check(forwards==N_PHYSICAL_DECODERS and
          all(torch.equal(v,full.state_dict()[k].detach().cpu())
              for k,v in gt_old.items())
          and old_original_param_digest(model)==original_digest
          and all(all(torch.equal(v,mdl.state_dict()[k].detach().cpu())
                      for k,v in parent_state[name].items())
                  for name,mdl in candidates.items()),
          "144 ORIGINAL physical forward heads, all S142 and all trained "
          "FP16 children immutable, S126 GT12 restored exactly")
    result={
        "stage":STAGE,
        "original_S93_SHA256":sha(s93),
        "original_S142_43M_GT_SHA256":sha(s142),
        "q2i1_physical_full_answer_report_SHA256":sha(q2i1_dir/Q2I1_REPORT),
        "q2i3_physical_988decoder_report_SHA256":sha(path),
        "prior_already_scored_OLD_DEV_SHA256":author["DEV_sha256"],
        "native_vocab":VOCAB_SIZE,
        "original_GT12_physical_state_provenance":gt_source_provenance,
        "four_registered_native_HEAD_readout_masks":list(MASKS),
        "three_registered_physical_original_core_GT12_variants":list(VARIANTS),
        "all_12_old_dev_six_domains_prior_spent_NOT_BLIND":True,
        "only_readout_additive_components_changed_per_same_neural_child":True,
        "first_greedy_native_BPE_matches_prior_q2i3_REAL_own_tokens":True,
        "number_real_original_Core12_GT_full_head_forward_calls":forwards,
        "native_ids_collapse_30_155_tokenizer_raw":{
            str(i):str(tok.convert_ids_to_tokens(i)) for i in COLLAPSE_IDS},
        "native_ids_collapse_30_155_decoded_SEPARATE_partial_piece":{
            str(i):tok.decode([i],skip_special_tokens=False)
            for i in COLLAPSE_IDS},
        "by_variant_source_repeated_HEAD_ON_OFF":observations,
        "summary":{
            variant:summarize_masks(cases)
            for variant,cases in observations.items()},
        "original_S142_43M_Core12_GT_and_original_native_head_UNCHANGED":True,
        "all_q2i1_q2i3_neural_child_FP16_UNCHANGED":True,
        "original_S115_old_53751_FP16_UNCHANGED":True,
        "no_training_no_optimizer_no_new_neural_weights":True,
        "no_new_BLIND_OR_eval_or_proven_generalization":True,
        "no_packed_all_in_bpp_KV_CPU_or_full_616_J7_EOS_replay":True,
        "no_GOLD_S118_B_old_EVAL128_J7_labels_or_Jessica_chat":True,
        "no_LIVE_Render_Supabase_main_APK":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"PHYSICAL_NATIVE_BYTE_COLLAPSE_CAUSAL_ATTRIBUTION_ONLY_NOT_ACCEPTED",
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(
        json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2I4_REAL43M_NATIVE_HEAD_COMPONENT_BYTE155_DOT30 "+
          json.dumps({
            "true_S142_43M_GT12_native_decoder_forwards":forwards,
            "original_native_30_155_pieces":
                result["native_ids_collapse_30_155_decoded_SEPARATE_partial_piece"],
            "diagnosis":result["summary"],
            "previous_own_native_full_predicted_token_EXACT":True,
            "original_weights_AND_q2i3_children_FROZEN":True,
            "candidate_accepted":False,
          },sort_keys=True,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir",
                 "q2i3_dir","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.q2i1_dir,
        a.q2i3_dir,a.out)


if __name__=="__main__":
    main()
