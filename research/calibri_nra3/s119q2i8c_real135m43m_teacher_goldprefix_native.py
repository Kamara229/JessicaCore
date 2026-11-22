"""S119q2i8b real original S142 43M vs frozen original source 135M teacher.

A decisive, SMALL read-only physical causal diagnostic after two full
Q2I8a joint-training variants still achieved 0/14 independent-of-training
(but previously inspected) internal Russian DEV correct full answers.

Compare SAME pinned SmolLM2-135M-Instruct SOURCE weights and tokenizer
to frozen 43M 12layer Core12+GreenTunnel reconstructed from S93/S142,
and q2i8a physically trained recurrent GT12 full neural child.
Use EXACT same original native chat tokens, 49152 native head, greedy
own-prefix autoregressive BPE, native EOS and cap 18 tokens.
No training or token masks, no prompt rewriting, no forced UTF-8 bytes.
Gold answers ONLY for post-forward scoring, never prefixes. This
isolates whether Russian output collapse existed in the SOURCE 135M
or was introduced by 43M compression/readout/extra neural areas.

7 Q2I8a source-only already-scored DEV prompts (one per seven kinds).
Not independent blind validation. Original SmolLM2-135M-Instruct is an
existing external pinned pretrained model and is not deployed/retrained;
do not claim it is authoritative Russian ground truth or guaranteed
to be fluent. If both fail, cannot attribute failure to compression alone.

No main/LIVE/Render/Supabase/Jessica chat/APK/secrets.
No protected original 616 J7+EOS, no full packed bpp or KV CPU claimed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import torch

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import (
    first_original_pass,
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    CONTEXT_CKPT as Q2D_CTX,LOCAL_CKPT as Q2D_LOCAL,
    REPORT as Q2D_REPORT,VOCAB_SIZE,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    load_q2d_proof,
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    load_q2d_trained_child,
)
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    CONTEXT_CKPT as Q2F_CTX,REPORT as Q2F_REPORT,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import (
    load_physical_q2f_child,
)
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    STAGE as Q2I1_STAGE,CONTEXT_CKPT as Q2I1_CTX,
    REPORT as Q2I1_REPORT,real_joint_native,
)
from research.calibri_nra3.s119q2i2_gt12_recurrence_connectivity import (
    load_q2i1_physical_context,native_full_gt_gate_contract,set_gt_route,
)
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    STAGE as Q2I3_STAGE,REPORT as Q2I3_REPORT,
    RECURRENT_CKPT,scoped_reduced_native_head_gain,
)
from research.calibri_nra3.s119q2i4_native_byte_head_component_forensics import (
    check_trained_joint,
)
from research.calibri_nra3.s119q2i5_real43m_own_vs_gold_utf8_sequence import (
    native_bytelevel_full_utf8_status,first_divergence,
)
from research.calibri_nra3.s119q2i7_real43m_utf8_weight_causal_ablation import (
    STAGE as Q2I7_STAGE,REPORT as Q2I7_REPORT,
)
from research.calibri_nra3.s119q2i8a_broader_russian_synthetic_source import (
    STAGE as SOURCE_STAGE,KINDS,rows as source_rows,
    verify as source_verify,
)
from research.calibri_nra3.s119q2i8a_real43m_broader_russian_joint import (
    STAGE as Q2I8A_STAGE,REPORT as Q2I8A_REPORT,
    RECURRENT_FP16 as Q2I8A_REC,RECURRENT_GT as Q2I8A_GT,
    verify_physical_1x_parent,verify_physical_1x_gt,
    prepare_native_original,
)

STAGE="CALIBRI_S119Q2I8C_REAL135M_TEACHER_VS_43M_CORRECT_NATIVE_GOLDPREFIX_UTF8"
REPORT="S119Q2I8C_REAL135M_VS_43M_NATIVE_SOFTLOGITS_GOLDPREFIX_UTF8_DIAGNOSTIC.json"
LABELS=("FROZEN_SOURCE_135M_INSTRUCT","FROZEN_S142_43M_ORIGINAL",
        "Q2I8A_FITTED_43M_GT12_NRA3_NPA1")
KIND_COUNT=7
MAX_TOKENS=18
NATIVE_VOCAB=49152
SEED=119148


def selected_already_scored_dev():
    # Select one of the TWO pre-registered Q2I8a internal DEV per kind
    # without inspecting any model output. No new benchmark claim.
    data=[row for row in source_rows("DEV")
          if int(row["id"].rsplit("_",1)[-1])==10]
    check(len(data)==KIND_COUNT and
          Counter(row["kind"] for row in data)=={k:1 for k in KINDS}
          and all(row["split"]=="DEV" for row in data),
          "same previously scored source 1x7 domains; zero optimizer gradients")
    return data


def physical_q2i8a_fitted_child(path,metadata,parent):
    check(path.name==Q2I8A_REC and sha(path)==metadata["sha256"]
          and path.stat().st_size==metadata["file_bytes"],
          "exact real physical Q2I8a recurrent native research FP16 child SHA")
    from safetensors.torch import load_file
    from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
        OLD_PREFIX,
    )
    stored=load_file(str(path),device="cpu")
    check(stored and
          sum(v.numel() for v in stored.values())==
              metadata["saved_total_child_FP16_params"]==421793
          and all(k.startswith(OLD_PREFIX) and
                  v.dtype==torch.float16 for k,v in stored.items()),
          "complete previous actual trained 421793 new neural FP16 child")
    child=copy.deepcopy(parent)
    original={k[len(OLD_PREFIX):]:v.float()
              for k,v in stored.items()}
    check(set(original)==set(child.state_dict()),
          "strict Q2I8a fitted Core NRA3 NPA1 and native head key parity")
    child.load_state_dict(original,strict=True)
    child.eval()
    check(child.output_gate==0 and not child.candidate_accepted
          and not child.activation_allowed
          and child.child.output_gate==0,
          "research sidecar not allowed in LIVE or main")
    return child


def physical_q2i8a_gates(path,metadata):
    check(path.name==Q2I8A_GT
          and sha(path)==metadata["sha256"]
          and path.stat().st_size==metadata["file_bytes"]==128
          and metadata["12_GT12_trainable_only"]
          and metadata["condition"]=="GT_RECURRENT",
          "exact Q2I8a original GT12 trained 12gate sidecar")
    from safetensors.torch import load_file
    data=load_file(str(path),device="cpu")
    check(set(data)=={"research.s119q2i2.fullGT12_gates_fp16"}
          and data["research.s119q2i2.fullGT12_gates_fp16"].shape==(12,)
          and data["research.s119q2i2.fullGT12_gates_fp16"].dtype==
              torch.float16,
          "physical only 12 S126 full-layer original trained GT gates")
    return data["research.s119q2i2.fullGT12_gates_fp16"].float()


def native_rank(logits,gold_id):
    x=logits.detach().float().cpu().reshape(-1)
    check(x.shape==(NATIVE_VOCAB,)
          and 0<=gold_id<NATIVE_VOCAB
          and bool(torch.isfinite(x).all()),
          "original 49,152 FULL native head, no restricted token proxy")
    pred=int(torch.argmax(x))
    return {
        "original_full_native_gold_first_rank":int((x>x[gold_id]).sum())+1,
        "original_full_native_greedy_first_id":pred,
        "original_full_native_correct_first":pred==gold_id,
        "native_gold_margin_to_greedy":float(x[pred]-x[gold_id]),
        "top8_native_ids":[int(z) for z in x.topk(8).indices.tolist()],
    }


def source_teacher_step(teacher,prefix):
    check(2<=len(prefix)<160
          and all(type(x) is int and 0<=x<NATIVE_VOCAB for x in prefix),
          "teacher source full native SAME chat prefix only")
    ids=torch.tensor([prefix],dtype=torch.long)
    with torch.inference_mode():
        out=teacher(input_ids=ids,past_key_values=None,use_cache=False)
        logits=out.logits[:,-1,:].detach().float().cpu()
    check(logits.shape==(1,NATIVE_VOCAB)
          and bool(torch.isfinite(logits).all()),
          "true pinned original 135M source same 49k tokenizer LM head")
    return logits


def physical_original_step(model,sources,bridge,full,expansion,prefix,feat):
    logits,_all_layer=first_original_pass(
        model,sources,bridge,full,expansion,prefix,feat)
    check(logits.shape==(1,NATIVE_VOCAB)
          and full.last_forward_coverage==tuple(range(12)),
          "real original S142 43M 12layer GreenTunnel and native LM head")
    return logits


def physical_trained_step(model,sources,bridge,full,expansion,
                          child,prefix,features):
    with scoped_reduced_native_head_gain(child.language_head) as witness:
        logits=real_joint_native(
            model,sources,bridge,full,expansion,child,
            prefix,features,contextual=True)
    check(witness[0]==1 and logits.shape==(1,NATIVE_VOCAB)
          and full.last_forward_coverage==tuple(range(12))
          and not full._active and full._carry is None,
          "same TRUE S142 43M 12layer + NEW joint neural 49k head")
    return logits.detach().float().cpu()


def greedy_own_stream(tok,example,step,*,label):
    """Gold is NEVER provided to step; only original source prompt + OWN BPE."""
    assert label in LABELS
    prefix=list(example["prefix"])
    check(2<=len(prefix)<160
          and example["split"]=="DEV"
          and example["answer_native_with_eos"][-1]==tok.eos_token_id
          and len(example["answer_native_with_eos"])<=MAX_TOKENS,
          "complete original native source + EOS fits actual generation cap")
    generated=[];first_metric=None;calls=0
    for token_index in range(MAX_TOKENS):
        check(prefix==example["prefix"]+generated,
              "strict actual autoregressive generated OWN native prefix")
        logits=step(prefix)
        check(logits.shape==(1,NATIVE_VOCAB),
              "real full 49152 native source/student decoder step")
        if token_index==0:
            first_metric=native_rank(
                logits,int(example["answer_native_with_eos"][0]))
        next_id=int(torch.argmax(logits,dim=-1).item())
        check(0<=next_id<NATIVE_VOCAB,
              "original full next native BPE must be in model vocabulary")
        generated.append(next_id)
        prefix.append(next_id)
        calls+=1
        if next_id==tok.eos_token_id:
            break
    assert first_metric is not None and calls==len(generated)
    correct=list(example["answer_native_with_eos"])
    div=first_divergence(correct,generated)
    decoded=tok.decode(generated,skip_special_tokens=True)
    expected=example["gold_answer_for_scoring_ONLY"]
    utf8=native_bytelevel_full_utf8_status(tok,generated)
    state=utf8["entire_original_native_BPE_prefix_UTF8_status"]
    # Count real EOS, actual full BPE answer and Unicode validity separately.
    return {
        "source_id":example["id"],"kind":example["kind"],
        "model":label,
        "real_original_native_49152_head_forward_calls":calls,
        "native_generated_BPE_ids":generated,
        "native_generated_text":decoded,
        "native_full_gold_BPE_target_including_EOS":correct,
        "native_EOS_emitted":bool(generated[-1]==tok.eos_token_id),
        "strict_entire_native_BPE_answer_including_EOS_correct":
            generated==correct,
        "native_visible_text_equals_gold":decoded==expected,
        "entire_generated_prefix_actual_GPT2_UTF8_state":state,
        "native_EOS_after_INVALID_or_INCOMPLETE_UTF8":
            generated[-1]==tok.eos_token_id and state!="VALID_COMPLETE_UTF8",
        "source_first_native_49152_logit_audit":first_metric,
        "true_first_native_BPE_correct":div["first_native_BPE_correct"],
        "first_wrong_or_missing_BPE_index_1based":div[
            "first_wrong_or_missing_native_BPE_position_1based"],
        "correct_initial_BPE_but_wrong_second":div[
            "first_native_correct_but_wrong_OR_missing_second"],
        "same_chat_and_native_tokenizer_source_vs_students":True,
        "never_gold_target_BPE_in_own_self_prefix":True,
        "previously_scored_internal_DEV_not_independent_blind":True,
    }


def summarize(streams):
    check(len(streams)==KIND_COUNT and
          Counter(x["kind"] for x in streams)=={k:1 for k in KINDS},
          "exact seven known-source DEV domain prompts per pretrained model")
    return {
        "source_count":KIND_COUNT,
        "exact_full_native_answer_and_EOS":sum(
            x["strict_entire_native_BPE_answer_including_EOS_correct"]
            for x in streams),
        "exact_visible_native_text_without_require_EOS":sum(
            x["native_visible_text_equals_gold"] for x in streams),
        "native_EOS_finished":sum(x["native_EOS_emitted"] for x in streams),
        "native_EOS_after_INVALID_or_INCOMPLETE_UTF8":sum(
            x["native_EOS_after_INVALID_or_INCOMPLETE_UTF8"] for x in streams),
        "full_valid_UTF8":sum(
            x["entire_generated_prefix_actual_GPT2_UTF8_state"]==
                "VALID_COMPLETE_UTF8" for x in streams),
        "correct_gold_FIRST_native_BPE_top1":sum(
            x["true_first_native_BPE_correct"] for x in streams),
        "first_correct_but_wrong_second":sum(
            x["correct_initial_BPE_but_wrong_second"] for x in streams),
        "total_actual_full_native_decoder_forwards":sum(
            x["real_original_native_49152_head_forward_calls"] for x in streams),
        "external_teacher_vs_compressed_student_A_B_is_not_language_competence_approval":True,
    }



from torch.nn import functional as F

TEACHER_TEMPERATURE=2.0
MAX_GOLD_BPE=30
AUDIT_BPE_CLASSES=("CONTINUATION","EOS","OTHER")


def teacher_soft_distribution_KL(teacher_logits,student_logits,temperature=TEACHER_TEMPERATURE):
    """PHYSICAL full 49,152-class teacher P vs compressed-student Q.

    Reuses true SOURCE logits at identical gold PREVIOUS BPE prefix.
    This is a numeric DISTILLATION-FEASIBILITY diagnostic, not training.
    """
    check(temperature==2.0
          and teacher_logits.shape==student_logits.shape==(1,NATIVE_VOCAB)
          and bool(torch.isfinite(teacher_logits).all())
          and bool(torch.isfinite(student_logits).all()),
          "actual 49,152 full source/student soft labels, fixed T=2")
    teacher=F.softmax(teacher_logits.detach().float()/temperature,dim=-1)
    student_log=F.log_softmax(student_logits.detach().float()/temperature,dim=-1)
    d=float(F.kl_div(student_log,teacher,reduction="sum").item())
    check(d>=-0.00001 and d<100.,
          "finite full-vocabulary SOURCE-to-compressed KL")
    return max(0.0,d)


def true_native_gold_position(tok,logits,gold_id,prior_answer_BPE,idx):
    """Gold is the CURRENT scoring label, never model prefill or optimizer."""
    audit=native_rank(logits,int(gold_id))
    v=logits.detach().float().cpu()[0]
    old=tok.convert_ids_to_tokens(int(gold_id))
    text_state=native_bytelevel_full_utf8_status(tok,prior_answer_BPE)[
        "entire_original_native_BPE_prefix_UTF8_status"]
    check(text_state in ("VALID_COMPLETE_UTF8",
                         "INCOMPLETE_UTF8_NEEDS_CONTINUATION"),
          "known CORRECT gold-prefix source must remain valid/incomplete UTF8")
    k=("CONTINUATION" if text_state=="INCOMPLETE_UTF8_NEEDS_CONTINUATION"
       else "EOS" if gold_id==tok.eos_token_id else "OTHER")
    return {
        "native_step_index_1based":idx+1,
        "native_gold_target_id":int(gold_id),
        "native_gold_piece":str(old),
        "gold_previous_native_BPE_UTF8_state":text_state,
        "correct_native_BPE_kind":k,
        "gold_top1":audit["original_full_native_correct_first"],
        "gold_rank_full_49152":
            audit["original_full_native_gold_first_rank"],
        "native_gold_target_CE":
            float(torch.logsumexp(v,dim=0)-v[int(gold_id)]),
        "native_gold_prob_softmax_T1":
            float(torch.softmax(v,dim=-1)[int(gold_id)]),
        "native_greedy_id":audit["original_full_native_greedy_first_id"],
        "native_top8_ids":audit["top8_native_ids"],
        "no_future_correct_native_BPE_used_as_INPUT":True,
    }


def audit_summarize(rows):
    check(len(rows)==KIND_COUNT and
          Counter(x["kind"] for x in rows)=={k:1 for k in KINDS},
          "all same seven ALREADY SCORED author source QA, no new EVAL")
    groups={k:[] for k in AUDIT_BPE_CLASSES}
    for example in rows:
        gold=example["full_correct_native_BPE_target_including_EOS"]
        steps=example["actual_original_full_native_teacher_forced_steps"]
        check(gold[-1]==example["real_EOS_id"]
              and len(steps)==len(gold)
              and 2<=len(gold)<=MAX_GOLD_BPE,
              "true original full-BPE plus native EOS complete source")
        for entry in steps:
            group=entry["kind"]
            check(group in groups,"known source BPE classification")
            groups[group].append(entry)
    total=sum(len(x) for x in groups.values())
    check(groups["CONTINUATION"] and groups["EOS"]
          and len(groups["EOS"])==KIND_COUNT,
          "real Cyrillic next-byte teacher and proper 7 EOS contrasts")
    result={}
    for kind,items in groups.items():
        if not items:continue
        output={"count_original_gold_native_positions":len(items)}
        for label in LABELS:
            pts=[s["actual_physically_run_per_model"][label] for s in items]
            output[label]={
                "gold_next_native_BPE_TOP1":sum(p["gold_top1"] for p in pts),
                "mean_gold_next_native_BPE_rank":
                    sum(p["gold_rank_full_49152"] for p in pts)/len(pts),
                "mean_true_gold_next_native_BPE_CE":
                    sum(p["native_gold_target_CE"] for p in pts)/len(pts),
                "mean_gold_next_native_BPE_probability":
                    sum(p["native_gold_prob_softmax_T1"] for p in pts)/len(pts),
            }
            if label!="FROZEN_SOURCE_135M_INSTRUCT":
                output[label]["mean_frozen135M_teacher_KL_at_T2"]=sum(
                    s["teacherKL_T2_full49152_against_source135m"][label]
                    for s in items)/len(items)
        result[kind]=output
    result["total_actual_original_native_GOLD_prefix_steps"]=total
    result["actual_135M_source_AND_two_43M_forward_calls"]=3*total
    result["gold_prefix_teacher_forcing_is_DIAGNOSTIC_NOT_free_answers"]=True
    return result


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,q2i3_dir,q2i7_dir,q2i8a_dir,out):
    check(not out.exists(),"research-only no old output overwrite")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    src=source_verify()
    check(src["stage"]==SOURCE_STAGE and src["TRAIN"]==70 and src["DEV"]==14,
          "only seven known already scored author-source DEV cases")
    old_dev=selected_already_scored_dev()
    q2d=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CTX,q2d_dir/Q2D_LOCAL,s93,s142)
    q2f=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    q2i1=json.loads((q2i1_dir/Q2I1_REPORT).read_text(encoding="utf8"))
    q2i3=json.loads((q2i3_dir/Q2I3_REPORT).read_text(encoding="utf8"))
    q2i7path=q2i7_dir/Q2I7_REPORT
    q2i7=json.loads(q2i7path.read_text(encoding="utf8"))
    q2i8apath=q2i8a_dir/Q2I8A_REPORT
    q2i8a=json.loads(q2i8apath.read_text(encoding="utf8"))
    check(q2i1["stage"]==Q2I1_STAGE and
          q2i3["stage"]==Q2I3_STAGE and
          q2i7["stage"]==Q2I7_STAGE and
          q2i8a["stage"]==Q2I8A_STAGE
          and q2i8a["original_S93_sha256"]==sha(s93)
          and q2i8a["original_S142_real43M_GT_sha256"]==sha(s142)
          and q2i8a["prior_true_Q2I7_1X_control_physical_run_SHA256"]==
              sha(q2i7path)
          and q2i8a["pre_registered_NEW_original_synthetic_7domain_source"]==
              src
          and q2i8a["candidate_accepted"] is False
          and q2i8a["output_gate"]==0,
          "must use genuine completed physical parent source/S142 and q2i8a")
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==NATIVE_VOCAB
          and tok.eos_token_id is not None,
          "exact original native 49,152 GPT ByteLevel tokenizer/EOS")
    examples=[prepare_native_original(tok,x) for x in old_dev]
    for x in examples:
        x["eos_native_id"]=tok.eos_token_id
        x["tokenizer"]=tok
    model,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    q2d_parent,_=load_q2d_trained_child(
        q2d_dir/Q2D_CTX,q2d["new_research_checkpoint_contextual"])
    q2f_parent=load_physical_q2f_child(
        q2f_dir/Q2F_CTX,
        q2f["projector_contextual_child_checkpoint"],q2d_parent)
    q2i1_parent=load_q2i1_physical_context(
        q2i1_dir/Q2I1_CTX,
        q2i1["learned_checkpoint_fp16"]["joint_contextual"],q2f_parent)
    q2i3_parent=check_trained_joint(
        q2i3_dir/RECURRENT_CKPT,
        q2i3["research_only_updated_FULL_neural_child_FP16"]["GT_RECURRENT"],
        q2i1_parent)
    q2i7_parent=verify_physical_1x_parent(
        q2i7_dir/"S119Q2I7_UNIFORM1X_GT12_NRA3_NPA1_NATIVEHEAD_FP16_G0.safetensors",
        q2i7["new_1x_physical_FP16_checkpoint_and_GT12_sidecar"][
            "trained_1x_neural_fp16"],q2i3_parent)
    fitted=physical_q2i8a_fitted_child(
        q2i8a_dir/Q2I8A_REC,
        q2i8a["reversible_research_only_FP16_neural_A_B_and_GT12_gates"][
            "GT_RECURRENT_neural"],q2i7_parent)
    gates=physical_q2i8a_gates(
        q2i8a_dir/Q2I8A_GT,
        q2i8a["reversible_research_only_FP16_neural_A_B_and_GT12_gates"][
            "GT_RECURRENT_12_gates"])
    model.eval()
    model.model.add_module("nra3_s119m_onpath",fitted)
    original_digest=old_original_param_digest(model)
    frozen_full_gt={k:v.detach().cpu().clone() for k,v in full.state_dict().items()}
    child_before={k:v.detach().cpu().clone()
                  for k,v in fitted.state_dict().items()}
    native_full_gt_gate_contract(full,bridge)
    check(all(not p.requires_grad for n,p in model.named_parameters()
              if "nra3_s119m_onpath" not in n),
          "all 43M original Core12/GreenTunnel/LM remain unchanged")
    # Independent pinned 135M SOURCE teacher: direct full native source
    # checkpoint, not a hand-written oracle, pseudo-label or proxy.
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32
    ).eval().requires_grad_(False)
    check(teacher.config.vocab_size==NATIVE_VOCAB and
          teacher.get_input_embeddings().weight.shape[0]==NATIVE_VOCAB
          and teacher.get_output_embeddings().weight.shape[0]==NATIVE_VOCAB,
          "real 135M SOURCE model same exact original native tokenizer/head")
    reported_teacher_parameters=sum(p.numel() for p in teacher.parameters())
    check(120_000_000<=reported_teacher_parameters<=150_000_000,
          "must load real pinned 135M pretrained instruct source, not 43M proxy")

    measure=[];physical=0
    with torch.inference_mode():
        # No original frozen neural weights or Q2I8a child may update.
        try:
            for source in examples:
                gold=list(source["answer_native_with_eos"])
                check(2<=len(gold)<=MAX_GOLD_BPE
                      and gold[-1]==tok.eos_token_id,
                      "all target native answer+EOS positions measured")
                steps=[]
                for t,target in enumerate(gold):
                    # GOLD-PREFIX diagnostic ONLY. This exact target is
                    # withheld until the three actual native forwards finish.
                    prefix=source["prefix"]+gold[:t]
                    check(prefix[len(source["prefix"]):]==gold[:t],
                          "strict PREVIOUS correct native BPE only")
                    teacher_logits=source_teacher_step(teacher,prefix)
                    set_gt_route(full,"GT_ZERO_BASELINE",frozen_full_gt["gates"])
                    compressed_logits=physical_original_step(
                        model,sources,bridge,full,expansion,
                        prefix,source["prompt_features"])
                    set_gt_route(full,"GT_RECURRENT",gates)
                    trained_logits=physical_trained_step(
                        model,sources,bridge,full,expansion,
                        fitted,prefix,source["prompt_features"])
                    all_heads={
                        "FROZEN_SOURCE_135M_INSTRUCT":teacher_logits,
                        "FROZEN_S142_43M_ORIGINAL":compressed_logits,
                        "Q2I8A_FITTED_43M_GT12_NRA3_NPA1":trained_logits,
                    }
                    check(set(all_heads)==set(LABELS)
                          and all(v.shape==(1,NATIVE_VOCAB)
                                  for v in all_heads.values()),
                          "original 135M/43M/GT all identical native 49k")
                    metrics={
                        label:true_native_gold_position(
                            tok,logits,int(target),gold[:t],t)
                        for label,logits in all_heads.items()
                    }
                    groups={x["correct_native_BPE_kind"] for x in metrics.values()}
                    check(len(groups)==1,"same native gold BPE and prior byte status")
                    kind=next(iter(groups))
                    kl={
                        label:teacher_soft_distribution_KL(
                            teacher_logits,all_heads[label])
                        for label in LABELS if label!="FROZEN_SOURCE_135M_INSTRUCT"
                    }
                    steps.append({
                        "native_position":t+1,
                        "kind":kind,
                        "prior_correct_answer_native_BPE_ids":gold[:t],
                        "current_correct_native_BPE_ONLY_SCORER":int(target),
                        "actual_physically_run_per_model":metrics,
                        "teacherKL_T2_full49152_against_source135m":kl,
                        "original_full_vocab":NATIVE_VOCAB,
                        "no_train_no_optimizer_no_OWN_claim":True,
                    })
                    physical+=3
                measure.append({
                    "source_id":source["id"],
                    "kind":source["kind"],
                    "source_split":"PREVIOUSLY_SCORED_DEV",
                    "real_EOS_id":int(tok.eos_token_id),
                    "full_correct_native_BPE_target_including_EOS":gold,
                    "actual_original_full_native_teacher_forced_steps":steps,
                    "same_identical_native_chat_prefix_and_PREVIOUS_gold_BPE_for_three_models":True,
                })
        finally:
            with torch.no_grad():
                full.gates.copy_(frozen_full_gt["gates"])
                full.mix_logits.copy_(frozen_full_gt["mix_logits"])
    check(old_original_param_digest(model)==original_digest
          and all(torch.equal(v,full.state_dict()[k].detach().cpu())
                  for k,v in frozen_full_gt.items())
          and all(torch.equal(v,fitted.state_dict()[k].detach().cpu())
                  for k,v in child_before.items()),
          "all 135M/43M originals, GT states and prior trained Q2I8a must be immutable")
    totals=audit_summarize(measure)
    check(totals["actual_135M_source_AND_two_43M_forward_calls"]==physical,
          "every real original 135M/43M full 49k native forward counted")
    result={
        "stage":STAGE,
        "pinned_source_original_pretrained_MODEL":MODEL,
        "pinned_source_original_pretrained_REVISION":MODEL_SHA,
        "true_original_source_pretrained_parameter_count":
            reported_teacher_parameters,
        "physical_original_S93_sha256":sha(s93),
        "physical_original_S142_43M_GT_sha256":sha(s142),
        "previous_Q2I8A_2571_physical_run_sha256":sha(q2i8apath),
        "same_7_source_original_authored_PREVIOUSLY_SCORED_DEV_only":True,
        "original_source_DEV_SHA256":src["DEV_SHA256"],
        "all_original_native_49152_vocabulary":NATIVE_VOCAB,
        "source_teacher_soft_logits_temperature":TEACHER_TEMPERATURE,
        "all_7_SOURCE_vs_43M_gold_prefix_causal_comparisons":measure,
        "by_actual_UTF8_continuation_EOS_and_other_gold_NATIVE_positions":totals,
        "real_teacher135M_and_43M_original_head_calls":physical,
        "the_pinned_teacher_is_not_a_correct_semantic_QA_oracle":True,
        "no_gold_in_autoregressive_OWN_prefix_claim_not_applicable":True,
        "ALL_ASSISTED_gold_BPE_enter_only_PREVIOUS_causal_prefix":True,
        "no_new_licensed_training_corpus_and_no_optimizers":True,
        "full_616_J7_EOS_and_packed_bpp_CPU_not_tested":True,
        "old_S118_GOLD_EVAL128_J7_answers_not_loaded":True,
        "research_original_S142_FROZEN_and_original_135M_FROZEN":True,
        "previous_q2i8a_child_FP16_and_original_GT12_RESTORED":True,
        "no_main_LIVE_Render_Supabase_Jessica_APK_or_secrets":True,
        "candidate_accepted":False,
        "output_gate":0,
        "decision":"SOURCE_SOFTLOGITS_DISTILLATION_TARGET_VALIDATION_ONLY_NOT_ACCEPTED",
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(json.dumps(
        result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2I8C_REAL_SOURCE135M_VS_S14243M_FULL_GOLD_PREFIX_NATIVE "+
          json.dumps({
              "physical_real_native_source135M_43M_calls":physical,
              "gold_native_positions_each_source":physical//3,
              "full_NATIVE_49152_teacherKL_T2":True,
              "actual_native_UTF8_continuation_and_EOS_by_model":totals,
              "original_weights_no_mutation":True,
              "candidate_accepted":False,
          },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2d_dir","q2f_dir",
                "q2i1_dir","q2i3_dir","q2i7_dir","q2i8a_dir","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.q2i1_dir,
        a.q2i3_dir,a.q2i7_dir,a.q2i8a_dir,a.out)


if __name__=="__main__":
    main()
