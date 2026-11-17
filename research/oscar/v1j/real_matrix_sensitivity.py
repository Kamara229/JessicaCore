#!/usr/bin/env python3
"""OSCAR v1J: REAL independent pretrained 42M FP32 per-matrix causal audit.

One original 8x512 pretrained TinyStories FP32 teacher vs original cold
5bit K32/V8 student with two genuinely trained recurrent full8 Green Tunnels.
For EACH of ALL 57 original 2D matrices, temporarily restore only that
pretrained FP32 matrix (UNPAID DIAGNOSTIC, not an inference/model package).
Compare original whole-vocab logits and all8 hidden states at the FIRST and
SECOND student-history token positions on the SAME 12 original TRAIN prompts.
Restore each original decoded 5-bit matrix BITWISE before proceeding. Never
touch the 20+8 previously exposed public DEBUG, sealed QA or CALIBRI S142.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from collections import Counter

import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from direct_codebook_v59 import CB59
from oscar.v1e.audit_checkpoint import source as external_fp32
from oscar.v1e.safe_pretrained_llama import from_checkpoint,materialize_decoded_cb59
from oscar.v1f.green_tunnel import FullLayerGreenTunnel,DEPTH
from oscar.v1f.train_real_all8_green_tunnel import (
    SOURCE_SHA,SOURCE_REV,TOKENIZER,TOKENIZER_REV,
    EXPECTED_ORIGINAL_UNIQUE,TRAIN_STORIES,prompt_ids,
)
from oscar.v1g.real_onpolicy_gt8 import (
    V1F_CB59_SHA,V1F_CB61_SHA,V1F_GT_SHA,
)
from oscar.v1h.real_dual_phase_inference import V1G_SHA

RANKING_FIRST_KL_WEIGHT=.65
RANKING_LATER_KL_WEIGHT=.35
RANKING_FINAL_HIDDEN_WEIGHT=2.0
TEMP=2.0
EXPECTED_MATRICES=57


def digest(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()


def dump(path,x):
    Path(path).write_text(json.dumps(x,indent=2,sort_keys=True,
                      ensure_ascii=False)+"\n",encoding="utf8")


def log(stage,**kw):
    print(json.dumps({"stage":stage,**kw},ensure_ascii=False),flush=True)


def classify(name):
    if name=="tok_embeddings.weight":return "tied_input_output_embedding"
    if ".attention." in name:
        return ("attention_output_WO" if name.endswith(".wo.weight")
                else "attention_QKV")
    if ".feed_forward." in name:
        return ("ffn_output_W2" if name.endswith(".w2.weight")
                else "ffn_expand_W1_W3")
    raise AssertionError("Unknown original pretrained 42M LLaMA physical 2D source matrix "+name)


def layer_of(name):
    return int(name.split(".")[1]) if name.startswith("layers.") else -1


@torch.inference_mode()
def measure_real_original_forward(student,gtfirst,gtlater,items):
    """Teacher targets already cached at EXACT candidate-independent contexts.

    No teacher-greedy replacements or debugging prompts are ever supplied.
    """
    rows=[]
    for entry in items:
        first_input=entry["original_TRUE_pretrained_training_prompt"]
        second_input=entry["fixed_baseline_student_self_history"]
        row={}
        for name,ids,orig_logits,orig_states in [
            ("first",first_input,entry["original_FP32_first_logits"],
             entry["original_FP32_first_hidden"]),
            ("later",second_input,entry["original_FP32_later_logits"],
             entry["original_FP32_later_hidden"])]:
            with gtfirst.scope(name=="first"),gtlater.scope(name=="later"):
                result,_=student(ids)
            if (gtfirst.last_coverage!=tuple(range(DEPTH))
                or gtlater.last_coverage!=tuple(range(DEPTH))
                or len(gtlater.latest_layers)!=DEPTH):
                raise AssertionError("Oscar original full8 recurrent GT did not visit all original pretrained layers")
            s=result[0,-1].float()
            t=orig_logits.detach().float()
            if s.shape!=t.shape or s.numel()!=32000:
                raise AssertionError("Oscar teacher original 32k 8layer step logits changed")
            KL=float(F.kl_div(
                 F.log_softmax(s[None]/TEMP,dim=-1),
                 F.softmax(t[None]/TEMP,dim=-1),
                 reduction="batchmean")*TEMP**2)
            expected_top=int(t.argmax())
            cos_by_layer=[]
            for i,corrected in enumerate(gtlater.latest_layers):
                a=corrected[0,-1].float()
                b=orig_states[i][0,-1].float()
                if a.shape!=b.shape or a.numel()!=512:
                    raise AssertionError("Original independently pretrained FP32 and 5bit state geometry mismatch")
                cos_by_layer.append(float(F.cosine_similarity(a[None],b[None])))
            row[name]={
                "pretrained_FP32_next_32k_KL_T2":KL,
                "teacher_top1_rank_under_original_5bit":int(
                    (s>s[expected_top]).sum())+1,
                "real_FP32_top1_equal":int(s.argmax())==expected_top,
                "real_teacher_FP32_all8_hidden_cos":cos_by_layer,
                "real_teacher_FP32_final_decoder_hidden_cos":cos_by_layer[-1],
                "real_teacher_FP32_mean8_hidden_cos":
                    sum(cos_by_layer)/DEPTH
            }
        rows.append(row)
    return rows


def summarize(rows):
    if len(rows)!=12:
        raise AssertionError("Oscar original 12 TRAIN-only prompts missing")
    result={}
    for name in ("first","later"):
        result[name]={
          "n":len(rows),
          "mean_true_original_FP32_32k_soft_KL_T2":sum(
              r[name]["pretrained_FP32_next_32k_KL_T2"] for r in rows)/len(rows),
          "teacher_FP32_first_token_TOP1_parity":sum(
              int(r[name]["real_FP32_top1_equal"]) for r in rows),
          "mean_FP32_teacher_top1_rank_under_student":sum(
              r[name]["teacher_top1_rank_under_original_5bit"] for r in rows)/len(rows),
          "mean_true_original_FP32_final_decoder_cos":sum(
              r[name]["real_teacher_FP32_final_decoder_hidden_cos"] for r in rows)/len(rows),
          "mean_true_original_FP32_all8_decoder_cos":sum(
              r[name]["real_teacher_FP32_mean8_hidden_cos"] for r in rows)/len(rows),
        }
        result[name]["per_decoder_layer_mean_hidden_cos"]=[
           sum(r[name]["real_teacher_FP32_all8_hidden_cos"][i]
               for r in rows)/len(rows) for i in range(DEPTH)
        ]
    return result


def improvement_score(before,after):
    f=before["first"]["mean_true_original_FP32_32k_soft_KL_T2"]-\
      after["first"]["mean_true_original_FP32_32k_soft_KL_T2"]
    later=before["later"]["mean_true_original_FP32_32k_soft_KL_T2"]-\
          after["later"]["mean_true_original_FP32_32k_soft_KL_T2"]
    diff=.5*((after["first"]["mean_true_original_FP32_final_decoder_cos"]-
              before["first"]["mean_true_original_FP32_final_decoder_cos"])+
             (after["later"]["mean_true_original_FP32_final_decoder_cos"]-
              before["later"]["mean_true_original_FP32_final_decoder_cos"]))
    return (RANKING_FIRST_KL_WEIGHT*f+
            RANKING_LATER_KL_WEIGHT*later+
            RANKING_FINAL_HIDDEN_WEIGHT*diff)


def run(args):
    torch.manual_seed(421912)
    torch.set_num_threads(2)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Oscar v1J sensitivity result directory must be initially empty")
    paths={
       "original_CB59":(Path(args.original_CB59),V1F_CB59_SHA),
       "original_CB61":(Path(args.original_CB61),V1F_CB61_SHA),
       "v1F_first_full8_GT":(Path(args.v1f_first_FP16),V1F_GT_SHA),
       "v1G_later_full8_GT":(Path(args.v1g_later_FP16),V1G_SHA)
    }
    for name,(path,expected) in paths.items():
        if not path.is_file() or digest(path.read_bytes())!=expected:
            raise AssertionError("Oscar source physical v1H SHA invalid: "+name)
    if len(TRAIN_STORIES)!=12 or len(set(TRAIN_STORIES))!=12:
        raise AssertionError("Oscar 12 previously preregistered TRAIN-only contexts changed")
    core=CB59(paths["original_CB59"][0].read_bytes())
    if core.meta["original_unique_parameter_count"]!=EXPECTED_ORIGINAL_UNIQUE:
        raise AssertionError("Oscar false original 41,689,600 unique trained source")
    matrix_names=sorted(k for k,x in core.meta["tensor_descriptors"].items()
                        if x["dtype"]=="CB59_K32V8")
    if (len(matrix_names)!=EXPECTED_MATRICES or
        Counter(classify(k) for k in matrix_names)!=Counter({
           "tied_input_output_embedding":1,
           "attention_output_WO":8,
           "attention_QKV":24,
           "ffn_output_W2":8,
           "ffn_expand_W1_W3":16
        })):
        raise AssertionError("Original source pretrained 8x512 57 matrix inventory/geometry changed")
    prereg={
      "stage":"OSCAR_v1J_original_pretrained_42M_FP32_matrix_specific_interlayer_causal_sensitivity_AUDIT",
      "externally_pretrained_teacher_original_42M_SHA256":SOURCE_SHA,
      "external_pretrained_original_FP32_HF_revision":SOURCE_REV,
      "original_uniquely_trained_FP32_weights":EXPECTED_ORIGINAL_UNIQUE,
      "original_model_8x512_NOT_Calibri_12x320":True,
      "original_FP32_teacher_UNPAID_for_each_MATRIX_counterfactual":True,
      "not_an_inference_0p7_bpp_model":True,
      "original_source_CB59_SHA":V1F_CB59_SHA,
      "original_source_CB61_SHA":V1F_CB61_SHA,
      "original_trained_GT_first_SHA":V1F_GT_SHA,
      "original_trained_GT_later_SHA":V1G_SHA,
      "source_original_57_2D_unique_matrix_names":matrix_names,
      "purely_predeclared_original_12_TRAIN_prompts":list(TRAIN_STORIES),
      "not_accessing_any_prior_public_DEBUG_or_sealed_QA":True,
      "each_candidate_true_FP32_matrix_replaces_only_one_5bit_decoded_matrix_UNPAID":True,
      "original_5bit_student_and_both_GT_exactly_restored_for_every_next_candidate":True,
      "second_position_teacher_conditions_on_FIXED_baseline_student_first_token":True,
      "ranking_first_KL_weight":RANKING_FIRST_KL_WEIGHT,
      "ranking_later_KL_weight":RANKING_LATER_KL_WEIGHT,
      "ranking_final_hidden_weight":RANKING_FINAL_HIDDEN_WEIGHT,
      "no_original_616_J7_nativeCPU90_or_production_12layer_test":True,
      "no_main_LIVE_Render_Supabase_NRA_APK_ordinaryJessicaChat":True,
    }
    dump(out/"OSCAR_V1J_PRECOMMIT_BEFORE_ORIGINAL_FP32_TEACHER_LOAD.json",prereg)
    _,original,audit=external_fp32(out)
    if (audit["archive_SHA256"]!=SOURCE_SHA or
        audit["pinned_exact_HF_revision_sha40"]!=SOURCE_REV or
        audit["model_parameters_real_unique"]!=EXPECTED_ORIGINAL_UNIQUE):
        raise AssertionError("Oscar original independent pretrained FP32 source unauthenticated")
    teacher=from_checkpoint(original).eval().requires_grad_(False)
    student=materialize_decoded_cb59(teacher,paths["original_CB59"][0].read_bytes())
    student.eval().requires_grad_(False)
    old=FullLayerGreenTunnel().install(student)
    new=FullLayerGreenTunnel().install(student)
    old.restore_paid(load_file(str(paths["v1F_first_full8_GT"][0])))
    new.restore_paid(load_file(str(paths["v1G_later_full8_GT"][0])))
    old.requires_grad_(False)
    new.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(TOKENIZER,revision=TOKENIZER_REV)
    if tok.vocab_size!=32000 or tok.eos_token_id is None:
        raise AssertionError("Original external 42M FP32 proxy tokenizer inventory wrong")
    # Establish candidate-independent student-generated second-token context;
    # it would confound matrix sensitivity if each hybrid used different
    # token histories, so one fixed original compressed student's first
    # generated token drives ALL 57 later counterfactual comparisons.
    data=[]
    with torch.inference_mode():
        for text in TRAIN_STORIES:
            q=prompt_ids(tok,text)
            with old.scope(True),new.scope(False):
                baseline_first,_=student(q)
            first_token=int(baseline_first[0,-1].argmax())
            p2=torch.cat((q,q.new_tensor([[first_token]])),dim=1)
            first_ref,h0=teacher(q,hidden=True)
            later_ref,h1=teacher(p2,hidden=True)
            data.append({
              "original_TRUE_pretrained_training_prompt":q,
              "fixed_baseline_student_self_history":p2,
              "original_FP32_first_logits":first_ref[0,-1].float().clone(),
              "original_FP32_later_logits":later_ref[0,-1].float().clone(),
              "original_FP32_first_hidden":[x.clone() for x in h0[1:9]],
              "original_FP32_later_hidden":[x.clone() for x in h1[1:9]],
              "original_TRY_BASELINE_student_FIRST_token":first_token,
              "TRAIN_original_prompt_sha256":digest(text.encode()),
            })
    baseline_rows=measure_real_original_forward(student,old,new,data)
    baseline=summarize(baseline_rows)
    log("OSCAR_V1J_REAL_FP32_ORIGINAL_42M_INTERLAYER_BASELINE",
        unique=EXPECTED_ORIGINAL_UNIQUE,
        original_matrices=len(matrix_names),
        first_KL=round(baseline["first"]["mean_true_original_FP32_32k_soft_KL_T2"],5),
        later_KL=round(baseline["later"]["mean_true_original_FP32_32k_soft_KL_T2"],5),
        final_first_hidden_cos=round(baseline["first"]["mean_true_original_FP32_final_decoder_cos"],5))
    # Move exactly one source tensor to original teacher FP32 and restore
    # it bitwise. NONE of these 57 hybrid students is physically compressed
    # or authorized for deployment; this is sensitivity diagnosis ONLY.
    candidate_data=[]
    begin=time.monotonic()
    s=student.state_dict()
    t=teacher.state_dict()
    with torch.inference_mode():
        for n,key in enumerate(matrix_names):
            if key not in s or key not in t or s[key].shape!=t[key].shape:
                raise AssertionError("Original trained FP32 matrix counterfactual missing "+key)
            original_5bit=s[key].detach().clone()
            if torch.equal(original_5bit,t[key]):
                raise AssertionError("Matrix unexpectedly already identical to original pretrained FP32: "+key)
            # Exact full FP32 tensor restoration ONLY in this scratch model.
            s[key].copy_(t[key])
            try:
                hybrid=measure_real_original_forward(student,old,new,data)
                measured=summarize(hybrid)
            finally:
                s[key].copy_(original_5bit)
            if not torch.equal(s[key],original_5bit):
                raise AssertionError("Original cold compressed source tensor wasn't RESTORED bitwise "+key)
            record={
              "source_unique_original_matrix_name":key,
              "category":classify(key),
              "decoder_layer_zero_based":layer_of(key),
              "original_matrix_unique_FP32_cells":int(t[key].numel()),
              "UNPAID_original_FP32_matrix_substitution":True,
              "not_a_compressed_inference_candidate":True,
              "original_FP32_teacher_same_context_hard_first_matches":measured["first"][
                 "teacher_FP32_first_token_TOP1_parity"],
              "original_FP32_teacher_same_context_hard_later_matches":measured["later"][
                 "teacher_FP32_first_token_TOP1_parity"],
              "original_TRUE_pretrained_FP32_new_first_KL_T2":measured["first"][
                 "mean_true_original_FP32_32k_soft_KL_T2"],
              "original_TRUE_pretrained_FP32_new_later_KL_T2":measured["later"][
                 "mean_true_original_FP32_32k_soft_KL_T2"],
              "first_soft_KL_T2_reduction_from_single_original_FP32_matrix":
                  baseline["first"]["mean_true_original_FP32_32k_soft_KL_T2"]-
                  measured["first"]["mean_true_original_FP32_32k_soft_KL_T2"],
              "later_soft_KL_T2_reduction_from_single_original_FP32_matrix":
                  baseline["later"]["mean_true_original_FP32_32k_soft_KL_T2"]-
                  measured["later"]["mean_true_original_FP32_32k_soft_KL_T2"],
              "first_original_FP32_hidden_last_cos_delta":
                 measured["first"]["mean_true_original_FP32_final_decoder_cos"]-
                 baseline["first"]["mean_true_original_FP32_final_decoder_cos"],
              "later_original_FP32_hidden_last_cos_delta":
                 measured["later"]["mean_true_original_FP32_final_decoder_cos"]-
                 baseline["later"]["mean_true_original_FP32_final_decoder_cos"],
              "predeclared_functional_matrix_sensitivity_score":
                   improvement_score(baseline,measured),
              "all8_first_hidden_cos_when_single_matrix_unpaid_FP32":
                  measured["first"]["per_decoder_layer_mean_hidden_cos"],
              "all8_later_hidden_cos_when_single_matrix_unpaid_FP32":
                  measured["later"]["per_decoder_layer_mean_hidden_cos"],
            }
            candidate_data.append(record)
            if (n+1)%8==0 or n==len(matrix_names)-1:
                log("OSCAR_V1J_REAL_ORIGINAL_PRETRAINED_FP32_MATRIX_INTERLAYER_AUDIT",
                    evaluated=n+1,all=len(matrix_names),
                    matrix=key,
                    classification=record["category"],
                    predeclared_sensitivity=round(
                       record["predeclared_functional_matrix_sensitivity_score"],5),
                    all_original_compressed_matrices_restored_after_each=True)
    if (len(candidate_data)!=57 or
        digest(paths["original_CB59"][0].read_bytes())!=V1F_CB59_SHA or
        digest(paths["original_CB61"][0].read_bytes())!=V1F_CB61_SHA or
        digest(paths["v1F_first_full8_GT"][0].read_bytes())!=V1F_GT_SHA or
        digest(paths["v1G_later_full8_GT"][0].read_bytes())!=V1G_SHA):
        raise AssertionError("Oscar v1J unauthorised changed pretrained source or GT during per-matrix audit")
    check=summarize(measure_real_original_forward(student,old,new,data))
    for phase in ("first","later"):
        for key in ("mean_true_original_FP32_32k_soft_KL_T2",
                    "mean_true_original_FP32_final_decoder_cos",
                    "teacher_FP32_first_token_TOP1_parity"):
            if check[phase][key]!=baseline[phase][key]:
                raise AssertionError("Oscar per-matrix intervention leaked original FP32 tensors into unmodified 5bit source")
    old.uninstall()
    new.uninstall()
    ordered=sorted(candidate_data,
        key=lambda r:r["predeclared_functional_matrix_sensitivity_score"],
        reverse=True)
    groups={}
    for k in sorted(set(r["category"] for r in ordered)):
        vals=[r for r in candidate_data if r["category"]==k]
        groups[k]={
           "candidate_count":len(vals),
           "mean_unpaid_single_original_matrix_sensitivity":sum(
               r["predeclared_functional_matrix_sensitivity_score"] for r in vals)/len(vals),
           "max_single_matrix":{"name":max(vals,
                   key=lambda r:r["predeclared_functional_matrix_sensitivity_score"])[
                   "source_unique_original_matrix_name"],
               "score":max(r["predeclared_functional_matrix_sensitivity_score"] for r in vals)},
           "mean_KL_T2_FIRST_reduction":sum(
               r["first_soft_KL_T2_reduction_from_single_original_FP32_matrix"] for r in vals)/len(vals),
           "mean_KL_T2_LATER_reduction":sum(
               r["later_soft_KL_T2_reduction_from_single_original_FP32_matrix"] for r in vals)/len(vals)
        }
    result={
      **prereg,
      "true_original_FP32_teacher_42M_8layer_verified":True,
      "original_PRETRAINED_FP32_matrix_substitutions_57_all_unpaid_diagnostic":True,
      "real_source_pack_CRC_SHA_authenticated_and_NO_MODIFICATION":True,
      "original_5bit_source_before_after_bitwise_same":True,
      "baseline_real_original_5bit_all8_layer_vs_original_FP32":baseline,
      "AFTER_ALL_57_original_5bit_state_exact_equal_baseline":check,
      "all_57_single_FP32_matrix_counterfactuals_ranked_by_predeclared_TRAIN_metric":ordered,
      "top_ten_most_sensitive_original_matrices":ordered[:10],
      "bottom_ten_negative_synergistic_single_matrix_restore":ordered[-10:],
      "category_group_sensitivity":groups,
      "actual_CPU_matrix_probes":len(matrix_names),
      "actual_CPU_seconds_matrix_audit":round(time.monotonic()-begin,3),
      "original_fp32_teacher_only_offline_not_in_runtime":True,
      "diagnostic_source_original_FP32_substitution_is_UNPAID_and_NOT_0p7_model":True,
      "no_original_FP32_source_43M_Calibri_or_616_J7_modified":True,
      "model_accepted_or_created":False,
      "next_reencode_sensitive_original_source_requires_separate_physical_proof":True,
    }
    dump(out/"OSCAR_V1J_FINAL_REAL_ORIGINAL42M_FP32_MATRIX_CAUSAL_SENSITIVITY_RANKING.json",
         result)
    log("OSCAR_V1J_FINAL_ORIGINAL_FP32_CAUSAL_INTERLAYER_MATRIX_RANKING",
        unique_FP32_original=EXPECTED_ORIGINAL_UNIQUE,
        original_source_matrices_measured=57,
        first_original_42M_FP32_KL_baseline=round(
            baseline["first"]["mean_true_original_FP32_32k_soft_KL_T2"],5),
        later_original_42M_FP32_KL_baseline=round(
            baseline["later"]["mean_true_original_FP32_32k_soft_KL_T2"],5),
        top5=[{
           "name":a["source_unique_original_matrix_name"],
           "type":a["category"],
           "score":round(a["predeclared_functional_matrix_sensitivity_score"],5),
           "first_KL_improvement":round(a[
             "first_soft_KL_T2_reduction_from_single_original_FP32_matrix"],4),
         } for a in ordered[:5]],
        top10_positive_count=sum(
            a["predeclared_functional_matrix_sensitivity_score"]>0
            for a in ordered[:10]),
        original_CB59_and_both_trainedGT_physically_unmodified=True,
        any_new_0p7_deployable_model_built=False,
        original_S142_LIVE_NRA_untouched=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("original_CB59","original_CB61","v1f_first_FP16","v1g_later_FP16","out"):
        p.add_argument(name,type=Path)
    run(p.parse_args())
