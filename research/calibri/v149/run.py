"""CB149: one PAID contextual int4 rank1 adapter, true disjoint VALID.

After v147 static prior lowered NLL but teacher top1 was almost absent and
v148 tiny QA improved BoolQ (10/16) without teacher-choice agreement,
this tests genuinely input-CONDITIONAL output correction, not another
global output token prior. Fit once on original TRAIN FIT; write and verify
complete archive BEFORE opening any new shifted source VALID.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model,get_tokens
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,MAX_SOURCE_BYTES
from v142.scoring import compare_logits,aggregate
from v147.codec import unpack as unpack_cb147
from v147.runtime import fused_half_head
from v141.run import ORIGINAL_V140_SHA
from v143.runtime import PackedFusedHead
from .protocol import (CB147_SHA,FIT_DOMAINS,VALID_DOMAINS,WINDOW,
                       FIT_WINDOWS_EACH,VALID_WINDOWS_EACH,TRAIN_HIDDEN,VOCAB)
from .fit import fit_strict_135m
from .codec import pack,unpack,report
from .runtime import attach
from .sources import load_new_valid


@torch.inference_mode()
def locked_fit(teacher,student):
    arrays={"hidden":[],"teacher":[],"prior":[]}
    origins={}
    for domain in FIT_DOMAINS:
        tokens,origin=get_tokens(domain)
        if origin.get("split")!="train" or len(tokens)!=FIT_WINDOWS_EACH*WINDOW:
            raise ValueError("unexpected fit source interval")
        origins[domain]=origin
        for k in range(FIT_WINDOWS_EACH):
            ids=tokens[k*WINDOW:(k+1)*WINDOW][None,:]
            reference=teacher(input_ids=ids,use_cache=False).logits[0,:-1]
            hidden=student.model(input_ids=ids,use_cache=False).last_hidden_state[:, :-1]
            approximate=student.lm_head(hidden)[0]
            if (hidden.shape!=(1,WINDOW-1,TRAIN_HIDDEN)
                    or reference.shape!=(WINDOW-1,VOCAB)
                    or approximate.shape!=reference.shape):
                raise ValueError("student/teacher full conditional decoder mismatch")
            arrays["hidden"].append(hidden[0].detach().cpu().numpy().copy())
            arrays["teacher"].append(reference.detach().cpu().numpy().copy())
            arrays["prior"].append(approximate.detach().cpu().numpy().copy())
    values={k:np.concatenate(v,axis=0).astype(np.float32)
            for k,v in arrays.items()}
    if values["hidden"].shape!=(504,TRAIN_HIDDEN):
        raise ValueError("not exactly 504 original predeclared FIT source tokens")
    return values,origins


@torch.inference_mode()
def validate(teacher,student,baseline,prior,context):
    # Physically frozen candidate already exists before opening these sources.
    sources,origins=load_new_valid()
    heads={"CB59":baseline,"CB147_static_int8_prior":prior,
           "CB149_contextual_int4_rank1":context}
    scores={a:{d:[] for d in VALID_DOMAINS} for a in heads}
    try:
        for domain in VALID_DOMAINS:
            tokens=sources[domain]
            for k in range(VALID_WINDOWS_EACH):
                ids=tokens[k*WINDOW:(k+1)*WINDOW][None,:]
                ref=teacher(input_ids=ids,use_cache=False).logits.detach()
                # Factorized hidden is identical across all three arms.
                clock=perf_counter()
                hidden=student.model(input_ids=ids,use_cache=False).last_hidden_state
                shared_decoder=perf_counter()-clock
                if hidden.shape!=(1,WINDOW,TRAIN_HIDDEN):
                    raise ValueError("full packed native decoder changed")
                for arm,head in heads.items():
                    t0=perf_counter()
                    pred=head(hidden).detach()
                    dt=perf_counter()-t0
                    observed=compare_logits(ref,pred,ids)
                    observed["window_index"]=k
                    observed["same_native_hidden_for_all_three_arms"]=True
                    observed["native_paid_head_seconds"]=dt
                    observed["shared_native_decoder_seconds"]=shared_decoder
                    scores[arm][domain].append(observed)
    finally:
        student.lm_head=baseline
    summary={arm:{domain:aggregate(records[domain]) for domain in VALID_DOMAINS}
             for arm,records in scores.items()}
    checks={}
    for domain in VALID_DOMAINS:
        old=summary["CB147_static_int8_prior"][domain]
        new=summary["CB149_contextual_int4_rank1"][domain]
        control=summary["CB59"][domain]
        checks[domain]={
            "new_minus_prior_true_next_token_NLL":
                new["mean_TRUE_next_token_NLL"]-old["mean_TRUE_next_token_NLL"],
            "new_minus_prior_teacher_KL":
                new["teacher_to_candidate_KL"]-old["teacher_to_candidate_KL"],
            "new_minus_CB59_NLL":
                new["mean_TRUE_next_token_NLL"]-control["mean_TRUE_next_token_NLL"],
            "new_minus_CB59_teacher_KL":
                new["teacher_to_candidate_KL"]-control["teacher_to_candidate_KL"],
            "new_exact_teacher_next_token_agreement":
                new["teacher_next_token_argmax_agreement_fraction"],
        }
    return summary,checks,origins,scores


def run(host,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    original=host.read_bytes()
    parent=unpack_cb147(original)
    if parent.sha256!=CB147_SHA or parent.paid.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("wrong CB147/CB140 source")
    if hashlib.sha256(parent.paid.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original CB59 source replaced")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("wrong FP32 source teacher")
    with TemporaryDirectory(dir=out) as temp:
        root=Path(temp)
        compressed=root/"locked_native.cb59"
        compressed.write_bytes(parent.paid.original)
        student,metadata,_=make_model(
            "baseline",SimpleNamespace(original=compressed,folder=root))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("not physically compressed original source")
        cb59_head=student.lm_head
        half_head=fused_half_head(student,parent.paid,root)
        prior_head=PackedFusedHead(half_head.handle,half_head.kernel,parent.bias)
        student.lm_head=prior_head
        if student.model.embed_tokens.handle is not cb59_head.handle:
            raise ValueError("tied embeddings/output lost")

        # FIT stage: only previously assigned original TRAIN FIT intervals.
        fit,fit_origins=locked_fit(teacher,student)
        u,intercept,vector,scale,fit_report=fit_strict_135m(
            fit["hidden"],fit["teacher"],fit["prior"])
        blob=pack(original,u,intercept,vector,scale)
        physical_path=out/"CALIBRI_v149_paid_int4_context_rank1.cb149"
        physical_path.write_bytes(blob)
        candidate=unpack(physical_path.read_bytes())
        if candidate.parent.sha256!=CB147_SHA or candidate.size>MAX_SOURCE_BYTES:
            raise ValueError("physically paid source or full-model budget failed")
        context_head=attach(student,candidate)
        if context_head.handle is not cb59_head.handle:
            raise AssertionError("pretrained base weights no longer tied")

        # VALID begins strictly AFTER physical serialization is frozen.
        metrics,checks,origins,records=validate(
            teacher,student,cb59_head,prior_head,context_head)
    del teacher
    robust=all(checks[d]["new_minus_prior_true_next_token_NLL"]<0 and
               checks[d]["new_minus_prior_teacher_KL"]<=0
               for d in VALID_DOMAINS)
    evidence={
        "stage":"v149_PAID_CONTEXT_DEPENDENT_INT4_RANK1",
        "original_complete_paid_CB147_SHA256":CB147_SHA,
        "new_complete_CB149_SHA256":candidate.sha256,
        "new_source_name":physical_path.name,
        "all_source_bytes_and_bpp":report(blob),
        "original_pretrained_all_layers_unchanged":True,
        "new_rank1_output_projection_conditioned_on_hidden":True,
        "only_SOURCE_TRAIN_FIT_used_before_archive_serialized":True,
        "fit_source_origins":fit_origins,
        "source_FIT_504_next_token_teacher_error_metrics":fit_report,
        "new_disjoint_source_TRAIN_VALID":origins,
        "all_3_source_VALID_arms":metrics,
        "unhidden_NLL_KL_top1_contrasts":checks,
        "candidate_improves_both_NLL_and_KL_vs_CB147_prior":robust,
        "raw_matched_per_window_results":records,
        "VALID_tokens_per_domain_per_arm":504,
        "no_independent_official_v89_TEST":True,
        "no_new_QA_gold_used_in_training":True,
        "no_semantic_language_or_quality_claim":True,
        "no_Jessica_LIVE_Render_Supabase_APK_changes":True,
    }
    path=out/"v149_paid_context_rank1_training_valid.json"
    path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":evidence["stage"],"physical":evidence["all_source_bytes_and_bpp"],
        "source_fit":fit_report,"robust_PASS_both_domains":robust,
        "all_source_VALID_contrasts":checks,
        "all_source_VALID_scores":metrics,
        "report":str(path),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("frozen_CB147",type=Path)
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    run(a.frozen_CB147,a.out)
