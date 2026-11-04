"""Three-arm source-only teacher-distilled int8 prior at real <=0.7 bpp.

Train: frozen v76 WikiText+AG News original TRAIN FIT 4x64-token windows.
First write complete immutable physical CB147: CB140 + 49152 signed int8
vocab-bias codes and a float32 step plus CRC/SHA, 0.697... all-in bpp.
Then open NEW disjoint source TRAIN VALID spans, never independent v89 TEST.
The 0.5 output residual gain is fixed BEFORE calibration from v144/145.
Do NOT infer QA competence from token PPL, KL or logit bias calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from time import perf_counter

import torch
from torch import nn
from transformers import AutoTokenizer
from datasets import load_dataset

from binary_geometry_v64 import MODEL,MODEL_SHA
from quality import load_token_ids
from paid_lowrank_v76 import make_model,get_tokens
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,MAX_SOURCE_BYTES
from v140.codec import unpack as load_paid
from v141.run import ORIGINAL_V140_SHA
from v142.scoring import compare_logits,aggregate
from v143.runtime import PackedFusedHead
from .codec import pack,unpack,measured_rate
from .fit import fit
from .runtime import fused_half_head
from .protocol import (FIT_DOMAINS,VALID_DOMAINS,WIKI_REVISION,NEWS_REVISION,
    WIKI_VALID_START,WIKI_VALID_COUNT,NEWS_VALID_START,NEWS_VALID_END,
    VALID_CONTEXT)


def source_valid(tokenizer):
    """Never called until the physical CB147 is frozen and written to disk."""
    wiki=load_token_ids(tokenizer,"train",WIKI_REVISION,
                        WIKI_VALID_START+WIKI_VALID_COUNT)
    wiki=wiki[WIKI_VALID_START:WIKI_VALID_START+WIKI_VALID_COUNT].clone()
    rows=load_dataset("fancyzhx/ag_news",revision=NEWS_REVISION,
                      split=f"train[{NEWS_VALID_START}:{NEWS_VALID_END}]")
    raw="\n\n".join(str(x["text"]) for x in rows)
    news=tokenizer(raw,add_special_tokens=False,return_tensors="pt").input_ids[0,:WIKI_VALID_COUNT].clone()
    if len(wiki)!=WIKI_VALID_COUNT or len(news)!=WIKI_VALID_COUNT:
        raise ValueError("new untouched source VALID unexpectedly short")
    return {"wiki_prior_valid":wiki,"news_prior_valid":news},{
        "wiki_prior_valid":{
            "dataset":"Salesforce/wikitext","split":"train",
            "revision":WIKI_REVISION,"token_span":[WIKI_VALID_START,WIKI_VALID_START+WIKI_VALID_COUNT],
            "token_SHA256":hashlib.sha256(wiki.numpy().tobytes()).hexdigest()},
        "news_prior_valid":{
            "dataset":"fancyzhx/ag_news","split":"train",
            "revision":NEWS_REVISION,"record_span":[NEWS_VALID_START,NEWS_VALID_END],
            "token_SHA256":hashlib.sha256(news.numpy().tobytes()).hexdigest()},
    }


@torch.inference_mode()
def frozen_fit(teacher,student,head):
    """No validation or benchmark questions enter the int8 output prior."""
    teacher_sum=None
    student_sum=None
    count=0
    sources={}
    for domain in FIT_DOMAINS:
        seq,origin=get_tokens(domain)
        if origin["split"]!="train" or len(seq)!=4*VALID_CONTEXT:
            raise ValueError("not original v76 TRAIN FIT source")
        sources[domain]=origin
        for j in range(4):
            x=seq[j*VALID_CONTEXT:(j+1)*VALID_CONTEXT][None,:]
            ref=teacher(input_ids=x,use_cache=False).logits[0,:-1].detach().to(torch.float64)
            got=student(input_ids=x,use_cache=False).logits[0,:-1].detach().to(torch.float64)
            if ref.shape!=got.shape or ref.shape!=(63,49152):
                raise ValueError("original teacher and compact student logits mismatched")
            aa=ref.sum(0)
            bb=got.sum(0)
            teacher_sum=aa if teacher_sum is None else teacher_sum+aa
            student_sum=bb if student_sum is None else student_sum+bb
            count+=len(ref)
    if count!=504:raise ValueError("must use exactly predeclared 504 FIT tokens")
    return (teacher_sum/count).numpy(),(student_sum/count).numpy(),sources,count


@torch.inference_mode()
def evaluate(teacher,model,base_head,half_head,bias_head,tokenizer):
    raw,origins=source_valid(tokenizer)
    domains=("wiki_prior_valid","news_prior_valid")
    arms={"original_CB59":base_head,
          "prior_v144_half_head":half_head,
          "new_CB147_paid_int8_prior":bias_head}
    scores={arm:{domain:[] for domain in domains} for arm in arms}
    try:
        for domain in domains:
            ids=raw[domain]
            for j in range(8):
                x=ids[j*VALID_CONTEXT:(j+1)*VALID_CONTEXT][None,:]
                ref=teacher(input_ids=x,use_cache=False).logits.detach()
                for arm,head in arms.items():
                    model.lm_head=head
                    clock=perf_counter()
                    actual=model(input_ids=x,use_cache=False).logits.detach()
                    duration=perf_counter()-clock
                    result=compare_logits(ref,actual,x)
                    result["cpu_full_context_seconds"]=duration
                    result["validation_window_id"]=j
                    scores[arm][domain].append(result)
    finally:
        model.lm_head=base_head
    compact={
        arm:{domain:aggregate(scores[arm][domain])
             for domain in domains} for arm in arms}
    for arm in arms:
        for domain in domains:
            compact[arm][domain]["native_CPU_prefill_seconds_8x64"]=sum(
                x["cpu_full_context_seconds"] for x in scores[arm][domain])
    checks={}
    for domain in domains:
        base=compact["original_CB59"][domain]
        previous=compact["prior_v144_half_head"][domain]
        newest=compact["new_CB147_paid_int8_prior"][domain]
        checks[domain]={
            "new_minus_v144_half_head_NLL":
                newest["mean_TRUE_next_token_NLL"]-previous["mean_TRUE_next_token_NLL"],
            "new_minus_v144_half_head_KL":
                newest["teacher_to_candidate_KL"]-previous["teacher_to_candidate_KL"],
            "new_minus_CB59_NLL":
                newest["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
            "new_minus_CB59_KL":
                newest["teacher_to_candidate_KL"]-base["teacher_to_candidate_KL"],
            "new_teacher_next_token_argmax_agreement":
                newest["teacher_next_token_argmax_agreement_fraction"],
        }
    return compact,checks,origins


def run(host,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    raw=host.read_bytes()
    paid=load_paid(raw)
    if paid.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("wrong physically paid complete pinned source")
    if hashlib.sha256(paid.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original CB59 changed")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    teacher=original_teacher()
    with TemporaryDirectory(dir=out) as temp:
        root=Path(temp)
        original=root/"real_CB59.cb59"
        original.write_bytes(paid.original)
        model,meta,_=make_model("baseline",SimpleNamespace(original=original,folder=root))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("frozen source accidentally dense")
        base_head=model.lm_head
        half_head=fused_half_head(model,paid,root)
        teacher_mean,student_mean,fit_origins,num_fit=frozen_fit(teacher,model,half_head)
        vocab_code,scale,fit_metrics=fit(teacher_mean,student_mean)
        full_source=pack(raw,vocab_code,scale)
        output=out/"CALIBRI_v147_paid_6bit_tied_plus_int8_logit_prior.cb147"
        output.write_bytes(full_source)
        parsed=unpack(output.read_bytes())
        if parsed.paid.sha256!=paid.sha256:
            raise ValueError("frozen compact model mutated by fit")
        bias_head=PackedFusedHead(half_head.handle,half_head.kernel,parsed.bias)
        compact,contrasts,val_origin=evaluate(teacher,model,base_head,
                                               half_head,bias_head,tokenizer)
    del teacher
    passing=all(
        contrasts[d]["new_minus_v144_half_head_NLL"]<0 and
        contrasts[d]["new_minus_v144_half_head_KL"]<=0
        for d in VALID_DOMAINS)
    report={
        "stage":"v147_PHYSICALLY_PAID_INT8_VOCAB_PRIOR_NEW_TEXT_CONFIRM",
        "frozen_CB140_source_SHA256":paid.sha256,
        "full_new_CB147_SHA256":parsed.sha256,
        "physical_file_name":output.name,
        "complete_source_size_AND_BPP":measured_rate(full_source),
        "v144_selected_precommitted_input_gain":0.0,
        "v144_selected_precommitted_head_gain":0.5,
        "corrective_bias_physical_int8_count":49152,
        "one_predeclared_teacher_fit_algorithm_only":True,
        "FIT_original_predefined_v76_sources":fit_origins,
        "FIT_next_tokens_used":num_fit,
        "FIT_quantization_report":fit_metrics,
        "VALID_NEW_original_train_split_source":val_origin,
        "all_scored_VALID_ground_truth_tokens_per_domain_per_arm":504,
        "no_prior_validation_or_QA_data_used_in_bias_fit":True,
        "no_model_or_hyperparameter_selection_based_on_NEW_VALID":True,
        "no_independent_official_v89_TEST":True,
        "no_live_Jessica_Render_Supabase_APK_change":True,
        "quality_by_arm_and_domain":compact,
        "new_heldout_contrasts":contrasts,
        "new_vocab_prior_improved_NLL_and_KL_both_new_domains":passing,
        "teacher_argmax_agreement_and_semantic_QA_must_be_checked_separately":True,
    }
    path=out/"v147_paid_vocab_prior_new_valid.json"
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "stage":report["stage"],
        "physical":report["complete_source_size_AND_BPP"],
        "gate_pass":passing,
        "contrasts":contrasts,
        "scores":compact,
        "report":str(path)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("pinned_CB140",type=Path)
    ap.add_argument("out",type=Path)
    args=ap.parse_args()
    run(args.pinned_CB140,args.out)
