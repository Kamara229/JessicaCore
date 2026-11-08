"""CALIBRI S27: independent *post-hoc* audit, no inference or training.

Inputs are preserved Actions outputs from S26's predeclared baseline S24,
one final S26 checkpoint, the independent public 135M FP32 model, paid S26
training trace and native C CPU speed. Never use the QA answers to train,
tune, choose a checkpoint, alter codebooks, stop learning or change S26.

The pretrained 135M FP32 is explicitly an UNMATCHED architecture/weights
reference, NOT proof of preserved FP32 quality after compression.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

QA_SHA="e8e3e75c5fedf845aef512a104723a4d04fab3980ecec7f4deb3575ae020ac3f"
SEALED_S24_SHA="b21619b12ff6014b63129916857a525455960852c8f5a530de233acf10e8f900"
PARAMS=36_380_480
EXPECTED_BYTES=3_141_513
PRETRAINED_135M_PARAMS=134_515_008
NOVEL="reserved_novel_color_noun"
DOMAINS={NOVEL:96,"color_one_word_control":24,"number_one_word_control":24}

def read_json(path):
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj,dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return obj

def ensure_rows(obj, name):
    got=obj.get("every_144_actual_answer_and_gold")
    if not isinstance(got,list) or len(got)!=144:
        raise AssertionError(f"{name}: exactly 144 archived rows required")
    rows={}
    domain_counts=Counter()
    for r in got:
        if not isinstance(r,dict):
            raise AssertionError(f"{name}: malformed row")
        key=r.get("id")
        if not isinstance(key,str) or key in rows:
            raise AssertionError(f"{name}: missing or duplicate ID")
        for k in ("domain","question","gold","strict_full_EOS",
                  "actual_token_ids","actual_text"):
            if k not in r:raise AssertionError(f"{name}/{key}: missing {k}")
        if not isinstance(r["strict_full_EOS"],bool):
            raise AssertionError(f"{name}/{key}: strict full answer not boolean")
        if not isinstance(r["actual_token_ids"],list):
            raise AssertionError(f"{name}/{key}: missing generated tokens")
        if not isinstance(r["actual_text"],str):
            raise AssertionError(f"{name}/{key}: missing generated text")
        rows[key]=r
        domain_counts[r["domain"]]+=1
    if dict(domain_counts)!=DOMAINS:
        raise AssertionError(f"{name}: QA domains changed: {domain_counts}")
    opened=obj.get("every_16_open_actual_answer")
    if not isinstance(opened,list) or len(opened)!=16 or any(
        not isinstance(r,dict) or not isinstance(r.get("actual_answer"),str)
        for r in opened):
        raise AssertionError(f"{name}: all 16 unmodified open outputs required")
    return rows,opened

def binomial_wilson(k,n,z=1.959963984540054):
    if not (0<=k<=n and n):raise AssertionError("invalid probability")
    p=k/n
    d=1+z*z/n
    mid=(p+z*z/(2*n))/d
    spread=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.0,mid-spread),min(1.0,mid+spread)]

def exact_paired_p_value(before,after):
    """Two-sided exact McNemar test for *one predeclared* paired question set."""
    gained=sum(not b and a for b,a in zip(before,after))
    lost=sum(b and not a for b,a in zip(before,after))
    n=gained+lost
    if not n:return {"gained":0,"lost":0,"p_two_sided":1.0}
    small=min(gained,lost)
    p=min(1.0,2.0*sum(math.comb(n,k) for k in range(small+1))/(2**n))
    return {"gained":gained,"lost":lost,"p_two_sided":p}

def audit_rows(before,after,external):
    first,oldopen=ensure_rows(before,"sealed_S24_before_S26")
    second,newopen=ensure_rows(after,"single_final_S26")
    pretrained,fpopen=ensure_rows(external,"unmatched_135m_pretrained_FP32")
    if set(first)!=set(second) or set(first)!=set(pretrained):
        raise AssertionError("Same 144 exact precommitted IDs required")
    for k in first:
        a,b,c=first[k],second[k],pretrained[k]
        for other in (b,c):
            for field in ("question","gold","domain","expected_token_ids_full_EOS"):
                if a.get(field)!=other.get(field):
                    raise AssertionError(f"QA content was changed: {k}/{field}")
    result={}
    for domain,count in DOMAINS.items():
        ordered=sorted(k for k,r in first.items() if r["domain"]==domain)
        before_bits=[first[k]["strict_full_EOS"] for k in ordered]
        after_bits=[second[k]["strict_full_EOS"] for k in ordered]
        public_bits=[pretrained[k]["strict_full_EOS"] for k in ordered]
        bb=sum(before_bits);aa=sum(after_bits);pp=sum(public_bits)
        result[domain]={
            "n":count,
            "S24_complete_correct":bb,
            "S26_complete_correct":aa,
            "PUBLIC_UNMATCHED_135M_FP32_complete_correct":pp,
            "S26_minus_S24":aa-bb,
            "S24_wilson95":binomial_wilson(bb,count),
            "S26_wilson95":binomial_wilson(aa,count),
            "paired_change_S24_to_S26":exact_paired_p_value(before_bits,after_bits),
            "S26_gained_ids":[k for k in ordered if not first[k]["strict_full_EOS"] and second[k]["strict_full_EOS"]],
            "S26_lost_ids":[k for k in ordered if first[k]["strict_full_EOS"] and not second[k]["strict_full_EOS"]],
        }
        if domain==NOVEL:
            for name,rows in (("S24",first),("S26",second),("public_135M_FP32",pretrained)):
                d=result[domain]
                d[name+"_valid_2word"]=sum(
                    bool(rows[k]["actual_text"].strip().lower().split().__len__()==2)
                    for k in ordered)
                # More stringent lexical correctness counts come from the
                # actual archived per-domain scorer, not the split heuristic.
                d[name+"_correct_first_token"]=sum(bool(rows[k]["correct_first_token"]) for k in ordered)
    result["open_response_pairs"]=[
        {"question":oldopen[i].get("question"),
         "S24":oldopen[i]["actual_answer"],
         "S26":newopen[i]["actual_answer"],
         "PUBLIC_UNMATCHED_135M_FP32":fpopen[i]["actual_answer"],
         "requires_human_coherence_review":True}
        for i in range(16)]
    if any(q["question"]!=newopen[i].get("question") or
           q["question"]!=fpopen[i].get("question")
           for i,q in enumerate(result["open_response_pairs"])):
        raise AssertionError("Open questions changed or reordered")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s24",type=Path,required=True)
    p.add_argument("--s26",type=Path,required=True)
    p.add_argument("--pretrained-fp32",type=Path,required=True)
    p.add_argument("--s26-train",type=Path,required=True)
    p.add_argument("--s26-native-speed",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    before=read_json(a.s24)
    after=read_json(a.s26)
    ext=read_json(a.pretrained_fp32)
    train=read_json(a.s26_train)
    speed=read_json(a.s26_native_speed)
    if (before.get("frozen_suite_sha256")!=QA_SHA or
        after.get("frozen_suite_sha256")!=QA_SHA or
        ext.get("S26_precommitted_frozen_qa_SHA256")!=QA_SHA):
        raise AssertionError("One fixed SHA for all THREE independent QA runs")
    if before.get("physical_archive_SHA256")!=SEALED_S24_SHA:
        raise AssertionError("S24 immutable source SHA mismatch")
    if train.get("starting_sealed_S20_sha256")!=SEALED_S24_SHA:
        raise AssertionError("S26 training source is not sealed S24")
    if train.get("final_SHA256")!=after.get("physical_archive_SHA256"):
        raise AssertionError("S26 QA archive != the single saved final paid checkpoint")
    if (before.get("unique_params")!=PARAMS or after.get("unique_params")!=PARAMS or
        train.get("original_params")!=PARAMS or
        before.get("complete_bytes")!=EXPECTED_BYTES or
        after.get("complete_bytes")!=EXPECTED_BYTES or
        train.get("fully_paid_archive_bytes")!=EXPECTED_BYTES):
        raise AssertionError("Changed compressed physical parameter count / byte cost")
    physical_bpp=8*EXPECTED_BYTES/PARAMS
    if any(abs(x-physical_bpp)>1e-10 for x in [
        before["physical_bpp"],after["physical_bpp"],train["complete_bpp"],
        speed["physical_bpp"]
    ]):raise AssertionError("Physical rate not identical across all four artifacts")
    if physical_bpp>.7:raise AssertionError("Rate exceeds 0.7bpp")
    if (ext.get("actual_model_params")!=PRETRAINED_135M_PARAMS or
        ext.get("actual_weight_dtype")!="float32" or
        ext.get("same_architecture_or_trained_initial_weights_as_36M") is not False or
        ext.get("is_not_an_FP32_teacher_for_S26") is not True):
        raise AssertionError("FP32 reference was misidentified as matched 36M")
    if (speed.get("stage")!="S26_NATIVE_PAIRED_CPU" or
        speed.get("s3_sha")!=SEALED_S24_SHA or
        speed.get("s4_sha")!=after["physical_archive_SHA256"] or
        speed.get("tokens_per_trial")!=8 or
        speed.get("no_dense_2D_inference_weight") is not True):
        raise AssertionError("Not same-architecture native S24-v-S26 timing")
    native_speed_ratio=float(speed["median_S26_native_speed_relative_to_S24"])
    if not math.isfinite(native_speed_ratio):
        raise AssertionError("Nonfinite CPU speed ratio")
    domains=audit_rows(before,after,ext)
    report={
        "stage":"S27_UNSEEN_S26_POST_HOC_QA_ALL_PAID_PHYSICAL_AND_NATIVE",
        "frozen_QA_SHA256":QA_SHA,
        "single_posttraining_S26_checkpoint":train["final_SHA256"],
        "source_S24_SHA256":SEALED_S24_SHA,
        "actual_complete_bpp":physical_bpp,
        "within_0_7_bpp":physical_bpp<=.7,
        "native_S26_relative_to_S24_ratio":native_speed_ratio,
        "native_meets_90pct":native_speed_ratio>=.9,
        "native_S26_at_least_90pct_field_crosscheck":
            speed.get("meets_90_percent")== (native_speed_ratio>=.9),
        "S26_accepted_paid_FP16_train_steps":train["accepted_FP16_steps"],
        "S26_changed_paid_5bit_groups":train["updated_paid_5bit_index_groups"],
        "S26_previous_S24_TRAIN_answers_lost":train.get("lost_previous_S24_TRAIN_answers"),
        "S26_previous_S20_TRAIN_answers_lost":train.get("lost_previous_multiword_TRAIN_answers"),
        "S26_original_old_TRAIN_answers_lost":train.get("lost_old_TRAIN_answers"),
        "domain_results":{k:v for k,v in domains.items() if k!="open_response_pairs"},
        "open_raw_answers_for_human_review":domains["open_response_pairs"],
        "different_model_pretrained_FP32_reference_is_not_compression_quality_parity":True,
        "QA_unseen_before_training_and_not_used_for_checkpoint_selection":True,
        "negative_results_preserved":True,
    }
    if not report["native_S26_at_least_90pct_field_crosscheck"]:
        raise AssertionError("Native CPU report contradicted its pass threshold")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
        "stage":report["stage"],
        "bpp":physical_bpp,"native_speed_ratio":native_speed_ratio,
        "novel":{k:v for k,v in report["domain_results"][NOVEL].items()
                 if k in ("S24_complete_correct","S26_complete_correct",
                          "PUBLIC_UNMATCHED_135M_FP32_complete_correct",
                          "paired_change_S24_to_S26")},
        "negative_results_preserved":True
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
