"""CALIBRI v221: read-only TRAIN first-token margin diagnosis of v220 regression.

Compare original SmolLM2 135M FP32, sealed v206 and sealed v220 on 48
opposing public TRAIN prompts. This does NOT fit, score, alter, or select
a source; it never opens public DEV/QA, VALID or private TEST. The native
packed QA and CPU scores already measured in v220 remain separate.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from v84.sources import original_teacher
from v174.run import save
from v176.operators import clear
from v196 import protocol as P
from v196.data import training_inputs
from v196.recover import build_trainable_from_paid_source
from v212.run import SOURCE_V206_SHA256

V220_SOURCE_SHA256="e7541169b5b2188393859953210f52a38ca24451129e246eebb20cb20466e34d"


@torch.inference_mode()
def top_one_probe(model,sample,tokenizer):
    prompt=sample["prompt_ids"].detach().clone().long()[None]
    target=int(sample["first_expected_id"])
    if len(prompt[0])<3 or target<0 or target>=P.VOCAB_SIZE:
        raise AssertionError("TRAIN prompt / next-token target is invalid")
    logits=model(input_ids=prompt,use_cache=False).logits[0,-1].float()
    if logits.shape!=(P.VOCAB_SIZE,) or not torch.isfinite(logits).all():
        raise AssertionError("Incomplete/non-finite actual 49152-vocabulary logits")
    top=torch.topk(logits,5)
    actual=int(top.indices[0])
    competitor=float(top.values[0] if actual!=target else top.values[1])
    margin=float(logits[target])-competitor
    eos=tokenizer.eos_token_id
    return {"first_token_id":actual, "first_token_text":tokenizer.decode([actual]),
            "expected_first_id":target, "expected_first_text":tokenizer.decode([target]),
            "is_expected_first":actual==target,
            "gold_first_margin_vs_best_other":round(margin,6),
            "gold_first_logit":round(float(logits[target]),6),
            "top5":[{"token_id":int(i),"text":tokenizer.decode([int(i)]),
                     "logit":round(float(v),6)}
                    for i,v in zip(top.indices,top.values)],
            "eos_id":int(eos), "eos_logit":round(float(logits[eos]),6)}


def summarize(rows):
    name_order=("FP32","CB206","GT220")
    by_source={}
    for name in name_order:
        values=[r["probes"][name] for r in rows]
        correct=[(r["pair_id"],r["side"]) for r in rows
                 if r["probes"][name]["is_expected_first"]]
        by_source[name]={
            "gold_first_correct_out_of_48":len(correct),
            "correct_ids":correct,
            "unique_predicted_first_ids":len({p["first_token_id"] for p in values}),
            "mean_gold_first_margin":sum(p["gold_first_margin_vs_best_other"]
                                          for p in values)/len(values),
            "per_domain_gold_first_correct":dict(Counter(r["domain"] for r in rows
                if r["probes"][name]["is_expected_first"]))}
    lost=[{"pair_id":r["pair_id"],"side":r["side"],"domain":r["domain"],
           "v206_gold_margin":r["probes"]["CB206"]["gold_first_margin_vs_best_other"],
           "v220_gold_margin":r["probes"]["GT220"]["gold_first_margin_vs_best_other"],
           "v206_first":r["probes"]["CB206"]["first_token_text"],
           "v220_first":r["probes"]["GT220"]["first_token_text"]}
          for r in rows if r["probes"]["CB206"]["is_expected_first"]
          and not r["probes"]["GT220"]["is_expected_first"]]
    gained=[(r["pair_id"],r["side"]) for r in rows
            if r["probes"]["GT220"]["is_expected_first"]
            and not r["probes"]["CB206"]["is_expected_first"]]
    choice_drift=[{"pair_id":r["pair_id"],"side":r["side"],"domain":r["domain"],
                   "original_FP32_first":r["probes"]["FP32"]["first_token_text"],
                   "v206_first":r["probes"]["CB206"]["first_token_text"],
                   "v220_first":r["probes"]["GT220"]["first_token_text"]}
                  for r in rows if r["probes"]["CB206"]["first_token_id"]!=
                                   r["probes"]["GT220"]["first_token_id"]]
    return {"by_source":by_source,"lost_v206_correct_TRAIN_first":lost,
            "gained_v220_correct_TRAIN_first":gained,
            "different_first_tokens_v206_to_v220":choice_drift,
            "first_token_changes":len(choice_drift),
            "important":("TRAIN-only frozen diagnostics; even FP32 is not "
                         "guaranteed gold at every first token. Public DEV "
                         "scores are reported separately, never used to "
                         "select or train this source.")}


def execute(source_v206,source_v220,out):
    torch.manual_seed(221)
    torch.set_num_threads(2)
    data={}
    for label,path,expected in (
            ("CB206",source_v206,SOURCE_V206_SHA256),
            ("GT220",source_v220,V220_SOURCE_SHA256)):
        raw=Path(path).read_bytes()
        if len(raw)!=P.SOURCE_BYTES or sha256(raw).hexdigest()!=expected:
            raise AssertionError("Not the original sealed "+label+" physical source")
        data[label]=CB59(raw)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    samples,train_pairs,valid_pairs,origin=training_inputs(tokenizer)
    if len(samples)!=20 or len(train_pairs)!=24 or len(valid_pairs)!=P.VALID_PAIRS:
        raise AssertionError("Original public TRAIN/VALID protocol drift")
    teacher=original_teacher().cpu().eval()
    source_models={}
    source_centroids={}
    for key in ("CB206","GT220"):
        model,centroids=build_trainable_from_paid_source(data[key])
        if len(centroids)!=211:
            raise AssertionError("Missing authentic 211 source matrices")
        model.eval()
        source_models[key]=model
        source_centroids[key]=centroids
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    save(output/"v221_PRECOMMITTED_TRAIN_FIRST_TOKEN_PROTOCOL.json",{
        "experiment":"read_only_frozen_TRAIN_48_first_token_logits",
        "reference_FP32":MODEL,"reference_revision":MODEL_SHA,
        "v206_SHA256":SOURCE_V206_SHA256,"v220_SHA256":V220_SOURCE_SHA256,
        "physically_paid_source_bytes":P.SOURCE_BYTES,
        "expected_paired_TRAIN_prompts":48,"v220_public_DEV16_not_opened":True,
        "no_training_no_fitting_no_changes_to_source":True,
        "no_Jessica_main_LIVE_Render_Supabase_APK_changes":True})
    rows=[]
    for pair in train_pairs:
        if pair["split"]!="TRAIN":raise AssertionError("VALID leaked to TRAIN")
        for side in ("left","right"):
            sample=pair[side]
            probes={"FP32":top_one_probe(teacher,sample,tokenizer)}
            for label in ("CB206","GT220"):
                clear(source_centroids[label])
                probes[label]=top_one_probe(source_models[label],sample,tokenizer)
                clear(source_centroids[label])
            correct_expected=int(sample["first_expected_id"])
            if any(probe["expected_first_id"]!=correct_expected for probe in probes.values()):
                raise AssertionError("Different next-token label between compared sources")
            rows.append({"split":"TRAIN","pair_id":pair["pair_id"],
                         "side":side,"domain":pair["domain"],
                         "prompt_len":len(sample["prompt_ids"]),
                         "probes":probes})
        print(json.dumps({"stage":"v221_TRAIN_first_token_diagnostics",
                          "pairs_completed":len(rows)//2}),flush=True)
    if len(rows)!=48 or len({(r["pair_id"],r["side"]) for r in rows})!=48:
        raise AssertionError("Missing 48 disjoint original TRAIN prompts")
    result={"completed":True,"rows":rows,"comparison":summarize(rows),
            "v220_is_not_a_quality_improvement":"2/16 public DEV vs v206 3/16",
            "TRAIN_only":True,"no_DEV_VALID_PRIVATE_TEST":True,
            "source_is_read_only":True,"Jessica_tests_ran":False}
    save(output/"v221_ACTUAL_FP32_v206_v220_ALL_TRAIN_48_MARGIN_AUDIT.json",result)
    print(json.dumps({"stage":"v221_ACTUAL_TRAIN_first_token_SUMMARY",
          "gold_counts":{k:v["gold_first_correct_out_of_48"] for k,v
                         in result["comparison"]["by_source"].items()},
          "lost":len(result["comparison"]["lost_v206_correct_TRAIN_first"]),
          "gained":len(result["comparison"]["gained_v220_correct_TRAIN_first"]),
          "changed_first":result["comparison"]["first_token_changes"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source_v206",type=Path)
    parser.add_argument("source_v220",type=Path)
    parser.add_argument("output",type=Path)
    a=parser.parse_args()
    execute(a.source_v206,a.source_v220,a.output)
