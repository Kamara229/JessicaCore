"""Read-only comparison of same physical CB140 with naive v140 vs LUT v141 CPU.

No model/weight/bitplane changes, no new fit. Same 16 previously-used prompts.
Actual generated tokens, first-token numeric deviation and CPU time are reported;
only a single source archive physically counted (unlike v119).
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from statistics import mean

import numpy as np
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v108.generation import build_chat
from v94.generation import generate_one
from v131.probe import first_logits
from v132.protocol import all_examples
from v140.codec import unpack,report
from v140.runtime import install as install_v140
from .kernel import install as install_v141

ORIGINAL_V140_SHA="5c19c9052cc40d9b5650a26993f05991522222686c95f8dd280a77b63412f4bc"
ARMS=("original_CB59","old_v140_scalar_onebit","fast_v141_lut_onebit")


def row(model,tok,case):
    pair,side,question,opposite,gold,foil=case
    with torch.inference_mode():
        logits,parts,prompt_tokens=first_logits(model,tok,question)
        generation=generate_one(model,tok,build_chat(tok,question),stop_at_eos=True)
    return {
        "pair":pair,"side":side,"question":question,
        "first_logit_max":float(logits.max()),
        "first_argmax":int(logits.argmax()),
        "first_logit_sample":np.asarray(logits.detach().cpu(),dtype=np.float32),
        "first_EOS":int(logits.argmax())==tok.eos_token_id,
        "generated_token_ids":generation["ids"],
        "text":generation["text"],
        "seconds":generation["elapsed_seconds"],
        "tokens":len(generation["ids"]),
        "eos_generated":generation["eos_generated"],
    }


def run(artifact,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    blob=artifact.read_bytes()
    physical=unpack(blob)
    if physical.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("not the EXACT pinned full valid CB140 source")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases=tuple(all_examples())
    if len(cases)!=16:raise ValueError("frozen counterfactual set changed")

    with TemporaryDirectory(dir=out) as directory:
        root=Path(directory)
        source=root/"cb59_native.bin"
        source.write_bytes(physical.original)
        model,metadata,_=make_model("baseline",SimpleNamespace(original=source,folder=root))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("original dense source is resident")
        original_embedding=model.model.embed_tokens
        original_head=model.lm_head
        install_v140(model,physical,root)
        old_embedding=model.model.embed_tokens
        old_head=model.lm_head
        model.model.embed_tokens=original_embedding
        model.lm_head=original_head
        install_v141(model,physical,root)
        fast_embedding=model.model.embed_tokens
        fast_head=model.lm_head

        operators={
            "original_CB59":(original_embedding,original_head),
            "old_v140_scalar_onebit":(old_embedding,old_head),
            "fast_v141_lut_onebit":(fast_embedding,fast_head),
        }
        rows={k:[] for k in ARMS}
        for idx,case in enumerate(cases):
            # Rotate the order on every prompt to limit systematic thermal bias.
            order=list(ARMS)
            order=order[idx%3:]+order[:idx%3]
            for key in order:
                embedding,head=operators[key]
                model.model.embed_tokens=embedding
                model.lm_head=head
                records=row(model,tok,case)
                rows[key].append(records)

        model.model.embed_tokens=original_embedding
        model.lm_head=original_head

    ref=rows["old_v140_scalar_onebit"]
    opt=rows["fast_v141_lut_onebit"]
    by_arm={}
    for name,values in rows.items():
        elapsed=sum(q["seconds"] for q in values)
        tokens=sum(q["tokens"] for q in values)
        by_arm[name]={
            "CPU_seconds_16_complete_responses":elapsed,
            "total_generated_tokens":tokens,
            "generated_token_per_second":tokens/max(elapsed,1e-8),
            "first_EOS":sum(q["first_EOS"] for q in values),
            "first_argmax_ids":[q["first_argmax"] for q in values],
            "eos_generated":sum(q["eos_generated"] for q in values),
        }
    values=[]
    for a,b in zip(ref,opt):
        if (a["pair"],a["side"],a["question"])!=(b["pair"],b["side"],b["question"]):
            raise ValueError("lost matched prompts")
        values.append({
            "pair":a["pair"],"side":a["side"],
            "max_abs_first_logit_error":float(
                np.max(np.abs(a["first_logit_sample"]-b["first_logit_sample"]))),
            "first_argmax_equal":a["first_argmax"]==b["first_argmax"],
            "all_generated_tokens_identical":a["generated_token_ids"]==b["generated_token_ids"],
        })
    report_object={
        "stage":"v141_SAME_PHYSICAL_CB140_CPU_LUT_PERFORMANCE",
        "complete_original_byte_count":len(blob),
        "pinned_physical_SHA256":physical.sha256,
        "physical":report(blob),
        "NO_model_weight_or_archive_change":True,
        "same_original_16_preused_counterfactuals":True,
        "no_independent_v89_TEST":True,
        "no_Jessica_or_LIVE_changes":True,
        "by_arm":by_arm,
        "relative_v141_vs_CB59_token_speed":
            by_arm["fast_v141_lut_onebit"]["generated_token_per_second"]/
            by_arm["original_CB59"]["generated_token_per_second"],
        "relative_v141_vs_old_v140_token_speed":
            by_arm["fast_v141_lut_onebit"]["generated_token_per_second"]/
            by_arm["old_v140_scalar_onebit"]["generated_token_per_second"],
        "same_numerical_model": {
            "matched_all_16_first_argmax":all(x["first_argmax_equal"] for x in values),
            "matched_all_16_complete_token_sequences":all(
                x["all_generated_tokens_identical"] for x in values),
            "worst_absolute_logit_error":max(x["max_abs_first_logit_error"] for x in values),
            "paired_logit_and_full_answer_checks":values,
        },
        "speed_limit":"Hardware-specific; includes prefill, EOS and greedy decoding; one run is not a latency distribution.",
    }
    path=out/"v141_same_CB140_lut_performance.json"
    path.write_text(json.dumps(report_object,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":report_object["stage"],"by_arm":by_arm,
        "relative_v141_vs_CB59_token_speed":report_object["relative_v141_vs_CB59_token_speed"],
        "relative_v141_vs_old_v140_token_speed":report_object["relative_v141_vs_old_v140_token_speed"],
        "same_numerical_model":report_object["same_numerical_model"],
        "physical_bpp":report_object["physical"]["full_source_bpp"],
        "report":str(path),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("paid_artifact",type=Path)
    parser.add_argument("out",type=Path)
    opts=parser.parse_args()
    run(opts.paid_artifact,opts.out)
