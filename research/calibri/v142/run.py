"""True NLL / PPL on two original TRAIN/VALID corpora, non-v89 TEST.

Frozen full FP32 + CB59 + exact CB140 physical SHA; native v141 packed
input/output residual. Inputs are unchanged 64-token WikiText and AG News
TRAIN split, positions pinned in v76, never independent TEST. No training.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from time import perf_counter

import torch
from torch.nn import functional as F

from paid_lowrank_v76 import make_model,get_tokens
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from v140.codec import unpack,report
from v141.kernel import install
from .protocol import DOMAINS,WINDOWS_PER_DOMAIN,WINDOW,ARMS,ORIGINAL_V140_SHA,pairs
from .scoring import compare_logits,aggregate,contrast


def run(physical_path,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    packed=physical_path.read_bytes()
    physical=unpack(packed)
    if physical.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("not frozen completed full CB140 physical source")
    if hashlib.sha256(physical.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("CB59 physical source changed")

    tokens,origins={},{}
    for domain in DOMAINS:
        raw,origin=get_tokens(domain)
        if origin.get("split")!="train" or len(raw)<WINDOWS_PER_DOMAIN*WINDOW:
            raise ValueError("not authorized source TRAIN VALID")
        tokens[domain]=pairs(raw)
        origins[domain]=origin
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("wrong FP32 teacher geometry")
    teacher_logits={}
    try:
        with torch.inference_mode():
            for domain in DOMAINS:
                teacher_logits[domain]=[teacher(input_ids=x,use_cache=False).logits.detach().clone()
                                       for x in tokens[domain]]
    finally:
        del teacher
    data={
        "stage":"v142_FROZEN_VALID_TRUE_NLL_KL",
        "original_full_CB140_sha256":physical.sha256,
        "native_complete_physical_bytes":report(packed)["complete_bytes"],
        "native_bpp_original_unique":8*physical.size/TOTAL_UNIQUE,
        "no_new_trained_or_repacked_weights":True,
        "FP32_only_for_teacher_EVALUATION_not_for_fitting":True,
        "train_split_VALID_windows_only":True,
        "no_independent_v89_TEST":True,
        "no_semantic_QA_proof":True,
        "no_Jessica_LIVE_changes":True,
        "validation_source_origins":origins,
        "validation_windows_each":WINDOWS_PER_DOMAIN,
        "context_tokens_each":WINDOW,
        "arms":{arm:{} for arm in ARMS},
    }
    with TemporaryDirectory(dir=out) as directory:
        root=Path(directory)
        original=root/"source.cb59"
        original.write_bytes(physical.original)
        model,meta,_=make_model("baseline",SimpleNamespace(original=original,folder=root))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("packed source changed to dense model")
        before_input,before_head=model.model.embed_tokens,model.lm_head
        kernel=install(model,physical,root)
        after_input,after_head=model.model.embed_tokens,model.lm_head
        if kernel.rows!=49152:
            raise ValueError("invalid tied residual geometry")
        for domain in DOMAINS:
            data["arms"]["FP32"][domain]=[]
            data["arms"]["CB59"][domain]=[]
            data["arms"]["CB140_v141_LUT"][domain]=[]
            for i,x in enumerate(tokens[domain]):
                ref=teacher_logits[domain][i]
                # Repeat same immutable input for both CPU packed arms.
                for key,(embedding,head) in (
                    ("CB59",(before_input,before_head)),
                    ("CB140_v141_LUT",(after_input,after_head))):
                    model.model.embed_tokens=embedding
                    model.lm_head=head
                    with torch.inference_mode():
                        t0=perf_counter()
                        logits=model(input_ids=x,use_cache=False).logits
                        wall=perf_counter()-t0
                        row=compare_logits(ref,logits,x)
                    row["cpu_prefill_seconds"]=wall
                    row["window_index"]=i
                    data["arms"][key][domain].append(row)
                data["arms"]["FP32"][domain].append(
                    {**compare_logits(ref,ref,x),"window_index":i})
        model.model.embed_tokens=before_input
        model.lm_head=before_head

    results=contrast(data["arms"]["CB59"],data["arms"]["CB140_v141_LUT"])
    data["scores"]=results
    for domain in DOMAINS:
        data["scores"][domain]["FP32"]=aggregate(data["arms"]["FP32"][domain])
        for key in ("CB59","CB140_v141_LUT"):
            records=data["arms"][key][domain]
            secs=sum(x["cpu_prefill_seconds"] for x in records)
            data["scores"][domain][key+"_actual_prefill_seconds"]=secs
            data["scores"][domain][key+"_real_input_tokens_per_second"]=(WINDOW*len(records))/max(secs,1e-9)
    target=out/"v142_two_domain_VALID_NLL_KL.json"
    target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":data["stage"],"sources":[origins[d]["dataset"] for d in DOMAINS],
                      "scores":results,"physical_bpp":data["native_bpp_original_unique"],
                      "report":str(target)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("physical_cb140",type=Path)
    a.add_argument("out",type=Path)
    p=a.parse_args()
    run(p.physical_cb140,p.out)
