"""Measured FP32/native packed layer-state divergence after first-band EOS finding.

Read-only teacher forward pass; same prompt and tokenizer for each arm. No gold
answer enters a forward pass, and no coefficients, tensors or archive bytes are
trained, optimized, mutated or persisted to the baseline.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v108.generation import build_chat
from v117.protocol import ORIGINAL_CB59_SHA, TOTAL_UNIQUE
from v128.protocol import read_v119
from v129.operators import installed
from v129.run import load_green
from .protocol import ARMS, corpus, apply
from .trace import capture, compare, aggregate


def run(host, green, out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    packed=read_v119(host)
    if hashlib.sha256(packed.base).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("unmatched CB59 physical archive")
    controller=load_green(green)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases=corpus()
    tokens=[]
    for name,side,question,opposite,gold,foil in cases:
        prompt=build_chat(tokenizer,question)
        ids=tokenizer.encode(prompt,add_special_tokens=False)
        if not 1 < len(ids) < 160: raise ValueError("invalid paired prompt")
        tokens.append(torch.tensor([ids],dtype=torch.long))

    teacher=original_teacher()
    try:
        if len(teacher.model.layers)!=30: raise ValueError("teacher depth")
        teacher_probes=[capture(teacher,x) for x in tokens]
    finally:
        del teacher

    result={
        "stage":"v134_FROZEN_FIRST_HIDDEN_DIVERGENCE",
        "source_model":MODEL,
        "frozen_host_sha256":hashlib.sha256(host.read_bytes()).hexdigest(),
        "frozen_green_sha256":hashlib.sha256(green.read_bytes()).hexdigest(),
        "physical_host_bpp_excluding_controller":8*host.stat().st_size/TOTAL_UNIQUE,
        "controller_bytes":green.stat().st_size,
        "teacher_answer_is_NOT_a_training_signal":True,
        "locked_official_v89_TEST_untouched":True,
        "no_Jessica_test_or_PRODUCTION_change":True,
        "cases":[{"pair":pair,"side":side,"question":q}
                 for pair,side,q,opposite,gold,foil in cases],
        "arms":{},
    }

    with TemporaryDirectory(dir=out) as scratch:
        root=Path(scratch)
        original=root/"native.cb59"
        original.write_bytes(packed.base)
        model,meta,_=make_model("baseline",SimpleNamespace(original=original,folder=root))
        if not meta["zero_dense_pretrained_2D_hot_weights"] or len(model.model.layers)!=30:
            raise ValueError("not a 30-layer direct-native CB59 runtime")
        def compute_rows():
            rows=[]
            for x,(hidden,logits) in zip(tokens,teacher_probes):
                actual,actual_logits=capture(model,x)
                rows.append(compare(hidden,actual,logits,actual_logits,tokenizer.eos_token_id))
            return rows
        result["arms"]["CB59"]=compute_rows()
        with installed(model,packed,moved=False) as bridge:
            for arm in ARMS[2:]:
                rows=[]
                for (pair,side,question,opposite,gold,foil),x,(hidden,logits) in zip(
                        cases,tokens,teacher_probes):
                    apply(bridge,controller,question,arm)
                    current,actual_logits=capture(model,x)
                    rows.append(compare(hidden,current,logits,actual_logits,tokenizer.eos_token_id))
                result["arms"][arm]=rows

    result["summary"]=aggregate(result["arms"])
    path=out/"v134_first_hidden_divergence.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    compact={a:{k:v for k,v in vals.items() if k!="mean_relative_L2_by_layer"}
             for a,vals in result["summary"]["arms"].items()}
    print(json.dumps({"stage":result["stage"],"summary":compact,"report":str(path)},
                     ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("host",type=Path)
    parser.add_argument("green",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    run(args.host,args.green,args.out)
