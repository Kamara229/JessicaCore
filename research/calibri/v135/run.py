"""FP32 component oracle ablation: diagnose early compression failure.

This intentionally imports UNCOMPRESSED teacher input/head operators into the
native student's CPU forward pass. Such arms are NOT compact candidates and
must never enter real 0.7-bpp comparisons. No weights trained or serialized.
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
from v134.protocol import corpus
from v134.trace import capture,compare
from .arms import ORACLE_ARMS,BASE_ARMS,PAID_ARMS,oracle_modules,set_paid
from .scoring import summarize


def run(host,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    source=read_v119(host)
    if hashlib.sha256(source.base).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original physical CB59 was modified")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases=corpus()
    inputs=[]
    for name,side,question,opposite,gold,foil in cases:
        ids=tok.encode(build_chat(tok,question),add_special_tokens=False)
        if not 1<len(ids)<160:raise ValueError("unexpected prompt")
        inputs.append(torch.tensor([ids],dtype=torch.long))
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("not original frozen 30-layer teacher")
    with torch.inference_mode():
        references=[capture(teacher,x) for x in inputs]
    result={
        "experiment":"v135 in-memory FP32 teacher input/output ORACLE localization",
        "teacher_model":MODEL,"teacher_revision":MODEL_SHA,
        "host_sha256":hashlib.sha256(host.read_bytes()).hexdigest(),
        "packed_core_bpp_in_NATIVE_arcs_only":8*host.stat().st_size/TOTAL_UNIQUE,
        "diagnostic_oracle_FP32_matrix_bytes_NOT_INCLUDED_in_core_bpp":
            int(teacher.model.embed_tokens.weight.numel()*4+
                teacher.lm_head.weight.numel()*4),
        "never_claim_oracle_arms_are_compressed":True,
        "not_a_compression_candidate":True,
        "no_oracle_answers_in_model_prompts":True,
        "official_v89_TEST_unopened":True,"Jessica_LIVE_unchanged":True,
        "case_names":[[case[0],case[1]] for case in cases],
        "arms":{}
    }
    with TemporaryDirectory(dir=out) as directory:
        root=Path(directory)
        base=root/"native.cb59"
        base.write_bytes(source.base)
        model,meta,_=make_model("baseline",
            SimpleNamespace(original=base,folder=root))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("dense source already present in native model")
        if len(model.model.layers)!=30:
            raise ValueError("unexpected compressed model depth")

        def evaluate():
            rows=[]
            with torch.inference_mode():
                for x,(ref_states,ref_logits) in zip(inputs,references):
                    states,logits=capture(model,x)
                    rows.append(compare(ref_states,states,ref_logits,logits,tok.eos_token_id))
            return rows

        for arm in BASE_ARMS:
            use_input="FP32_input" in arm
            use_head="FP32_head" in arm
            if arm.endswith("and_head"):
                use_head=True
            with oracle_modules(model,teacher,input=use_input,head=use_head):
                result["arms"][arm]=evaluate()

        with installed(model,source,moved=False) as bridge:
            for arm in PAID_ARMS:
                if not set_paid(bridge,arm):
                    raise ValueError("failed to disable paid early layers")
                with oracle_modules(model,teacher,input="FP32_input" in arm):
                    result["arms"][arm]=evaluate()

    result["summary"]=summarize(result["arms"])
    target=out/"v135_FP32_component_oracle.json"
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v135_NONDEPLOYABLE_TEACHER_COMPONENT_ORACLE",
                      "summary":result["summary"],"report":str(target)},
                     ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("host",type=Path)
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    run(a.host,a.out)
