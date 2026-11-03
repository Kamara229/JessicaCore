"""Fresh disjoint original TRAIN and TRAIN-VALID tokens for CB85.

No official TEST source here. v84 used fit windows 0,1 and valid window 0;
v85 uses fit windows 2,3 and valid window 1, frozen by source code.
Teacher target is recorded at the corresponding token from whole original
model; gate input is from the actual original compressed model, not teacher.
"""
from __future__ import annotations
import hashlib
import json
import torch
from paid_lowrank_v76 import get_tokens, WINDOW
from v84.sources import TARGETS


TRAIN_WINDOWS={"wiki_fit":(2,3),"news_fit":(2,3)}
VALID_WINDOWS={"wiki_valid":(1,),"news_valid":(1,)}


def collect(teacher,student,phase:str):
    if phase not in ("train","valid"):
        raise AssertionError("only original TRAIN/VALID allowed")
    definition=TRAIN_WINDOWS if phase=="train" else VALID_WINDOWS
    result={}
    for domain,indices in definition.items():
        ids,provenance=get_tokens(domain)
        if provenance.get("split")!="train":
            raise AssertionError("TEST leaked into operator design")
        probes={p:{"student_input":[],"teacher_output":[]} for p in TARGETS}
        listeners=[]
        try:
            for path in TARGETS:
                t=teacher.get_submodule(path)
                s=student.get_submodule(path)
                if not isinstance(t,torch.nn.Linear):
                    raise AssertionError("missing real pretrained FP32 operator")
                def on_teacher(_m,x,y,p=path):
                    probes[p]["teacher_output"].append(
                        y.detach().reshape(-1,y.shape[-1]).clone())
                def on_packed(_m,x,p=path):
                    probes[p]["student_input"].append(
                        x[0].detach().reshape(-1,x[0].shape[-1]).clone())
                listeners.append(t.register_forward_hook(on_teacher))
                listeners.append(s.register_forward_pre_hook(on_packed))
            windows=[]
            with torch.inference_mode():
                for ix in indices:
                    x=ids[ix*WINDOW:(ix+1)*WINDOW].clone()
                    if x.numel()!=WINDOW:
                        raise AssertionError("insufficient real original source positions")
                    a=teacher(input_ids=x[None],use_cache=False).logits[0]
                    student(input_ids=x[None],use_cache=False)
                    windows.append({"tokens":x,"token_SHA":hashlib.sha256(
                        x.numpy().tobytes()).hexdigest(),
                        "index":ix,"teacher_logits": a.detach().clone() if
                        phase=="valid" else None})
            for path in TARGETS:
                if any(len(probes[path][name])!=len(indices)
                       for name in ("student_input","teacher_output")):
                    raise AssertionError("original vs compressed operator hooks incomplete")
        finally:
            for h in listeners:
                h.remove()
        result[domain]={"source":provenance,"windows":windows,"probes":probes}
        print(json.dumps({"stage":"v85_genuine_original_and_native_source",
                          "phase":phase,"domain":domain,"windows":list(indices)}),
                          flush=True)
    return result


def paired_model_activations(data:dict,path:str):
    x,y=[],[]
    for domain,part in data.items():
        probes=part["probes"][path]
        for original_input,teacher_output in zip(
            probes["student_input"],probes["teacher_output"]):
            # The hook ran under inference_mode: clone to ordinary
            # Tensor outside it for autograd on the small physical gate.
            x.append(original_input.detach().clone())
            y.append(teacher_output.detach().clone())
    if not x:
        raise AssertionError("no actual compressed model input trajectories")
    return torch.cat(x,dim=0),torch.cat(y,dim=0)
