"""F21: end-to-end native generation audit with last-only packed tied head.
Reuse the tested F19 C kernel, no new weights, no protected J7 training.
Preserve all original causal hooks; compare actual generated sequences and
same-host time against the existing F10 full-prefix research forward.
"""
import argparse
import inspect
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from research.calibri2.fusion_growth.f01_physical_budget import digest
from research.calibri2.fusion_growth.f03_rate_train_core import decode
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f06_integrated_recovery import forward,scoped_reduced_native_head_gain
from research.calibri2.fusion_growth.f17_packed_cpu import PARENT,data
from research.calibri2.fusion_growth.f19_cpu_row_kernel import build_v2
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import single_pass_early_intervention
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import temporary_learned_native_head
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import strict_chat_source_ids
from calibri2.s160 import real_parallel_slots_j7 as s160
from v176.operators import clear

PROBES=(
 "Назови число 28.", "Повтори слово берёза.", "Сколько будет 13 плюс 9?",
 "Что больше: 41 или 18?", "У Иры нет зонта. Есть ли у неё зонт?",
 "Сначала был дождь, затем снег. Что было первым?",
 "Ключ находится в сумке. Где ключ?", "Повтори слова синее небо.",
 "Repeat the word stone.", "What is 45 minus 12?",
 "Which comes first, morning or evening?", "Write only the number 71.")
MODES=("FULL_PREFIX","LAST_ONLY","PACKED_LAST")
CAP=8

def keep_keyword(model):
    params=inspect.signature(model.forward).parameters
    for k in ("num_logits_to_keep","logits_to_keep"):
        if k in params:return k
    raise RuntimeError("Pinned causal LM does not expose last-logits-only ABI")

def last_forward(state,p,features,key):
    model,sources,bridge,full,expansion,child,_=state
    ids=torch.tensor([p],dtype=torch.long)
    with torch.enable_grad(),scoped_reduced_native_head_gain(child.language_head),          temporary_learned_native_head(model,child.language_head) as audit,          expansion.prefill_only(model),s160.prefill_only_tunnel(model,full),          bridge.prompt_scope("S93",features),          single_pass_early_intervention(model,child,contextual=True) as witness:
        clear(sources)
        try:
            out=model(input_ids=ids,past_key_values=None,use_cache=True,
                      output_hidden_states=False,**{key:1})
            assert out.logits.shape==(1,1,49152) and out.past_key_values is not None
            z=out.logits[:,-1,:].float().detach().clone()
        finally:clear(sources)
    assert audit["original_head_calls"]==1 and witness["candidate_forward_calls"]==1
    assert full.last_forward_coverage==tuple(range(12)) and torch.isfinite(z).all()
    return z

def install_packed(state,archive,f):
    model,sources=state[:2]
    src=sources["model.embed_tokens.weight"]
    assert model.model.embed_tokens.source is src and model.lm_head.source is src
    h=archive.handles["model.embed_tokens.weight"]
    p,c,s,_=data(h,16);rows,cols=h.shape
    counts={"packed_head":0,"gather_embedding":0}
    def packed_head(self,x):
        assert x.shape==(1,1,cols)
        v=np.ascontiguousarray(x.detach().cpu().numpy().reshape(cols),dtype=np.float32)
        y=np.empty(rows,np.float32)
        rc=f(p,len(p),c,s,v,y,rows,cols,4)
        if rc:raise RuntimeError(f"F19 kernel status {rc}")
        counts["packed_head"]+=1
        return torch.from_numpy(y).view(1,1,rows)
    def gather_embedding(self,ids):
        # No dense tied matrix allocation: reconstruct only requested rows.
        centers,scales=src.fields()
        indices=src.codes.reshape(rows,cols//8)[ids]
        result=centers[indices].reshape(*ids.shape,cols)*scales[ids].unsqueeze(-1)
        counts["gather_embedding"]+=1
        return result
    model.lm_head.forward=MethodType(packed_head,model.lm_head)
    model.model.embed_tokens.forward=MethodType(gather_embedding,model.model.embed_tokens)
    # The diagnostic source still holds its original integer code buffer.
    # No assertion of overall resident-memory reduction is made.
    return counts

def generate(state,p,f,mode,key):
    seq=[];logits=[];start=time.perf_counter()
    for _ in range(CAP):
        z=forward(state,p+seq,f).detach() if mode=="FULL_PREFIX" else last_forward(state,p+seq,f,key)
        logits.append(z.clone())
        seq.append(int(z.argmax()))
        if seq[-1]==2:break
    return seq,logits,time.perf_counter()-start

def run(f10,f08,out):
    torch.set_num_threads(2);torch.manual_seed(20261209)
    raw=(f10/"F10_FP16.bundle").read_bytes();assert digest(raw)==PARENT
    archive,side=decode(raw)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    assert tok.vocab_size==49152 and tok.eos_token_id==2
    out.mkdir(parents=True,exist_ok=False)
    records=[];results={}
    with tempfile.TemporaryDirectory() as tmp:
        lib,f=build_v2(tmp)
        states={m:direct(archive,side) for m in MODES}
        keys={m:keep_keyword(states[m][0]) for m in MODES}
        counters=install_packed(states["PACKED_LAST"],archive,f)
        # Warm all three modes with identical neutral prompt.
        q="Повтори слово лист.";p=strict_chat_source_ids(tok,q);feat=prompt_features(tok,q)
        for m in MODES:generate(states[m],p,feat,m,keys[m])
        for i,q in enumerate(PROBES):
            p=strict_chat_source_ids(tok,q);feat=prompt_features(tok,q)
            order=MODES[i%3:]+MODES[:i%3];allruns={}
            for m in order:
                allruns[m]=generate(states[m],p,feat,m,keys[m])
            seq,z,t=allruns["FULL_PREFIX"]
            row={"prompt_sha256":digest(q.encode()),"baseline_ids":seq,"baseline_seconds":t,"trials":{}}
            for m in MODES[1:]:
                own,ls,dt=allruns[m]
                # Compare logits only while prefixes actually match.
                n=0;err=0.;same=0
                for j,(a,b) in enumerate(zip(z,ls)):
                    if seq[:j]!=own[:j]:break
                    n+=1;same+=int(a.argmax()==b.argmax())
                    err=max(err,float((a-b).abs().max()))
                row["trials"][m]={"ids":own,"seconds":dt,"exact":own==seq,
                                  "matched_prefix_steps":n,"same_top1_steps":same,
                                  "max_abs_logit_error":err}
            records.append(row)
        for m in MODES[1:]:
            matched=[r for r in records if r["trials"][m]["exact"]]
            ratios=[r["baseline_seconds"]/r["trials"][m]["seconds"] for r in matched]
            results[m]={"exact_sequences":len(matched),"prompts":len(records),
                        "same_top1_steps":sum(r["trials"][m]["same_top1_steps"] for r in records),
                        "matched_prefix_steps":sum(r["trials"][m]["matched_prefix_steps"] for r in records),
                        "max_abs_logit_error":max(r["trials"][m]["max_abs_logit_error"] for r in records),
                        "same_output_median_time_ratio":statistics.median(ratios) if ratios else None,
                        "same_output_timing_cases":len(ratios)}
        assert counters["packed_head"]==counters["gather_embedding"] and counters["packed_head"]>0
    report={"stage":"F21_NATIVE_CPU_INTEGRATION","parent_sha256":PARENT,"depth":12,
            "arms":results,"actual_packed_calls":counters,"records":records,
            "candidate_accepted":False,"CPU_speed_certified":False,"output_gate":0,
            "limitations":["Existing F10 native research path only; quality remains unaccepted. No legacy J7 decoder or protected cohort.",
                           "Full-prefix recomputation retained to preserve original early intervention semantics; no KV cache equivalence claim.",
                           "Packed tied readout and gathered embedding only; body operators still reconstruct weights.",
                           "Actual same-run generation timings on 12 authored prompts, not historical CPU reference. Diagnostic gradients remain enabled by original hooks, no backward or training.",
                           "No new model weights or tokenizer changes; no whole-process RSS claim."]}
    (out/"F21_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F21_ASSESS",json.dumps({k:v for k,v in report.items() if k!="records"},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f10","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f10,a.f08,a.out)
