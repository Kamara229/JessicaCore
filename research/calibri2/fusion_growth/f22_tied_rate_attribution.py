"""F22: isolate tied vocabulary from body quantization after F18.

Fixed read-only factorial ablations from original physical K32. No J7
optimization. Q8/Q4 native-head storage charged, not tested by legacy J7.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from research.calibri2.fusion_growth import f03_rate_train_core as codec
from research.calibri2.fusion_growth import f18_retention_attribution as attr
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,digest,metrics,unpack
from research.calibri2.fusion_growth.f02_entropy_core import raw_core
from research.calibri2.fusion_growth.f09_head_rate_codec import split_head,HDR,open_bundle
from research.calibri2.fusion_growth.f13_profile_rebase import extra_open,BANK_NAME,BANK_SHA
from research.calibri2.fusion_growth.f15_legacy_j7_replay import (
    configure_original_selector,model_state,replay,output_digest,s30,s87,
    source_questions_456,materialize_train_only,new_train_paraphrase_probes)
from direct_codebook_v59 import CB59

EMBED="model.embed_tokens.weight"
ARMS={"K16_EMBED_K32_BODY":(16,32),"K32_EMBED_K8_BODY":(32,8),
      "K32_EMBED_K16_BODY":(32,16)}
def k_for(rate,name):
    e,b=ARMS[rate]
    return e if name==EMBED else b

def run(f01,f08,f15,f18,out):
    torch.set_num_threads(2);torch.manual_seed(20261210)
    original_files=unpack((f01/"F01.xz.bundle").read_bytes())
    for k,v in PIN.items():assert digest(original_files[k])==v
    old=CB59(raw_core(original_files[NAMES[0]]))
    side={k:original_files[k] for k in NAMES[1:]}
    extra=(f08/"legacy_extras.bundle").read_bytes()
    inventory,extras=extra_open(extra)
    assert digest(extras[BANK_NAME])==BANK_SHA
    previous=json.loads((f18/"F18_REPORT.json").read_text())
    assert previous["baseline_exact_616"]==616
    assert previous["arms"]["RATE_ONLY_K16_all"]["exact_616"]==549
    assert previous["arms"]["RATE_ONLY_K16_embed_K8_body"]["exact_616"]==549
    assert previous["arms"]["ORIGINAL_K32_TRAINED_GT"]["exact_616"]==615
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    assert tok.vocab_size==49152 and tok.eos_token_id==2
    s30.TOK=tok
    oq=source_questions_456(s87.families(tok));nq,_,_=materialize_train_only(tok)
    probes=new_train_paraphrase_probes()
    rows={**{"OLD/"+k:{"question":q} for k,q in oq.items()},
          **{"NEW/"+r["id"]:{"question":r["question"]} for r in nq},
          **{f"PROBE/{i:02d}":{"question":q} for i,(q,_,_) in enumerate(probes)}}
    del oq,nq,probes
    forced={k for k in rows if k.startswith("PROBE/")}
    keys={k:digest(v["question"].encode()) for k,v in rows.items()}
    assert len(rows)==len(set(keys.values()))==616
    probe_ids={keys[k] for k in forced}
    ref=json.loads((f15/"F15_BASELINE_IDS.json").read_text())
    assert set(ref)==set(keys.values())
    assert output_digest({k:v for k,v in ref.items() if k not in probe_ids})==attr.H584
    assert output_digest({k:v for k,v in ref.items() if k in probe_ids})==attr.H32
    out.mkdir(parents=True,exist_ok=False)
    configure_original_selector()
    original=model_state(old,side,extras,False)
    pred,_,_=replay(original,tok,rows,forced);del original
    assert pred==ref
    results={}
    # All configurations fixed in source before evaluation. No fitting or
    # model choice from historical outputs. Scoped codec overrides only.
    for rate,(e,b) in ARMS.items():
        with patch.object(codec,"k_for",k_for),patch.object(attr,"k_for",k_for):
            packet,archive=attr.rate_only(old,side,rate)
            # Retain alternate bank byte-for-byte when all body IDs are K32.
            if b==32:packed_extra,new_extras=extra,extras
            else:packed_extra,new_extras=attr.rebase(old,archive,inventory,extras,rate,digest(packet))
            assert archive.handles["lm_head.weight"] is archive.handles[EMBED]
            for name,h in archive.handles.items():
                canonical=archive.meta["source_alias_keys"].get(name,name)
                if k_for(rate,canonical)==32:
                    source=old.handles[name]
                    assert h.packed.tobytes()==source.packed.tobytes()
                    assert np.array_equal(h.centroids,source.centroids)
                assert np.array_equal(h.row_scales,old.handles[name].row_scales)
            model,sources=codec.make(archive)
            budgets={}
            for bits in (8,4):
                compact,head,_=split_head(side,bits)
                cp=codec.encode(archive.blob,sources,rate,compact)
                q=HDR.pack(b"FG09",len(cp),len(head),len(packed_extra))+cp+head+packed_extra
                a,sd,ex=open_bundle(q)
                assert a.blob==archive.blob and ex==packed_extra
                assert all(sd[k]==side[k] for k in NAMES[1:] if k!=NAMES[3])
                budgets[f"Q{bits}"]={**metrics(len(q)),"sha256":digest(q),
                                     "head_active_in_J7":False}
                (out/(rate+f"_Q{bits}.bundle")).write_bytes(q)
            del model,sources
        state=model_state(archive,side,new_extras,False)
        pred,_,counts=replay(state,tok,rows,forced);del state
        same={k for k in ref if pred[k]==ref[k]}
        r={"exact_616":len(same),"drift":616-len(same),
           "exact_groups":{g:sum(keys[k] in same for k in rows if k.startswith(g+"/")) for g in ("OLD","NEW","PROBE")},
           "embedding_K":e,"body_K":b,"route_counts":counts,
           "FP16_plus_legacy":metrics(len(packet)+len(packed_extra)),
           "quantized_native_head_storage_only":budgets}
        results[rate]=r
        (out/(rate+"_IDS.json")).write_text(json.dumps(pred,sort_keys=True))
        print("F22_ARM",rate,json.dumps(r,sort_keys=True),flush=True)
    report={"stage":"F22_TIED_VOCABULARY_RATE_ATTRIBUTION","baseline_exact_616":616,
            "arms":results,"optimizer_steps":0,"depth":12,"FP32_teacher_calls":0,
            "candidate_accepted":False,"output_gate":0,"all_in_certified":False,
            "limitations":["Factorial frozen diagnostic, no training, no adaptive model selection on protected J7.",
                           "Q8/Q4 native-head bytes charged but those heads are inactive in legacy replay; not a quality or integrated inference claim.",
                           "Body rate reduction also requires reprojected alternate banks. A K32 embedding changes both input lookup and tied output together.",
                           "No tokenizer/runtime budget or CPU speed certification. Historical exact sequences are not equivalent to correct answers."]}
    (out/"F22_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F22_ASSESS",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f08","f15","f18","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f08,a.f15,a.f18,a.out)
