"""Preregistered 2x2x2 + matched-prefix/suffix scans on real native CPU."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_runtime_v59 import PackedEmbedding,PackedLinear
from v110.experiment import load_controls, probes, summarize_text
from v110.mix import distributions,compare
from v110.protocol import CUTS, validate_geometry
from v94.generation import generate_one
from .factors import compose,arrangements

# Reuse diagnosis-only prompts, including 3 objective known-answer probes.
# No model training, threshold fitting or candidate selection in this run.
FACTORS=("E","B","H")
@torch.inference_mode()
def run(archive,out):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    packed,int6,physical=load_controls(archive,out)
    validate_geometry(len(int6.model.layers))
    prompts=probes(tok)
    inputs=[torch.tensor([tok.encode(p["chat"],add_special_tokens=False)],dtype=torch.long) for p in prompts]
    reference=[distributions(int6,x) for x in inputs]
    report={"source":physical,"model_revision":MODEL_SHA,
            "source_actual_TEST_not_opened":True,
            "unpaid_untied_factors_not_compact_model":True,
            "factorial":{}, "scans":{"prefix":{},"suffix":{}}}
    # 8 genuine paired whole-model next token distributions and first replies.
    for e,b,h in arrangements():
        key=f"E{e}B{b}H{h}"
        with compose(packed,int6,embed_int6=bool(e),
                     blocks_int6=range(30) if b else (),head_int6=bool(h)):
            rows=[]
            for i, (prompt,x) in enumerate(zip(prompts,inputs)):
                logits=distributions(packed,x)
                row=compare(reference[i],logits)
                answer=generate_one(packed,tok,prompt["chat"],stop_at_eos=True)
                rows.append({"index":i,**row,**summarize_text(tok,answer)})
            if (e,b,h)==(1,1,1):
                # 1D norms are exact; all actual source 2D modules now match.
                assert all(r["top10_overlap"]==10 and abs(r["kl_from_INT6_nats"])<1e-4 for r in rows), "INT6 oracle restoration mismatch"
        report["factorial"][key]=rows
        print(json.dumps({"experiment":"v111_factorial","case":key,
                          "top10":[r["top10_overlap"] for r in rows],
                          "kl":[round(r["kl_from_INT6_nats"],3) for r in rows],
                          "text_preview":[r["first_turn_text"][:90] for r in rows]}),flush=True)

    # Head AND embeddings always INT6, isolate which body segments are vital.
    for direction in ("prefix","suffix"):
        for cut in CUTS:
            layer_ids=range(cut) if direction=="prefix" else range(30-cut,30)
            with compose(packed,int6,embed_int6=True,
                         blocks_int6=layer_ids,head_int6=True):
                rows=[compare(reference[i],distributions(packed,x))
                      for i,x in enumerate(inputs[:3])]
            report["scans"][direction][str(cut)]=rows
            print(json.dumps({"experiment":"v111_scan","direction":direction,
                              "layers":cut,"overlap":[r["top10_overlap"] for r in rows],
                              "kl":[round(r["kl_from_INT6_nats"],3) for r in rows]}),flush=True)

    if not isinstance(packed.model.embed_tokens,PackedEmbedding) or not isinstance(packed.lm_head,PackedLinear):
        raise AssertionError("packed source tied modules were not restored")
    if any(not isinstance(m.self_attn.q_proj,PackedLinear) for m in packed.model.layers):
        raise AssertionError("CB59 direct transformer failed restoration")
    report["diagnostic_packed_source_restored"]=True
    (out/"v111_three_factor.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"experiment":"v111_completed","physical_new_model":False,
                      "official_TEST_unopened":True}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("original",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    run(args.original,args.output)
