"""V007 orchestration: validated V006 lineage, packed CPU inference candidate."""
import argparse,json
from pathlib import Path
import torch
from .lineage import load_parent,PARENT
from .weights.cpu_probe import run_probe
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import install,parameters,digest
from research.calibri2.fusion_growth.g41_balanced_joint import balance_scope
from research.calibri2.fusion_growth.g36_alphabet_margin import encode
from research.calibri2.fusion_growth.g43_two_letter_generalization import TRAIN,LETTERS

def run(g05,f08,parent,previous,out,lib):
    torch.set_num_threads(2);torch.manual_seed(20270206)
    out.mkdir(parents=True,exist_ok=False)
    prev=json.loads((previous/"V006_REPORT.json").read_text())
    assert prev["operator_gate"] and digest((previous/"v006_parent.safetensors").read_bytes())==PARENT
    restored,l6,slot,balance,tok,raw,bpp=load_parent(g05,f08,parent)
    st=restored[0];model,sources=st[:2]
    assert len(model.model.layers)==12 and bpp<=.7
    rows=[encode(tok,t.format(a),a) for t in TRAIN for a in LETTERS]
    chosen=[sources[f"model.layers.{i}.{k}.weight"] for i in (5,6) for k in ("self_attn.v_proj","self_attn.o_proj","mlp.down_proj")]
    ps=list({id(p):p for p in parameters((restored,l6,slot,balance))}.values());versions={id(p):p._version for p in ps}
    with install(st,6,l6),balance_scope(st,l6,balance,"balanced"):
        result=run_probe(st,tok,rows,slot,chosen,lib)
    assert all(p._version==versions[id(p)] for p in ps)
    records=result["arms"]["packed"].pop("records")
    result["arms"]["reference"].pop("records")
    prior=json.loads((previous/"REPLAY.json").read_text())
    same=len(prior)==len(records) and all(a["ids"]==b["ids"] for a,b in zip(prior,records))
    (out/"v007_parent.safetensors").write_bytes(raw)
    (out/"REPLAY.json").write_text(json.dumps(records,ensure_ascii=False))
    report={"parent_version":"V006","checkpoint_sha":digest(raw),"known_payload_bpp":bpp,"logical_layers":12,
            "probe":result,"all_replay_ids_equal":same,"packed_gate":result["parity_gate"] and same,
            "speed_gate":result["speed_ratio"]>=.9 and result["replay_speed_ratio"]>=.9,
            "training_updates":0,"candidate_accepted":False,
            "limitations":["Six original bank matrices only; actual packed C CPU forward, inference-only; FP32 arithmetic with no full dense matrix for selected sources. Packed plans coexist with legacy int64 indices; whole runtime memory and executable bpp not certified. V006 active MPS operator rejected for speed; V001 remains training operator. No optimizer update or 616 J7 certification."]}
    (out/"V007_REPORT.json").write_text(json.dumps(report,indent=2));print("V007_REPORT",json.dumps(report),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","parent","previous","out","lib"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.parent,a.previous,a.out,a.lib)
