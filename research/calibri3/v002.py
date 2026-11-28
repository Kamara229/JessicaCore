"""V002 root experiment; V001 checkpoint byte-identical, no optimization."""
import argparse,json,os
from pathlib import Path
import torch
from .lineage import load_parent,PARENT
from .device import move
from .training.architecture import run_probe
from .weights.architecture import replace_selected
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import install,parameters,digest
from research.calibri2.fusion_growth.g41_balanced_joint import balance_scope
from research.calibri2.fusion_growth.g36_alphabet_margin import encode,evaluate
from research.calibri2.fusion_growth.g43_two_letter_generalization import TRAIN,LETTERS

def run(g05,f08,parent,previous,out):
    assert torch.backends.mps.is_available() and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="0"
    torch.set_num_threads(2);torch.manual_seed(20270201)
    out.mkdir(parents=True,exist_ok=False)
    prev=json.loads((previous/"V001_REPORT.json").read_text())
    assert prev["operator_parity_gate"] and digest((previous/"v001_parent.safetensors").read_bytes())==PARENT
    restored,l6,slot,balance,tok,raw,bpp=load_parent(g05,f08,parent)
    st=restored[0];model,sources=st[:2]
    rows=[encode(tok,t.format(a),a) for t in TRAIN for a in LETTERS]
    assert len(model.model.layers)==12 and bpp<=.7
    move(restored,l6,slot,balance,"mps")
    ps=list({id(p):p for p in parameters((restored,l6,slot,balance))}.values());versions={id(p):p._version for p in ps}
    with install(st,6,l6),balance_scope(st,l6,balance,"balanced"):
        result=run_probe(restored,slot,rows[:2])
    assert all(p._version==versions[id(p)] for p in ps)
    print("V002_MPS",json.dumps(result),flush=True)
    move(restored,l6,slot,balance,"cpu")
    chosen=[sources[f"model.layers.{i}.{k}.weight"] for i in (5,6) for k in ("self_attn.v_proj","self_attn.o_proj","mlp.down_proj")]
    with install(st,6,l6),balance_scope(st,l6,balance,"balanced"),replace_selected(chosen):
        replay,records=evaluate(st,tok,rows,slot)
    prior=json.loads((previous/"REPLAY.json").read_text())["compact"]
    same=len(prior)==len(records) and all(a["ids"]==b["ids"] for a,b in zip(prior,records))
    (out/"v002_parent.safetensors").write_bytes(raw)
    (out/"REPLAY.json").write_text(json.dumps(records,ensure_ascii=False))
    report={"parent_version":"V001","parent_sha":PARENT,"checkpoint_sha":digest(raw),"known_payload_bpp":bpp,"logical_layers":12,"probe":result,
            "replay":replay,"all_replay_ids_equal":same,"freeze_gate":result["parity_gate"] and same,
            "training_updates":0,"candidate_accepted":False,
            "limitations":["Only source parameters explicitly frozen; legacy NRA3 gradient-required hooks untouched. Matched MPS diagnostic two TRAIN prompts, warmup plus three rounds, no optimizer update. V001 six-operator compact route retained in both arms. No full J7 or deployment speed certification."]}
    (out/"V002_REPORT.json").write_text(json.dumps(report,indent=2));print("V002_REPORT",json.dumps(report),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","parent","previous","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.parent,a.previous,a.out)
