"""G25: reversible, research-only layer 6/7 contribution ablation.
Uses G23 identical pair probe and pinned G05 physical checkpoint. No training.
Layer numbers are human one-based. Distances are not semantic scores.
"""
import argparse,json
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
from transformers import AutoTokenizer
from research.calibri2.fusion_growth.g23_original_vs_scaled import PIN,restore,all_rows,pair_probe,digest

@contextmanager
def intervention(st,kind,idx):
    layer=st[0].model.layers[idx]
    bridge,full=st[2],st[3]
    handle=None
    saved=None
    def scale(_m,_a,out):
        if isinstance(out,tuple):
            return (out[0]*0.5,*out[1:])
        return out*0.5
    try:
        if kind=="FULL_GT":
            saved=full.gates[idx].detach().clone()
            with torch.no_grad():full.gates[idx].zero_()
        elif kind=="BRIDGE":
            j=bridge.additional_layers.index(idx)
            saved=bridge.coefficients[:,j,:].detach().clone()
            with torch.no_grad():bridge.coefficients[:,j,:].zero_()
        elif kind in ("ATTN_HALF","MLP_HALF"):
            mod=layer.self_attn if kind=="ATTN_HALF" else layer.mlp
            handle=mod.register_forward_hook(scale)
        elif kind!="BASE":
            raise ValueError(kind)
        yield
    finally:
        if handle is not None:handle.remove()
        if saved is not None:
            with torch.no_grad():
                if kind=="FULL_GT":full.gates[idx].copy_(saved)
                else:bridge.coefficients[:,j,:].copy_(saved)

def summarize(probe,base):
    b={r["pair_id"]:r for r in base["records"]}
    assert set(b)=={r["pair_id"] for r in probe["records"]}
    ds=np.array([r["relative_hidden_difference"] for r in probe["records"]])
    bs=np.array([b[r["pair_id"]]["relative_hidden_difference"] for r in probe["records"]])
    gs=np.array([r["gaps"] for r in probe["records"]])
    bg=np.array([b[r["pair_id"]]["gaps"] for r in probe["records"]])
    return {"paired_n":len(ds),"median_distance":np.median(ds,axis=0).tolist(),
      "median_distance_ratio":np.median(ds/np.maximum(bs,1e-12),axis=0).tolist(),
      "median_gap_delta":float(np.median(gs-bg)),
      "correct_direction_count":int((gs>0).sum()),"both_preferred":int((gs>0).all(axis=1).sum())}

def run(g05,f08,out):
    torch.manual_seed(20270113);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    st,*rest=restore(blobs[:6],blobs[6],blobs[7])
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    rows=all_rows(tok)
    assert len(rows)==192
    out.mkdir(parents=True,exist_ok=False)
    arms={}
    base=pair_probe(st,tok,rows)
    assert base["n"]==30
    arms["BASE"]={"probe":base,"summary":summarize(base,base)}
    def save():
        (out/"G25_REPORT.json").write_text(json.dumps({"stage":"G25_BOUNDARY_ABLATION",
          "arms":arms,"candidate_accepted":False,"training_updates":0,
          "limitations":["Full GT gate zero leaves recurrent carry intact; bridge zero isolates donor mix, not other routes.",
          "Attention and MLP outputs are halved, not fully removed; no generation or protected skill certification.",
          "G23 diagnostic data reused, not independent holdout; relative distance is not semantic accuracy."]},indent=2))
    save()
    for idx in (5,6):
        for kind in ("FULL_GT","BRIDGE","ATTN_HALF","MLP_HALF"):
            if kind=="BRIDGE" and idx not in st[2].additional_layers:continue
            name=f"L{idx+1}_{kind}"
            with intervention(st,kind,idx):
                p=pair_probe(st,tok,rows)
            arms[name]={"probe":p,"summary":summarize(p,base)}
            print("G25_ARM",name,json.dumps(arms[name]["summary"]),flush=True)
            save()
    end=pair_probe(st,tok,rows)
    witness=summarize(end,base)
    arms["RESTORED"]={"probe":end,"summary":witness}
    save()
    for a,b in zip(end["records"],base["records"]):
        assert a["pair_id"]==b["pair_id"]
        assert np.allclose(a["gaps"],b["gaps"],atol=1e-4,rtol=1e-5)
        assert np.allclose(a["relative_hidden_difference"],b["relative_hidden_difference"],atol=1e-5)
    print("G25_ASSESS",json.dumps({"restoration_verified":True,"candidate_accepted":False}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out)
