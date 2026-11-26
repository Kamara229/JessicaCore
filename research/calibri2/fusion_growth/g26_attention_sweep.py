"""G26: fixed attention gain sweep; real Russian free generation, no fitting.
G25 evidence only weakly favors L6 attention reduction. Test frozen G05.
"""
import argparse,json,statistics
from pathlib import Path
from contextlib import contextmanager
import torch
from transformers import AutoTokenizer
from research.calibri2.fusion_growth.g23_original_vs_scaled import PIN,restore,all_rows,pair_probe,digest,generate
from research.calibri2.fusion_growth.g25_boundary_ablation import summarize
@contextmanager
def gain(st,g):
    def hook(_m,_a,out):
        return (out[0]*g,*out[1:]) if isinstance(out,tuple) else out*g
    h=st[0].model.layers[5].self_attn.register_forward_hook(hook)
    try:yield
    finally:h.remove()
def run(g05,f08,out):
    torch.manual_seed(20270113);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    st,*_=restore(blobs[:6],blobs[6],blobs[7])
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    rows=all_rows(tok);assert len(rows)==192
    out.mkdir(parents=True,exist_ok=False)
    base=pair_probe(st,tok,rows)
    arms={}
    # Predeclared gains, same entire suite; exploratory, not independent test.
    for g in (1.0,0.75,0.5,0.25):
        with gain(st,g):
            p=pair_probe(st,tok,rows)
            answers=[]
            with (out/("gain_"+str(g)+".jsonl")).open("w") as f:
                for i,row in enumerate(rows,1):
                    a=generate(st,tok,row);answers.append(a)
                    f.write(json.dumps({"id":row["id"],"q":row["q"],"accepted":row["accepted"],**a},ensure_ascii=False)+"\n");f.flush()
                    if i%32==0:print("G26_PROGRESS",g,i,flush=True)
        closed=[a for a in answers if a["accepted_exact"] is not None]
        s={"gain":g,"exact":sum(a["accepted_exact"] is True for a in closed),"closed_n":len(closed),"EOS":sum(a["eos"] for a in answers),"first_word":sum(a["first_word_exact"] for a in closed),"replacement_char":sum(a["replacement_char"] for a in answers),"unique":len(set(a["text"] for a in answers)),"tokens_per_second":sum(len(a["ids"]) for a in answers)/sum(a["seconds"] for a in answers),"pair":summarize(p,base)}
        arms[str(g)]=s
        print("G26_ARM",json.dumps(s),flush=True)
        (out/"G26_REPORT.json").write_text(json.dumps({"arms":arms,"candidate_accepted":False,"training_updates":0,"limitations":["Reused G23 suite, diagnostic gain sweep only; no independent holdout or J7 certification.","No additional learned weights; hook overhead and variable output lengths not production CPU certification."]},indent=2))
    end=pair_probe(st,tok,rows)
    import numpy as np
    for a,b in zip(end["records"],base["records"]):
        assert a["pair_id"]==b["pair_id"] and np.allclose(a["gaps"],b["gaps"],atol=1e-4,rtol=1e-5)
    print("G26_ASSESS",json.dumps({"candidate_accepted":False,"restoration_verified":True}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out)
