"""First retained-learning candidate from validated V008 computational lineage."""
import argparse,json,os
from pathlib import Path
import torch
from .lineage import load_parent,PARENT
from .device import move
from .training.curriculum import suite
from .training.learn import train_epoch
from .training.evaluate_checkpoint import load_packet,score
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import install,digest,G05

def run(g05,f08,parent,previous,out):
    assert torch.backends.mps.is_available() and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="0"
    torch.set_num_threads(2);torch.manual_seed(20270208)
    out.mkdir(parents=True,exist_ok=False)
    prev=json.loads((previous/"V008_REPORT.json").read_text())
    assert prev["integrated_gate"] and digest((previous/"v008_parent.safetensors").read_bytes())==PARENT
    restored,l6,slot,balance,tok,raw,bpp=load_parent(g05,f08,parent)
    st=restored[0];assert len(st[0].model.layers)==12 and bpp<=.7
    train,old,fresh=suite(tok)
    sets={"replay":train[:32],"old_dev":old,"fresh":fresh}
    (out/"SPLITS.json").write_text(json.dumps({k:[{"q":r["q"],"answer":r["answer"],"g":r["g"]} for r in v] for k,v in {"train":train,**sets}.items()},ensure_ascii=False))
    before,baseline=score(st,l6,slot,balance,tok,sets)
    assert all(v["exact_EOS"]==32 for v in before["replay"].values())
    (out/"BASELINE.json").write_text(json.dumps(baseline,ensure_ascii=False))
    move(restored,l6,slot,balance,"mps")
    with install(st,6,l6):
        final,training=train_epoch(restored,l6,slot,balance,train,raw,out)
    move(restored,l6,slot,balance,"cpu")
    load_packet(st,l6,slot,balance,final)
    after,records=score(st,l6,slot,balance,tok,{"train":train,**sets})
    (out/"EVAL.json").write_text(json.dumps(records,ensure_ascii=False))
    def correct(r):return r["answer"]==r["expected"] and r["ids"][-1]==2
    lost={k:[a for a,b in zip(baseline[k],records[k]) if correct(a) and not correct(b)] for k in sets}
    assert all(len(baseline[k])==len(records[k]) for k in sets)
    bpp+=8*(len(final)-len(raw))/G05
    gates={"replay":all(v["exact_EOS"]==32 for v in after["replay"].values()),
           "train":all(v["exact_EOS"]==512 for v in after["train"].values()),
           "fresh95":all(v["exact_EOS"]>=61 for v in after["fresh"].values()),
           "no_previously_correct_dev_lost":not any(lost.values()),"bpp":bpp<=.7,
           "physical_group_changes":all(v["changed_fp16_values"]>0 for v in training["groups"].values())}
    report={"parent_version":"V008","parent_sha":PARENT,"candidate_sha":digest(final),"known_payload_bpp":bpp,
            "training":training,"before":before,"after":after,"lost":lost,"gates":gates,
            "local_learning_gate":all(gates.values()),"candidate_accepted":False,
            "limitations":["Same G44 512 TRAIN prompts (two letters, factorial wording), 64 replay exposures; one epoch, 32 real optimizer updates. Slot frozen due to V008 zero update on this task; head/core/L6/GT train with original objective. Equal-length microbatches up to four, no padding. Fixed old/fresh sets are not independent blind evaluation. No 616 J7 or production CPU certification. Candidate retained separately, not automatically promoted."]}
    (out/"V009_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2));print("V009_REPORT",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","parent","previous","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.parent,a.previous,a.out)
