"""Paired actual CPU model comparisons, V001 versus packed C operator."""
from contextlib import nullcontext
from time import perf_counter
import torch
from .architecture import replace_selected,error
from .packed_cpu import PackedCPU
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import block
from research.calibri2.fusion_growth.g36_alphabet_margin import evaluate

def run_probe(st,tok,rows,slot,chosen,lib):
    operator=PackedCPU(lib)
    ref=None;results={}
    def once():
        return torch.stack([block(st,r["p"],r["features"],slot)[0].detach() for r in rows[:4]])
    for arm in ("reference","packed"):
        with replace_selected(chosen,operator) if arm=="packed" else replace_selected(chosen):
            once()
            times=[]
            for _ in range(3):
                start=perf_counter();z=once();times.append(perf_counter()-start)
            if ref is None:ref=z
            parity={**error(z,ref),"top1_equal":bool(torch.equal(z.argmax(-1),ref.argmax(-1)))}
            start=perf_counter();score,records=evaluate(st,tok,rows,slot);elapsed=perf_counter()-start
        results[arm]={"seconds":times,"median_seconds":sorted(times)[1],"parity":parity,
                      "replay":score,"replay_seconds":elapsed,"records":records}
        print("V007_ARM",arm,{k:v for k,v in results[arm].items() if k!="records"},flush=True)
    a,b=results["reference"],results["packed"]
    same=a["records"]==b["records"]
    assert operator.stats["calls"]>0
    return {"arms":results,"packed_stats":operator.stats,"all_replay_equal":same,
            "speed_ratio":a["median_seconds"]/b["median_seconds"],
            "replay_speed_ratio":a["replay_seconds"]/b["replay_seconds"],
            "parity_gate":same and b["parity"]["top1_equal"] and b["parity"]["relative_l2"]<.002}
