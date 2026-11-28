"""Root architecture: lineage -> weight engine -> unchanged model verification."""
import json,os,time
import torch
from .lineage import load_parent,PARENT
from .weights.architecture import check_matrix,replace_selected,error
from .weights.profile import profile_legacy
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import install,block,parameters
from research.calibri2.fusion_growth.g41_balanced_joint import balance_scope
from research.calibri2.fusion_growth.g36_alphabet_margin import encode,evaluate
from research.calibri2.fusion_growth.g43_two_letter_generalization import TRAIN,LETTERS
from v176.operators import TrainLinear,clear

def run(g05,f08,parent,out):
    assert torch.backends.mps.is_available() and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="0"
    torch.set_num_threads(2);torch.manual_seed(20270131)
    out.mkdir(parents=True,exist_ok=False)
    restored,l6,slot,balance,tok,raw,bpp=load_parent(g05,f08,parent)
    st=restored[0];model,sources=st[:2]
    assert len(model.model.layers)==12 and bpp<=.7
    rows=[encode(tok,t.format(a),a) for t in TRAIN for a in LETTERS]
    keys=[f"model.layers.{i}.{k}.weight" for i in (5,6) for k in ("self_attn.v_proj","self_attn.o_proj","mlp.down_proj")]
    chosen=[sources[k] for k in keys];byid={id(s):k for k,s in zip(keys,chosen)}
    captures={};handles=[]
    for m in model.modules():
        if isinstance(m,TrainLinear) and id(m.source) in byid:
            key=byid[id(m.source)]
            def capture(_m,args,key=key):
                if key not in captures:captures[key]=args[0].detach().reshape(-1,args[0].shape[-1])[-2:].clone()
            handles.append(m.register_forward_pre_hook(capture))
    ps=list({id(p):p for p in parameters((restored,l6,slot,balance))}.values())
    versions={id(p):p._version for p in ps}
    with install(st,6,l6),balance_scope(st,l6,balance,"balanced"):
        try:
            start=time.perf_counter()
            with profile_legacy() as stats:
                ref=[block(st,r["p"],r["features"],slot)[0].detach().clone() for r in rows[:4]]
            baseline_seconds=time.perf_counter()-start
        finally:
            for h in handles:h.remove()
        assert set(captures)==set(keys),list(captures)
        base,base_records=evaluate(st,tok,rows,slot)
        assert all(d["exact_EOS"]==32 for d in base.values()),base
        print("V001_LEGACY",json.dumps({"profile":stats,"seconds":baseline_seconds,"replay":base}),flush=True)
        (out/"LEGACY_PROFILE.json").write_text(json.dumps(stats,indent=2))
        bench={}
        for k,s in zip(keys,chosen):
            bench[k]={d:check_matrix(s,captures[k],d) for d in ("cpu","mps")}
            print("V001_MATRIX",k,json.dumps(bench[k]),flush=True)
        (out/"MATRIX_PARITY.json").write_text(json.dumps(bench,indent=2))
        clear(sources)
        with replace_selected(chosen) as counts:
            start=time.perf_counter()
            actual=[block(st,r["p"],r["features"],slot)[0].detach().clone() for r in rows[:4]]
            candidate_seconds=time.perf_counter()-start
            parity=[{**error(b,a),"top1_equal":bool(torch.equal(a.argmax(-1),b.argmax(-1)))} for a,b in zip(ref,actual)]
            score,records=evaluate(st,tok,rows,slot)
        clear(sources)
        again=block(st,rows[0]["p"],rows[0]["features"],slot)[0].detach()
    assert all(p._version==versions[id(p)] for p in ps)
    assert torch.equal(again,ref[0])
    same=all(a["ids"]==b["ids"] for a,b in zip(base_records,records))
    passed=all(p["relative_l2"]<.002 and p["top1_equal"] for p in parity) and same and counts["compact_calls"]>0
    (out/"REPLAY.json").write_text(json.dumps({"parent":base_records,"compact":records},ensure_ascii=False))
    # Store unchanged physical parent explicitly; not a new trained model.
    (out/"v001_parent.safetensors").write_bytes(raw)
    report={"parent_sha":PARENT,"checkpoint_sha":PARENT,"logical_layers":12,"known_payload_bpp":bpp,
            "legacy_profile":stats,"legacy_four_prompt_seconds":baseline_seconds,
            "compact_four_prompt_seconds":candidate_seconds,"matrix_benchmark":bench,
            "compact_calls":counts["compact_calls"],"full_model_logit_parity":parity,
            "parent_replay":base,"candidate_replay":score,"all_replay_ids_equal":same,
            "operator_parity_gate":passed,"training_updates":0,"candidate_accepted":False,
            "limitations":["Six real original-bank linear operators only; compact lookup still uses unpacked int64 codes and FP32 dot products, but no dense O x I reconstruction for those operators. MPS operator microbench on two captured token inputs; whole-model replacement tested on CPU only. No new learned weights, full runtime speed, 616 historical skills or all-in executable bytes certified.","Timings are exploratory single-run CPU full model and median of three microbench runs, not production throughput. Legacy profiling host dispatch is not isolated GPU elapsed time."]}
    (out/"V001_REPORT.json").write_text(json.dumps(report,indent=2));print("V001_REPORT",json.dumps(report),flush=True)
