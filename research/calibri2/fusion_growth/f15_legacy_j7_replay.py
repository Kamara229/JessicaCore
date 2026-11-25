"""F15: actual legacy J7 replay after profile rebasing. No training.
Reconstruct S194 source selector chain and independently verify all 616
baseline generated-token digests BEFORE evaluating any mixed-rate weights.
This is the legacy decoder path ONLY; new F06 native regions are not active.
"""
import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
import torch
from safetensors.torch import load
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,digest,metrics,unpack
from research.calibri2.fusion_growth.f02_entropy_core import raw_core
from research.calibri2.fusion_growth.f03_rate_train_core import decode
from research.calibri2.fusion_growth.f13_profile_rebase import extra_open,BANK_NAME
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s112.prompt_conditioned_first_token_mps import load_cpu_original
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel
from calibri2.s126.compact_rank4_q2 import ENCODED_KEYS,load_compact_q2
from calibri2.s132.progressive_core import ProgressiveRouteCore,IDS_KEY,WEIGHTS_KEY
from calibri2.s95.train_and_audit_router import PromptOnlyRouter,read_verified_bank
from calibri2.s98.train_owner_anchored_copy import TinyCopyScorer
from calibri2.s99.guard_neural_copy_456_cpu import apply_profile
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s122.independent_binding_train_curriculum import materialize_train_only
from calibri2.s144.train_one_continued_j7 import frozen_routes
from calibri2.s159.parallel_slot_binding import ParallelSlotBindingHead
from calibri2.s160 import real_parallel_slots_j7 as s160
from calibri2.s168 import novel_train_frames_generalization_cpu as s168
from calibri2.s170 import structural_owner_integrated_cpu as s170
from calibri2.s172 import confident_owner_consistency_cpu as s172
from calibri2.s175 import negative_owner_guard_real_j7 as s175
from calibri2.s176 import s174_legacy_compatible_j7 as s176
from calibri2.s181.full_456_tokenized_shadow_replay import source_questions_456
from calibri2.s182 import real_cpu_j7_shadow_guard as s182
from calibri2.s187 import real_cpu_j7_history_s69_guard as s187
from calibri2.s190.real_cpu_new_train_paraphrase_probe import new_train_paraphrase_probes
from calibri2.s194.actual_cpu_j7_bridge import output_digest
from direct_codebook_v59 import CB59

PARENT_SHA="7dcc391ce6001f88f6e7c12498c4f371b83431b049e4db2cb65d7f0b89ab722a"
ARM_SHA={"DIRECT":"5dbb574737bab118d07cb66b04c5d09216587daae81e71f9e7f7676cb643d11b",
         "DELTA":"41399c41fe4ccd63db03de3e7aa67edaadd3864332b161451d86d504558a4e5a"}
ROUTER="S95_PROMPT_ONLY_HASHED_NEURAL_ROUTER_FP16.safetensors"
COPY="S98_MPS_TRAINED_NUMERIC_OWNER_COPY_FP16.safetensors"
PARALLEL="S160_REAL_PARALLEL_SLOT_FP16.safetensors"

def configure_original_selector():
    # Identical bindings made by S187 -> S182 -> S176 -> S175 -> S172
    # -> S170 -> S168; no prompt/answer table or candidate-specific routing.
    s182.REAL_S175_BASE_GUARD=s175.OLD_S172
    s187.ORIGINAL_S182_SELECTOR=s182.shadow_selector
    s175.OLD_S172=s187.select_with_s186
    s170.owner_value=s176.owner_value_compatible
    s172.owner_consistent_confident_guard=s175.guard
    s170.s170_owner_guarded_parallel=s175.guard
    s168.owner_guarded_parallel=s175.guard
    s160.select_parallel=s168.owner_guarded_parallel
    s160.cached_full_cpu_parallel=s168.owner_capture_decoder

def model_state(archive,side,extras,trained):
    legacy=load(side[NAMES[1]]);inherited=load(side[NAMES[2]])
    assert digest(side[NAMES[2]])==PIN[NAMES[2]]
    assert set(inherited)==set(legacy)|set(ENCODED_KEYS)|{IDS_KEY,WEIGHTS_KEY}
    assert all(torch.equal(inherited[k],v) for k,v in legacy.items())
    cpu=load_cpu_original(archive.blob,legacy)
    model=cpu[0]
    bridge=PromptConditionedDualTunnel(cpu[6]).eval().install(model)
    bridge.coefficients.requires_grad_(False)
    full=FullLayerStateTunnel(width=320,depth=12,rank=4).eval()
    load_compact_q2(full,{k:inherited[k] for k in ENCODED_KEYS});full.install(model)
    expansion=ProgressiveRouteCore(rank=896).eval().bind_original_route_bridge(bridge).install(model)
    expansion.load_active_fp16({IDS_KEY:inherited[IDS_KEY],WEIGHTS_KEY:inherited[WEIGHTS_KEY]})
    if trained:
        gate=next(iter(load(side[NAMES[4]]).values())).float()
        assert gate.shape==(12,)
        with torch.no_grad():full.gates.copy_(gate)
    model.eval().requires_grad_(False);full.requires_grad_(False)
    expansion.requires_grad_(False)
    assert len(model.model.layers)==12 and len(cpu[1])==85
    router=PromptOnlyRouter();router.load_state_dict(load(extras[ROUTER]),strict=True);router.eval().requires_grad_(False)
    scorer=TinyCopyScorer();scorer.load_state_dict(load(extras[COPY]),strict=True);scorer.eval().requires_grad_(False)
    head=ParallelSlotBindingHead(320,slots=2,rank=s160.SLOT_RANK)
    head.load_state_dict({k:v.float() for k,v in load(extras[PARALLEL]).items()},strict=True)
    head.eval().requires_grad_(False)
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/BANK_NAME;p.write_bytes(extras[BANK_NAME]);bank=read_verified_bank(p,digest(extras[BANK_NAME]))
    return cpu,bridge,full,expansion,router,scorer,head,bank

def replay(state,tok,rows,forced):
    cpu,bridge,full,expansion,router,scorer,head,bank=state
    model,sources,first,later,pointer,lm_head,inner,base,base_gt=cpu
    routes=frozen_routes(tok,router,rows)
    for k in forced:routes[k]="S93"
    pred={};times={};current=None;counts=Counter()
    for k in sorted(rows):
        q=rows[k]["question"];route=routes[k]
        if route!=current:
            apply_profile(sources,inner,route,bank,base,base_gt);current=route
        feat=torch.as_tensor(s160.prompt_only_vector(tok,q),dtype=torch.float32) if route=="S93" else None
        t=time.perf_counter()
        with torch.inference_mode(),expansion.prefill_only(model),s160.prefill_only_tunnel(model,full):
            with s160.frozen_cpu_phase_scope(model,bridge,route,feat,phase="prefill_only"):
                if route=="S93":
                    ans,note=s160.cached_full_cpu_parallel(model,sources,first,later,lm_head,tok,q,pointer,scorer,head)
                    counts["parallel"]+=int(note["parallel"])
                else:
                    ans,_=cached_full_cpu_with_neural_copy(model,sources,first,later,lm_head,tok,q,pointer,scorer)
        key=digest(q.encode());pred[key]=list(ans);times[key]=time.perf_counter()-t
        counts[route]+=1
    assert len(pred)==len(rows)
    return pred,times,dict(counts)

def run(f01,f06,f08,f13,s194,out):
    torch.set_num_threads(2);torch.manual_seed(20261203)
    files=unpack((f01/"F01.xz.bundle").read_bytes())
    for k,v in PIN.items():assert digest(files[k])==v
    original=CB59(raw_core(files[NAMES[0]]));old_side={k:files[k] for k in NAMES[1:]}
    packet=(f06/"F06_JOINT.bundle").read_bytes();assert digest(packet)==PARENT_SHA
    mixed,side=decode(packet)
    _,extras=extra_open((f08/"legacy_extras.bundle").read_bytes())
    assert digest(extras[BANK_NAME])=="3ea693cc95cae5af3f09a01e72ac739ce2899251d10a7949455d60405c3ca8f5"
    reference=json.loads((s194/"S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json").read_text())
    assert reference["all_616_exact_real_CPU_generated_tokens_captured"] is True
    cert=reference["J7Retention584_frozen_baseline"]
    assert cert["old_full_eos_correct"]==456 and cert["new_full_eos_correct"]==128
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    assert tok.vocab_size==49152 and tok.eos_token_id==2
    s30.TOK=tok
    oq=source_questions_456(s87.families(tok))
    nq,_,_=materialize_train_only(tok)
    probes=new_train_paraphrase_probes()
    rows={**{"OLD/"+k:{"question":q} for k,q in oq.items()},
          **{"NEW/"+r["id"]:{"question":r["question"]} for r in nq},
          **{f"PROBE/{i:02d}":{"question":q} for i,(q,_,_) in enumerate(probes)}}
    forced={k for k in rows if k.startswith("PROBE/")}
    keys={k:digest(v["question"].encode()) for k,v in rows.items()}
    assert len(rows)==len(set(keys.values()))==616 and len(forced)==32
    # Drop labels from all inputs before inference; historical hashes are used
    # exclusively after all predictions in each arm have been generated.
    del oq,nq,probes
    out.mkdir(parents=True,exist_ok=False)
    configure_original_selector()
    base=model_state(original,old_side,extras,False)
    bp,bt,bc=replay(base,tok,rows,forced);del base
    probe_ids={keys[k] for k in forced}
    old584={k:v for k,v in bp.items() if k not in probe_ids}
    new32={k:v for k,v in bp.items() if k in probe_ids}
    dg584=output_digest(old584);dg32=output_digest(new32)
    matched=dg584==cert["protected_584_generation_digest"] and dg32==cert["train32_generation_digest"]
    (out/"F15_BASELINE_IDS.json").write_text(json.dumps(bp,sort_keys=True))
    report={"stage":"F15_LEGACY_J7_REPLAY","parent_sha256":PARENT_SHA,
            "baseline_584_digest":dg584,"baseline_32_digest":dg32,
            "historical_584_digest":cert["protected_584_generation_digest"],
            "historical_32_digest":cert["train32_generation_digest"],
            "baseline_exact_historical":matched,"baseline_route_counts":bc,
            "candidate_accepted":False,"all_in_certified":False,"output_gate":0,"arms":{},
            "limitations":["Legacy J7 decoder only; F06 joint native head and on-path NRA3/NPA1 child are NOT active.",
                           "No updates to protected prompts, original source selectors or model weights. Historical digest check required before candidate evaluation.",
                           "Per-case decode durations diagnostic only; no speed certification if tokens differ.",
                           "No independent understanding cohort or complete runtime/tokenizer budget certification."]}
    print("F15_BASELINE",json.dumps(report,sort_keys=True),flush=True)
    if matched:
        for arm in ("DIRECT","DELTA"):
            e=(f13/(arm+"_legacy.bundle")).read_bytes();assert digest(e)==ARM_SHA[arm]
            _,b=extra_open(e)
            assert all(b[k]==extras[k] for k in extras if k!=BANK_NAME)
            candidate=model_state(mixed,side,b,True)
            cp,ct,cc=replay(candidate,tok,rows,forced);del candidate
            same={k for k in bp if bp[k]==cp[k]}
            groups={g:sum(keys[k] in same for k in rows if k.startswith(g+"/")) for g in ("OLD","NEW","PROBE")}
            a={**metrics(len(packet)+len(e)),"exact_616":len(same),"drift":616-len(same),"exact_groups":groups,
               "route_counts":cc,"same_token_decode_time_ratio":sum(bt[k] for k in same)/max(1e-12,sum(ct[k] for k in same)),
               "same_token_timing_cases":len(same),"CPU_speed_certified":False}
            report["arms"][arm]=a
            (out/(arm+"_IDS.json")).write_text(json.dumps(cp,sort_keys=True))
            print("F15_ARM",arm,json.dumps(a,sort_keys=True),flush=True)
    else:
        report["decision"]="BASELINE_RECONSTRUCTION_MISMATCH_NO_CANDIDATE_TEST"
    (out/"F15_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F15_ASSESS",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f06","f08","f13","s194","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f06,a.f08,a.f13,a.s194,a.out)
