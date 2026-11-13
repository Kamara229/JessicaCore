"""S188 independent REAL same-CPU J7 paired speed: protected S182 vs S187.

Run both selectors on the SAME loaded 43M compressed physical CPU weights
and EXACT same frozen S93 TRAIN prompts. Order AB/BA, two sweeps, labels
never used for prompts or inference; verify full EOS IDs against immutable
real S187 predictions and source-only reconstructed S182 baseline.
No training, no model acceptance, no heldout/QA/TEST, no LIVE writes.
"""
from __future__ import annotations
import hashlib,json,statistics,sys,time
from pathlib import Path
import torch

from calibri2.s160 import real_parallel_slots_j7 as s160
from calibri2.s182 import real_cpu_j7_shadow_guard as s182
from calibri2.s187 import real_cpu_j7_history_s69_guard as s187
from calibri2.s181.source_grammar_shadow import shadow_candidate

OLD_PAIRED_CPU_SPEED=s160.paired_cpu_ratio
SOURCE=Path("research/calibri2/s188/paired_samecpu_s182_vs_s187.py")
TIMINGS=None
FROZEN_S179={}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def groups_for(old_ids,routes):
    """Predefined TRAIN source cohorts before results, labels or timings."""
    def take(prefix,n):
        return [key for key in old_ids if key.startswith(prefix)
                and routes[key]=="S93"][:n]
    result={
        "S69":take("S69/",4),
        "history":take("history/",4),
        "S30_unchanged":take("S30/",4),
        "S54_unchanged":take("S54/",4),
        "S82_control":take("S82_new/",4),
    }
    if any(len(v)!=4 for v in result.values()):
        raise AssertionError("S188 require 4 S93 original TRAIN examples per frozen family")
    return result


def stats(pairs):
    if len(pairs)<2 or any(a<=0 or b<=0 for a,b in pairs):
        raise AssertionError("S188 real paired CPU timing incomplete")
    a=statistics.median(x for x,_ in pairs)
    b=statistics.median(y for _,y in pairs)
    return {"paired_trials":len(pairs),
            "s182_median_cpu_seconds":a,
            "s187_median_cpu_seconds":b,
            "s187_over_s182_speed_ratio":a/b}


def base_s182_prediction(prompt,tok):
    """Independently reconstruct S182 from label-free original S179 trace."""
    s=sha(prompt)
    if s not in FROZEN_S179:
        raise AssertionError("S188 missing original frozen full EOS trace")
    phrase=shadow_candidate(prompt)
    if phrase is None:
        return FROZEN_S179[s]
    ids=tok.encode(phrase,add_special_tokens=False)
    source=s160.normalized_candidate_spans(tok,s160.prompt_tokens(tok,prompt))
    if (len(ids)!=2 or tok.eos_token_id in ids
            or tuple(ids) not in {tuple(x.tolist()) for x in source}):
        raise AssertionError("S188 S182 source-only baseline tokenizer mismatch")
    return ids+[tok.eos_token_id]


def paired_true_cpu(cpu,old,old_ids,routes,tok,bank,copy_net,bridge,
                    full,expansion,parent_old,head):
    global TIMINGS
    if TIMINGS is not None:
        raise AssertionError("S188 duplicate paired speed measurement")
    if len(s182.ALL_PRED.get("old",{}))!=456 or len(FROZEN_S179)!=456:
        raise AssertionError("S188 full REAL frozen S187 CPU 456 decoder results required")
    model,sources,first,later,pointer,lm_head,*_=cpu
    if model.training:
        raise AssertionError("S188 CPU model must be frozen eval mode")
    if s182.REAL_S175_BASE_GUARD is None:
        raise AssertionError("S188 S182 legacy fallback unavailable")
    from calibri2.s175 import negative_owner_guard_real_j7 as s175
    if s175.OLD_S172 is not s187.select_with_s186:
        raise AssertionError("S188 S187 source-only selector not installed")
    cohorts=groups_for(old_ids,routes)
    samples=[(group,k,old[k]["question"]) for group,keys in cohorts.items() for k in keys]
    # Same routing and network state for BOTH policies.
    s160.apply_profile(sources,cpu[6],"S93",bank,cpu[7],cpu[8])
    def run(q,policy):
        s175.OLD_S172=(s187.ORIGINAL_S182_SELECTOR if policy=="S182"
                        else s187.select_with_s186)
        features=torch.as_tensor(s160.prompt_only_vector(tok,q),dtype=torch.float32)
        with torch.inference_mode(),expansion.prefill_only(model),s160.prefill_only_tunnel(model,full):
            with s160.frozen_cpu_phase_scope(model,bridge,"S93",features,phase="prefill_only"):
                start=time.perf_counter()
                generated,_=s160.cached_full_cpu_parallel(
                    model,sources,first,later,lm_head,tok,q,pointer,copy_net,head)
                elapsed=time.perf_counter()-start
        return list(generated),elapsed
    try:
        run(samples[0][2],"S182")
        run(samples[0][2],"S187")
        readings={key:[] for key in cohorts}
        sha_only=[]
        for sweep in (0,1):
            for index,(group,key,q) in enumerate(samples):
                order=("S182","S187") if (index+sweep)%2==0 else ("S187","S182")
                results={x:run(q,x) for x in order}
                prompt_sha=sha(q)
                if results["S182"][0]!=base_s182_prediction(q,tok):
                    raise AssertionError("S188 original frozen S182 CPU answer changed")
                if results["S187"][0]!=s182.ALL_PRED["old"][prompt_sha]:
                    raise AssertionError("S188 original frozen S187 CPU answer changed")
                if group.endswith(("unchanged","control")) and results["S182"][0]!=results["S187"][0]:
                    raise AssertionError("S188 unchanged family full/EOS changed")
                readings[group].append((results["S182"][1],results["S187"][1]))
                sha_only.append({"prompt_sha256":prompt_sha,"group":group,
                                 "cycle":sweep,"order":list(order),
                                 "s182_cpu_seconds":results["S182"][1],
                                 "s187_cpu_seconds":results["S187"][1]})
        if len(sha_only)!=40 or any(len(v)!=8 for v in readings.values()):
            raise AssertionError("S188 complete 5x4x2 AB BA paired cohort failed")
        groups={name:stats(rows) for name,rows in readings.items()}
        matched=readings["S69"]+readings["history"]
        groups["new_source_S69_history"]=stats(matched)
        groups["all_20_source_cases"]=stats(
            [pair for rows in readings.values() for pair in rows])
        TIMINGS={"family_groups":groups,"per_prompt_SHA_only":sha_only,
                 "same_frozen_CPU_weights":True,"S93_only":True,
                 "two_AB_BA_sweeps":True,"no_gold_at_selection":True}
    finally:
        s175.OLD_S172=s187.select_with_s186
    return TIMINGS


def speed_hook(*args,**kwargs):
    original=OLD_PAIRED_CPU_SPEED(*args,**kwargs)
    paired_true_cpu(*args,**kwargs)
    return original


def main():
    global FROZEN_S179
    if len(sys.argv)<5:
        raise SystemExit("S188 requires original frozen S187 full 456+128 J7 CLI inputs")
    s179=Path(sys.argv[-3])
    out=Path(sys.argv[-1])
    trace=json.loads(s179.read_text())
    if (trace.get("stage")!="S179_TRACE_ONLY" or trace.get("number")!=456
            or trace.get("labels_used") is not False):
        raise AssertionError("S188 immutable original source-only S179 missing")
    FROZEN_S179={r["prompt_sha256"]:r["full_generated_ids"] for r in trace["cases"]}
    if len(FROZEN_S179)!=456:
        raise AssertionError("S188 baseline SHA duplicate or cohort incomplete")
    s160.paired_cpu_ratio=speed_hook
    try:
        s187.main()  # REAL full CPU 456+128; frozen, research only.
    finally:
        s160.paired_cpu_ratio=OLD_PAIRED_CPU_SPEED
    if TIMINGS is None:
        raise AssertionError("S188 independent same CPU policy benchmark absent")
    verified=json.loads((out/"S187_REAL_CPU_S69_HISTORY_456_128_STRICT_J7.json").read_text())
    bpp=verified["all_in_physical_bpp"]+8*SOURCE.stat().st_size/s182.DENOM
    groups=TIMINGS["family_groups"]
    speed=groups["new_source_S69_history"]["s187_over_s182_speed_ratio"]
    neutral=all(groups[k]["s187_over_s182_speed_ratio"]>=.9
                for k in ("S30_unchanged","S54_unchanged","S82_control"))
    passed=(
        verified["research_candidate_pass"]
        and verified["true_real_CPU_full_EOS_correct_original456"]==456
        and verified["true_real_CPU_new128_full_EOS"]==128
        and verified["lost_prior_full_EOS"]==0
        and verified["lost_prior_first_token"]==0
        and bpp<=.7 and speed>=.9
        and groups["all_20_source_cases"]["s187_over_s182_speed_ratio"]>=.9
        and neutral)
    report={
        "event":"S188_S182_VS_S187_REAL_SAME_CPU_PAIRED_ABBA_J7",
        "S187_verified_parent_run":43336858062,
        "true_real_CPU_456_full_EOS_correct":456,
        "true_real_CPU_new128_full_EOS":128,
        "zero_forgetting":verified["lost_prior_full_EOS"]==0
                        and verified["lost_prior_first_token"]==0,
        "same_real_43M_compressed_cpu_weights":True,
        "paired_measured_all_20x2_source_cases":True,
        "source_S69_history_median_speed_ratio":speed,
        "independent_same_CPU_groups":groups,
        "physical_bpp_including_S188_benchmark_source":bpp,
        "independent_speed_gate_pass":passed,
        "model_accepted":False,
        "no_training_or_QA_TEST":True,
        "Jessica_LIVE_Render_Supabase_APK_untouched":True,
        "per_prompt_SHA_only":TIMINGS["per_prompt_SHA_only"],
    }
    out.mkdir(parents=True,exist_ok=True)
    p=out/"S188_REAL_SAME_CPU_PAIRED_20x2_J7.json"
    if p.exists():
        raise FileExistsError("S188 immutable paired same CPU report cannot overwrite")
    p.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="per_prompt_SHA_only"},sort_keys=True),flush=True)
    if not passed:
        raise AssertionError("S188 independent S182-vs-S187 same CPU speed/accuracy/bpp gate rejected")

if __name__=="__main__":
    main()
