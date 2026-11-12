"""S152: memory-safe one macOS runner, CPU+GPU parallel, two real GPU Skill trials sequenced.

All three workers individually load original frozen S142 weights and use
original S101/S122 TRAIN. The two GPU Skills run ONE AT A TIME to protect Apple MPS memory, while
the CPU completion Skill is independent. Each worker performs its own
physical FP16 reload, all 456+128 exact greedy+EOS retention, bpp and CPU
speed assessment, and J7 rejection/optimizer rollback. This orchestrator
NEVER automatically joins/reloads/recommits rejected Skills.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

MAX_SECONDS=6900
WORKERS=(
    ("binding_gpu","mps","research/calibri2/s148/protected_replay_neural_skill.py","S148"),
    ("language_gpu","mps","research/calibri2/s151/language_eos_gpu.py","S152"),
    ("completion_cpu","cpu","research/calibri2/s149/cpu_completion_neural_skill.py","S149"),
)


def candidate_is_safe(report):
    """Defence in depth: full J7, positive original TRAIN gain and physical limits."""
    if not isinstance(report,dict):
        return False
    audit=report.get("full456_and_new128") or {}
    physical=report.get("physical") or {}
    speed=report.get("paired_CPU") or {}
    losses=("lost_old_full_ids","lost_old_first_ids",
            "lost_new_full_ids","lost_new_first_ids")
    return (
        report.get("strict_no_forgetting") is True and
        report.get("TRAIN_only_skill_gain") is True and
        report.get("research_safe_development_candidate") is True and
        report.get("physical_safetensors_reloaded") is True and
        all(audit.get(k)==[] for k in losses) and
        physical.get("all_in_physical_bpp",float("inf"))<=0.7 and
        speed.get("ratio",0)>=0.90 and
        report.get("model_accepted") is False
    )


def ps_rss_bytes(pid: int):
    try:
        r=subprocess.run(["ps","-o","rss=","-p",str(pid)],
                         text=True,capture_output=True,timeout=3,check=False)
        return int(r.stdout.strip().split()[0])*1024 if r.returncode==0 and r.stdout.strip() else None
    except (OSError,ValueError,subprocess.TimeoutExpired,IndexError):
        return None


def run_shared_host(paths:list[str],out:Path,*,max_seconds:int=MAX_SECONDS):
    out.mkdir(parents=True,exist_ok=True)
    if len(paths)!=22:
        raise AssertionError(f"S152 requires 22 immutable original input paths, got {len(paths)}")
    if not all(Path(x).exists() for x in paths):
        raise FileNotFoundError("S152 missing pinned original input")
    # The parent SHA and original full validation happen independently in each
    # child; even a successful joint run never claims model acceptance.
    state={}
    file_handles=[]
    start=time.monotonic()
    def spawn_one(entry):
        name,device,source,stage=entry
        if name in state:
            raise AssertionError("S152 duplicate trainer spawn")
        if device=="mps" and any(
            q["device"]=="mps" and q["returncode"] is None for q in state.values()):
            raise AssertionError("S152 will not concurrently load two real S142 MPS models")
        destination=out/name
        destination.mkdir(exist_ok=True)
        log=destination/"training.log"
        f=log.open("w",buffering=1)
        file_handles.append(f)
        env=os.environ.copy()
        env.update({
            "PYTHONUNBUFFERED":"1",
            "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
            "OPENBLAS_NUM_THREADS":"1","TOKENIZERS_PARALLELISM":"false",
            "PYTORCH_ENABLE_MPS_FALLBACK":"0",
        })
        # Do not cap a real 43M GPU model to S151's unproven 55% memory limit;
        # prevent two GPU backbones occupying unified memory simultaneously.
        env.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO",None)
        env.pop("PYTORCH_MPS_LOW_WATERMARK_RATIO",None)
        proc=subprocess.Popen(
            [sys.executable,"-u",source,*paths,str(destination)],
            stdout=f,stderr=subprocess.STDOUT,env=env)
        state[name]={
            "device":device,"source":source,"stage":stage,"pid":proc.pid,
            "process":proc,"started_at_elapsed_seconds":time.monotonic()-start,
            "ended_at_elapsed_seconds":None,
            "returncode":None,"log":str(log),
            "peak_process_rss_bytes":None,
        }
        print("S152_SPAWN",json.dumps({"skill":name,"device":device,
                                     "pid":proc.pid}),flush=True)
    try:
        # Start first GPU and independent CPU simultaneously.
        spawn_one(WORKERS[0])
        spawn_one(WORKERS[2])
        deadline=time.monotonic()+max_seconds
        # Second GPU Skill starts only after first MPS process exits.
        while len(state)<len(WORKERS) or any(s["returncode"] is None for s in state.values()):
            for name,s in state.items():
                if s["returncode"] is not None:
                    continue
                rss=ps_rss_bytes(s["pid"])
                if rss is not None:
                    s["peak_process_rss_bytes"]=max(
                        s["peak_process_rss_bytes"] or 0,rss)
                rc=s["process"].poll()
                if rc is not None:
                    s["returncode"]=rc
                    s["ended_at_elapsed_seconds"]=time.monotonic()-start
                    print("S152_EXIT",json.dumps({
                        "skill":name,"rc":rc,"elapsed":s["ended_at_elapsed_seconds"]}),
                        flush=True)
            if "language_gpu" not in state and state["binding_gpu"]["returncode"] is not None:
                # If binding failed, still exercise independent language Skill
                # with the intact immutable S142 source; no failed Skill is reused.
                spawn_one(WORKERS[1])
            if time.monotonic()>deadline:
                for s in state.values():
                    if s["returncode"] is None:
                        s["process"].terminate()
                raise TimeoutError("S152 real one-host GPU/CPU training budget exhausted")
            if len(state)<len(WORKERS) or any(s["returncode"] is None for s in state.values()):
                time.sleep(2.0)
        for f in file_handles:
            f.flush()
        workers={}
        for name,s in state.items():
            report_file=out/name/(s["stage"]+"_REAL_NEURAL_SKILL_456_128_AUDIT.json")
            report=json.loads(report_file.read_text()) if report_file.exists() else None
            audit=(report or {}).get("full456_and_new128") or {}
            workers[name]={
                "device":s["device"],"pid":s["pid"],"source":s["source"],
                "exit_code":s["returncode"],
                "started_at_elapsed_seconds":s["started_at_elapsed_seconds"],
                "ended_at_elapsed_seconds":s["ended_at_elapsed_seconds"],
                "peak_process_rss_bytes":s["peak_process_rss_bytes"],
                "log_path":s["log"],
                "physical_report_path":str(report_file) if report else None,
                "individual_research_safe":candidate_is_safe(report),
                "rolled_back":(report or {}).get("J7_guarded_skill_rollback_verified"),
                "old_full_before":audit.get("old_full_before"),
                "old_full_after":audit.get("old_full_after"),
                "new_full_before":audit.get("new_full_before"),
                "new_full_after":audit.get("new_full_after"),
                "lost_old_full":len(audit.get("lost_old_full_ids",[])) if report else None,
                "lost_new_full":len(audit.get("lost_new_full_ids",[])) if report else None,
                "bpp":((report or {}).get("physical") or {}).get("all_in_physical_bpp"),
                "cpu_speed_ratio":((report or {}).get("paired_CPU") or {}).get("ratio"),
            }
        all_complete=all(s["exit_code"]==0 for s in workers.values())
        all_safe=all(s["individual_research_safe"] for s in workers.values())
        report={
            "stage":"S152_MEMORY_SAFE_GPU_STAGGERED_CPU_OVERLAP_REAL_S142",
            "shared_runner":True,
            "attempted_two_distinct_GPU_skills_and_one_CPU_skill":True,
            "only_one_MPS_model_process_at_a_time":True,
            "cpu_and_first_gpu_processes_spawned_concurrently":True,
            "training_overlap_not_independently_instrumented":True,
            "all_training_and_physical_audits_finished":all_complete,
            "every_skill_individually_safe":all_safe,
            "joint_456_128_guard_not_run":True,
            "joint_compatible_accepted":False,
            "model_accepted":False,
            "source_S142_parent_immutable":True,
            "no_heldout_QA":True,"no_LIVE_changes":True,
            "wall_seconds":time.monotonic()-start,
            "workers":workers,
        }
        (out/"S152_ALL_REAL_PARALLEL_SKILLS_AUDIT.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2))
        print("S152_REAL_THREE_RESULT",json.dumps({
            "all_completed":all_complete,
            "individually_safe":[name for name,w in workers.items()
                                 if w["individual_research_safe"]],
            "workers":{name:{
                "rc":w["exit_code"],"old_after":w["old_full_after"],
                "new_after":w["new_full_after"],"bpp":w["bpp"],
                "cpu_ratio":w["cpu_speed_ratio"],"rolled_back":w["rolled_back"]}
                for name,w in workers.items()},"model_accepted":False
        }),flush=True)
        if not all_complete:
            raise RuntimeError("S152 at least one actual trainer failed: see per-worker logs and reports")
        return report
    except BaseException as e:
        (out/"S152_FAILED_ORCHESTRATOR.json").write_text(json.dumps({
            "failure":repr(e),"traceback":traceback.format_exc(),
            "model_accepted":False,
            "workers":{k:{x:y for x,y in v.items() if x!="process"}
                       for k,v in state.items()},
        },default=str,ensure_ascii=False,indent=2))
        raise
    finally:
        for s in state.values():
            p=s["process"]
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
        for f in file_handles:
            f.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("arguments",nargs="+")
    args=ap.parse_args()
    if len(args.arguments)!=23:
        raise AssertionError(f"S152 expect 22 input paths + one output dir, got {len(args.arguments)}")
    run_shared_host(args.arguments[:-1],Path(args.arguments[-1]))


if __name__=="__main__":
    main()
