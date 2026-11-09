"""S93: independent physical full/EOS 456 TRAIN CPU union guard.

S87 and S91 were both separately measured. Strict protection is the
union of their correctly generated full answers AND first tokens.
Even passing TRAIN is not sealed QA/generalization or a LIVE release.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s91 import train_joint_quality_rehearsal as s91


def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_original_CB59",type=Path)
    p.add_argument("S82_original_controller",type=Path)
    p.add_argument("S93_physical_CB61",type=Path)
    p.add_argument("S93_controller",type=Path)
    p.add_argument("S93_manifest",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("S87_frozen_456",type=Path)
    p.add_argument("S91_frozen_456",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    s87_original=json.loads(a.S87_frozen_456.read_text())
    s91_original=json.loads(a.S91_frozen_456.read_text())
    meta=json.loads(a.S93_manifest.read_text())
    if (meta.get("stage")!="S93_RESEARCH_ONLY_PROJECTED_JOINT_S91_REPAIR_MPS_UNACCEPTED" or
        meta.get("candidate_accepted") is not False or
        meta.get("union_of_previously_full_correct")!=282 or
        meta.get("S87_source_sha256") is None or
        meta.get("base_source_S91_sha256") is None):
        raise AssertionError("S93 physical model manifest wrong provenance")
    if (s87_original.get("stage")!="S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT" or
        s91_original.get("stage")!="S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT"):
        raise AssertionError("S93 requires frozen complete S87 and S91 CPU reports")
    if (len(s87_original.get("all_456_per_identity_results",[]))!=456 or
        len(s91_original.get("all_456_per_identity_results",[]))!=456):
        raise AssertionError("S93 missing frozen per-identity report")
    old_build=s87.build_cpu
    patched=[]
    def build_with_proven_fastpath(raw, side, active):
        out=old_build(raw,side,active)
        if active:
            patched.extend(install_verified_fastpath(out[-1]))
        return out
    s87.build_cpu=build_with_proven_fastpath
    orig_argv=sys.argv
    failure=None
    try:
        sys.argv=[sys.argv[0],str(a.S45_original_CB59),
          str(a.S82_original_controller),str(a.S93_physical_CB61),
          str(a.S93_controller),str(a.S93_manifest),str(a.out)]
        try:s87.main()
        except SystemExit as exc:failure=str(exc)
    finally:
        s87.build_cpu=old_build
        sys.argv=orig_argv
    result=json.loads((a.out/"S87_ALL_456_TRAIN_GUARD_REPORT.json").read_text())
    old_a={x["id"]:x for x in s87_original["all_456_per_identity_results"]}
    old_b={x["id"]:x for x in s91_original["all_456_per_identity_results"]}
    new_a={x["id"]:x for x in result["all_456_per_identity_results"]}
    if not (len(old_a)==len(old_b)==len(new_a)==456 and
            set(old_a)==set(old_b)==set(new_a)):
        raise AssertionError("S93 incomplete immutable 456 cohort")
    protected_full=[]
    protected_first=[]
    newly_correct=[]
    lost_full=[]
    lost_first=[]
    for key in sorted(new_a):
        p,q,r=old_a[key],old_b[key],new_a[key]
        if any(z["question_sha256"]!=p["question_sha256"] or
               z["target_ids"]!=p["target_ids"] or
               z["S82_ids"]!=p["S82_ids"] for z in (q,r)):
            raise AssertionError("S93 frozen question/target/original output drift: "+key)
        goal=p["target_ids"]
        old_full=p["S87_ids"]==goal
        s91_full=q["S87_ids"]==goal
        new_full=r["S87_ids"]==goal
        old_first=bool(p["S87_ids"]) and p["S87_ids"][0]==goal[0]
        s91_first=bool(q["S87_ids"]) and q["S87_ids"][0]==goal[0]
        new_first=bool(r["S87_ids"]) and r["S87_ids"][0]==goal[0]
        if old_full or s91_full:
            protected_full.append(key)
            if not new_full:lost_full.append(key)
        if old_first or s91_first:
            protected_first.append(key)
            if not new_first:lost_first.append(key)
        if new_full and not (old_full or s91_full):newly_correct.append(key)
    if len(protected_full)!=282:
        raise AssertionError("S93 baseline union 282 changed")
    full_correct=sum(r["S87_ids"]==r["target_ids"] for r in new_a.values())
    score={
       "stage":"S93_RESEARCH_ONLY_456_TRAIN_STRICT_PHYSICAL_UNION_FULL_EOS_AUDIT",
       "all_456_train_ids_checked":True,
       "protected_union_full_answer_count":len(protected_full),
       "protected_union_first_token_count":len(protected_first),
       "S93_full_correct":full_correct,
       "beyond_union_new_strict_full_EOS_correct_ids":newly_correct,
       "lost_S87_or_S91_previously_full_correct_ids":lost_full,
       "lost_S87_or_S91_previously_first_correct_ids":lost_first,
       "physical_all_inclusive_bpp":result["all_inclusive_bpp"],
       "reconstructed_CPU_speed_ratio_S82_over_S93":
          result["cpu_speed_ratio_48_prompt_median_S82_over_S87"],
       "CPU_speed_at_least_90_percent":result["cpu_speed_90pct_guard"],
       "physical_exact_zero_gate_fastpath_verified":patched==["3","9"],
       "all_train_output_union_retained":not lost_full and not lost_first,
       "sealed_QA_TEST_not_accessed":True,
       "LIVE_untouched":True,
       "candidate_accepted":False
    }
    (a.out/"S93_ALL_456_UNION_STRICT_FULL_EOS_REPORT.json").write_text(
       json.dumps(score,indent=2,sort_keys=True))
    print(json.dumps({"event":"S93_ALL456_PHYSICAL_TRAIN_UNION_GUARD",
        "full_correct":full_correct,"union_protected":len(protected_full),
        "new_full_beyond_union":len(newly_correct),
        "lost_full":len(lost_full),"lost_first":len(lost_first),
        "bpp":score["physical_all_inclusive_bpp"],
        "CPU_ratio":score["reconstructed_CPU_speed_ratio_S82_over_S93"],
        "candidate_accepted":False}),flush=True)
    if failure:
        raise SystemExit("REJECTED S93: "+failure)
    if lost_full or lost_first:
        raise SystemExit("REJECTED S93: S87/S91 union forgotten")
    if not newly_correct or full_correct<=282:
        raise SystemExit("REJECTED S93: no STRICT new full/EOS gain beyond protected union")
    if not score["CPU_speed_at_least_90_percent"] or not score["physical_exact_zero_gate_fastpath_verified"]:
        raise SystemExit("REJECTED S93: CPU speed/fastpath guard")
    if score["physical_all_inclusive_bpp"]>0.7:
        raise SystemExit("REJECTED S93: compressed bpp")


if __name__=="__main__":
    main()
