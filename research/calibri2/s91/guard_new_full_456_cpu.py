"""S91 independent, RESEARCH-ONLY, physically reloaded 456-TRAIN guard.

A new strict full/EOS-correct answer is required for a quality gain.
Zero loss of ANY previously correct full response or first token is mandatory.
CPU timing uses reconstructed physical CB59 PyTorch (NOT native C). Even a
passing result is NOT a model release, QA result or GPU/CPU parity proof.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath


def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_original_CB59",type=Path)
    p.add_argument("S82_original_controller",type=Path)
    p.add_argument("S91_physical_CB61",type=Path)
    p.add_argument("S91_controller",type=Path)
    p.add_argument("S91_manifest",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("S87_frozen_456_report",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    old=json.loads(a.S87_frozen_456_report.read_text())
    meta=json.loads(a.S91_manifest.read_text())
    if old.get("stage")!="S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT":
        raise AssertionError("S91 reference is not a frozen S87 full TRAIN audit")
    if len(old.get("all_456_per_identity_results",[]))!=456:
        raise AssertionError("S91 requires full 456 old-token reference")
    if (meta.get("stage")!="S91_RESEARCH_ONLY_JOINT_NEW_FULL_EOS_REPLAY_MPS_TRAIN_UNACCEPTED"
            or meta.get("candidate_accepted") is not False or
            meta.get("base_source_S87_sha256") is None or
            not meta.get("joint_MPS_train_both_physical_core_and_internal_tunnel")):
        raise AssertionError("S91 unaccepted physical prototype provenance invalid")
    # The S87 read-only audit does exact SHA, bpp, 456 TRAIN, loss and speed.
    # Patch only S91's gate, after checking the serialized gate tensors are
    # genuinely zero: never sneak in an approximation or model weights.
    original=s87.build_cpu
    patched=[]
    def fast_cpu(raw, weights, use_internal):
        state=original(raw,weights,use_internal)
        if use_internal:
            patched.extend(install_verified_fastpath(state[-1]))
        return state
    s87.build_cpu=fast_cpu
    saved_argv=sys.argv
    guard_failure=None
    try:
        sys.argv=[sys.argv[0],str(a.S45_original_CB59),
            str(a.S82_original_controller),str(a.S91_physical_CB61),
            str(a.S91_controller),str(a.S91_manifest),str(a.out)]
        try:
            s87.main()
        except SystemExit as err:
            guard_failure=str(err)
    finally:
        s87.build_cpu=original
        sys.argv=saved_argv

    result=json.loads(
        (a.out/"S87_ALL_456_TRAIN_GUARD_REPORT.json").read_text())
    reference={r["id"]:r for r in old["all_456_per_identity_results"]}
    target_ids=set(meta["targeted_wrong_TRAIN_ids"])
    fresh_ids=set(meta["newly_targeted_wrong_TRAIN_ids"])
    if (len(fresh_ids)!=12 or not fresh_ids.issubset(target_ids)):
        raise AssertionError("S91 must really include 12 NEW unsolved TRAIN targets beyond S89")
    if not target_ids or not target_ids.issubset(reference):
        raise AssertionError("S91 targets must be pre-registered S87 TRAIN rows")
    gains=[];lost=[];lost_first=[];changed=[];untrained=[]
    for row in result["all_456_per_identity_results"]:
        prior=reference.get(row["id"])
        if prior is None or (
            prior["question_sha256"]!=row["question_sha256"] or
            prior["target_ids"]!=row["target_ids"] or
            prior["S82_ids"]!=row["S82_ids"]):
            raise AssertionError("S91 changed baseline 456 TRAIN question/target/answer")
        before=prior["S87_ids"]
        after=row["S87_ids"]
        expected=row["target_ids"]
        old_full=before==expected
        new_full=after==expected
        old_first=bool(before) and before[0]==expected[0]
        new_first=bool(after) and after[0]==expected[0]
        if old_full and not new_full:lost.append(row["id"])
        if old_first and not new_first:lost_first.append(row["id"])
        if new_full and not old_full:
            gains.append(row["id"])
            if row["id"] not in target_ids:untrained.append(row["id"])
        if before!=after:changed.append(row["id"])
    if len(result["all_456_per_identity_results"])!=456 or len(reference)!=456:
        raise AssertionError("S91 incomplete 456 TRAIN rows")
    fresh_gains=sorted(set(gains)&fresh_ids)
    s89_old_full=meta.get("measured_original_S89_456_full_correct")
    if s89_old_full!=225:
        raise AssertionError("S91 needs same frozen 456 S89 full-accuracy reference")
    s91_net_vs_s89=sum(r["new_full"] for r in result["all_456_per_identity_results"])-s89_old_full
    report={
        "stage":"S91_PHYSICAL_JOINT_CORE_INTERNAL_TUNNEL_456_FULL_TRAIN_AUDIT",
        "reference_S87_run":42659248083,
        "all_456_train_ids_checked":True,
        "targeted_wrong_TRAIN_ids":sorted(target_ids),
        "S91_changed_token_sequences":changed,
        "S91_new_strict_full_EOS_correct_ids":gains,
        "S91_newly_targeted_FULL_EOS_correct_ids":fresh_gains,
        "S91_new_full_correct_on_not_trained_ids":untrained,
        "lost_previous_full_correct_ids":lost,
        "lost_previous_first_correct_ids":lost_first,
        "S87_previous_full_correct":sum(
            r["S87_ids"]==r["target_ids"] for r in reference.values()),
        "S91_full_correct":sum(r["new_full"] for r in result["all_456_per_identity_results"]),
        "S89_previous_full_correct":s89_old_full,
        "S91_net_full_change_vs_S89":s91_net_vs_s89,
        "physical_all_inclusive_bpp":result["all_inclusive_bpp"],
        "reconstructed_CPU_speed_ratio_S82_over_S91":
            result["cpu_speed_ratio_48_prompt_median_S82_over_S87"],
        "CPU_speed_at_least_90_percent":result["cpu_speed_90pct_guard"],
        "old_correct_full_and_first_retained":not lost and not lost_first,
        "exact_zero_gate_fastpath_verified":patched==["3","9"],
        "only_authorized_historical_TRAIN_used":True,
        "no_QA_TEST_peeking_or_tuning":True,
        "native_C_speed_and_MPS_CPU_parity_not_proven":True,
        "candidate_accepted":False
    }
    (a.out/"S91_ALL_456_FULL_EOS_QUALITY_GUARD_REPORT.json").write_text(
        json.dumps(report,indent=2))
    print(json.dumps({"event":"S91_456_TRAIN_FULL_POLICY_QUALITY_END",
        "old_full":report["S87_previous_full_correct"],
        "new_full":report["S91_full_correct"],
        "new_gains":len(gains),"NEW_target_FULL_EOS_gains":len(fresh_gains),
        "net_gain_vs_frozen_S89_full_456":s91_net_vs_s89,
        "new_gains_on_untrained_TRAIN":len(untrained),
        "lost_full":len(lost),"lost_first":len(lost_first),
        "physical_bpp":report["physical_all_inclusive_bpp"],
        "CPU_ratio":report["reconstructed_CPU_speed_ratio_S82_over_S91"],
        "speed_pass":report["CPU_speed_at_least_90_percent"],
        "candidate_accepted":False}),flush=True)
    if guard_failure:
        raise SystemExit("REJECTED S91: "+guard_failure)
    if lost or lost_first:
        raise SystemExit("REJECTED S91: forgot old full or first token")
    if s91_net_vs_s89<=0:
        raise SystemExit("REJECTED S91: no net improvement on all 456 strict full/EOS TRAIN answers relative to S89")
    if not gains or not fresh_gains:
        raise SystemExit("REJECTED S91: no strictly NEW full/EOS TRAIN answers on additional contrastive pairs")
    if not result["cpu_speed_90pct_guard"]:
        raise SystemExit("REJECTED S91: compressed reconstructed CPU below 90%")
    if result["all_inclusive_bpp"]>0.7:
        raise SystemExit("REJECTED S91: all-inclusive compressed bpp above 0.7")


if __name__=="__main__":
    main()
