"""CALIBRI v203: read-only forensic of v202 *TRAIN-only* global rollback.

Consumes preexisting sealed v202 artifact. No model loading, training,
generation, threshold changes, DEV/VALID use, or new model tests.

A per-step guard checks one frozen pair per step, while the final gate
checks ALL 8 frozen pair rollouts and ALL 24 first-token paired TRAIN cases.
A sequence failure after individually allowed steps is an interaction or
coverage finding, not proof that a given layer alone caused it.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

EXPECTED_SOURCE_SHA="933daaecf9011f77e3705308faab9d6074b5fb7c4c847d5e1bdb830ad47b965a"


def _candidate_flags(guard):
    return {key: value for key, value in sorted(guard.items())
            if key.startswith("ACCEPT_") or key in (
                "diversity_regressed", "diversity_below_precommitted_floor",
                "prefix_regressed", "repetition_regressed",
                "v199_full_24_TRAIN_still_below_25pct_goal"
            )}


def diagnose(train, seal, final):
    if train.get("actual_pilot_steps")!=16 or len(train.get("steps",[]))!=16:
        raise AssertionError("Wrong v202 actual TRAIN-step count")
    if train.get("trained_original_135M_unique_matrices")!=211:
        raise AssertionError("Original 211-source architecture lost")
    if not train.get("QA_and_VALID_never_seen_for_TRAIN_acceptance"):
        raise AssertionError("Not an independent TRAIN-only acceptance")
    if not seal.get("v202_frozen_before_ANY_VALID_or_DEV"):
        raise AssertionError("Weight seal must precede all public QA")
    if final.get("source_SHA256")!=seal.get("sha256"):
        raise AssertionError("Final physical source differs from sealed")
    if final.get("source_SHA256")!=EXPECTED_SOURCE_SHA:
        raise AssertionError("Unexpected restored baseline source")
    if not final.get("final_guard_rolled_back"):
        raise AssertionError("Only v202 global rollback is being audited")
    if (train.get("all_pilot_mutations_rolled_back_on_full_TRAIN_fail") is not True
        or train.get("final_paid_5bit_net_changes")!=0
        or train.get("final_FP16_centroid_net_cells_changed")!=0):
        raise AssertionError("v202 rollback and physical pay do not agree")
    first=train["immutable_v186_full_TRAIN_guard"]
    sequence=train["v202_immutable_greedy_full_sequence_TRAIN_guard"]
    first_failed=first["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"] is False
    sequence_failed=sequence["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"] is False
    if not (first_failed or sequence_failed):
        raise AssertionError("Rollback reported but all final gates pass")
    step_names=set()
    step_report=[]
    accepted=0
    by_rejection=Counter()
    for i, row in enumerate(train["steps"]):
        if row["step"]!=i+1:
            raise AssertionError("Out-of-order actual step")
        if row["operator"] in step_names:
            raise AssertionError("Duplicate scheduled operator in pilot")
        step_names.add(row["operator"])
        if not row.get("never_consulted_VALID_or_DEV_to_accept"):
            raise AssertionError("DEV/VALID leak into per-step acceptance")
        g=row["TRAIN_guard"]; seq=row["per_step_TRAIN_sequence_guard"]
        g_pass=g["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"]
        seq_pass=seq["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"]
        loss_ok=row["TRAIN_no_more_than_1percent_loss_increase"]
        proposed_keep=g_pass and seq_pass and loss_ok
        if proposed_keep != row["source_step_accepted"]:
            raise AssertionError("Per-step acceptance predicate changed")
        accepted+=int(proposed_keep)
        reasons=[]
        if not g_pass: reasons.append("frozen_8pair_first_token_guard")
        if not seq_pass: reasons.append("frozen_one_pair_full_sequence_guard")
        if not loss_ok: reasons.append("TRAIN_objective_1percent_gate")
        for reason in reasons:by_rejection[reason]+=1
        step_report.append({
            "step":row["step"],"operator":row["operator"],
            "accepted_before_global_rollback":proposed_keep,
            "rejection_reasons":reasons,
            "TRAIN_first_guard_flags":_candidate_flags(g),
            "TRAIN_sequence_guard_flags":_candidate_flags(seq),
            "TRAIN_sequence_repetition_change":(
                seq["adjacent_repeats_after"]-seq["adjacent_repeats_before"]),
            "TRAIN_sequence_correct_prefix_change":(
                seq["correct_prefix_after"]-seq["correct_prefix_before"])
        })
    if accepted != train["accepted_steps_before_full_TRAIN_guard"]:
        raise AssertionError("Accepted step summary mismatches step records")
    if accepted != final["guard_accepted_step_count"]:
        raise AssertionError("v202 report disagrees on training updates")
    raw_first_failures = {
        "lost_first_correct":first["lost_previous_correct_TRAIN_first_tokens"],
        "lost_opposite_pairs":first["lost_original_both_correct_opposite_TRAIN_pairs"],
        "domain_regressions":first["correct_by_domain_regressions"],
        "diversity_regressed":first["diversity_regressed"],
        "diversity_below_precommitted_floor":first["diversity_below_precommitted_floor"]
    }
    raw_sequence_failures={
        "lost_exact_TRAIN_answers":sequence["lost_original_complete_TRAIN_answers"],
        "correct_prefix_before":sequence["correct_prefix_before"],
        "correct_prefix_after":sequence["correct_prefix_after"],
        "prefix_regressed":sequence["prefix_regressed"],
        "adjacent_repeats_before":sequence["adjacent_repeats_before"],
        "adjacent_repeats_after":sequence["adjacent_repeats_after"],
        "repetition_regressed":sequence["repetition_regressed"],
        "exact_domain_regressions":sequence["complete_exact_domain_regressions"]
    }
    return {
        "stage":"v203_v202_postmortem",
        "complete":True,
        "source_changed":False,
        "read_only":True,
        "new_model_tests":0,
        "final_rollback_confirmed":True,
        "trained_135M_physical_bpp":final["whole_original_source_bpp"],
        "actual_retained_source_sha256":final["source_SHA256"],
        "accepted_local_steps_but_all_reverted":accepted,
        "local_rejected_reasons":dict(by_rejection),
        "final_first_token_gate_failed":first_failed,
        "final_sequence_gate_failed":sequence_failed,
        "full_frozen_24pair_first_token_failures":raw_first_failures,
        "full_frozen_8pair_sequence_failures":raw_sequence_failures,
        "steps":step_report,
        "do_not_infer_causality_from_public_DEV":True,
        "no_LIVE_Jessica_main_Render_Supabase_APK_changes":True
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("v202_artifacts",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    folder=args.v202_artifacts
    names={
        "training":"ACTUAL_guarded_TRAIN_16_step_audit.json",
        "seal":"SEALED_paid_source_before_untouched_QA.json",
        "final":"v202_real_NATIVE_SOURCE_QA_summary.json"
    }
    raw={name:(folder/filename).read_bytes() for name,filename in names.items()}
    parsed={name:json.loads(blob.decode("utf8")) for name,blob in raw.items()}
    report=diagnose(parsed["training"],parsed["seal"],parsed["final"])
    report["inputs_sha256"]={name:hashlib.sha256(blob).hexdigest()
                             for name,blob in sorted(raw.items())}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"v203_v202_rollback_forensic.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf8")
    print(json.dumps({
        k:v for k,v in report.items() if k!="steps"
    },ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
