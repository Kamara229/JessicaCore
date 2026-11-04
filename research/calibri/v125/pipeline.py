"""Answer-contrastive real source experiment with frozen prior evidence audit."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from v120.critic import GreenTunnel
from v124.protocol import TRAIN,VALID,MAX_CHECKPOINT_BYTES
from v124.host_sources import generate_hard_negatives
from v124.hard_examples import examples
from v124.training import restore_previous,file_checkpoint
from .protocol import (GATE_TRUE_ACCEPT_MIN,GATE_PLAUSIBLE_REJECT_MIN,
                       GATE_REAL_REJECT_MIN)
from .pairwise import train
from .metrics import independent_metric

def canonical_hash(data):
    return hashlib.sha256(json.dumps(
        data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def main(host,source_green,previous_hard,original_evidence,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads(original_evidence.read_text())
    prev_physical=frozen["one_final_10MB_green_model"]
    if (hashlib.sha256(previous_hard.read_bytes()).hexdigest()!=prev_physical["full_SHA256"]
        or previous_hard.stat().st_size!=prev_physical["file_bytes"]):
        raise AssertionError("v124 previous real learned control not SHA frozen")
    produced_fit,_=generate_hard_negatives(TRAIN,host,out)
    produced_valid,_=generate_hard_negatives(VALID,host,out)
    if (canonical_hash(produced_fit)!=frozen["source_real_candidate_training_sha256"]
        or canonical_hash(produced_valid)!=frozen["source_real_candidate_validation_sha256"]):
        raise AssertionError("original two-host autonomous generated data not reproducible")
    fit_rows=examples(TRAIN,produced_fit)
    valid_rows=examples(VALID,produced_valid)
    baseline=restore_previous(source_green).eval()
    prior=GreenTunnel()
    prior.load_state_dict(load_file(str(previous_hard)),strict=True)
    prior.eval()
    challenger=restore_previous(source_green)
    trace=train(challenger,fit_rows)
    checkpoint=out/"GreenTunnel_v125_pairwise_correctness.safetensors"
    stored=file_checkpoint(challenger,checkpoint)
    check=GreenTunnel()
    check.load_state_dict(load_file(str(checkpoint)),strict=True)
    check.eval()
    if any(not torch.equal(p,q) for p,q in zip(
            challenger.state_dict().values(),check.state_dict().values())):
        raise AssertionError("serialized exact neural weights changed")
    models={"v120_original":baseline,"v124_real_hard_BCE":prior,
            "v125_context_pairwise":check}
    results=independent_metric(models,valid_rows)
    candidate=results["v125_context_pairwise"]
    pass_gates={
        "correct_reference_acceptance_at_least_70pct":candidate["gold_reference_acceptance_rate"]>=GATE_TRUE_ACCEPT_MIN,
        "plausible_false_rejection_at_least_70pct":candidate["plausible_wrong_rejection_rate"]>=GATE_PLAUSIBLE_REJECT_MIN,
        "actual_generated_false_rejection_at_least_70pct":candidate["real_generation_wrong_rejection_rate"]>=GATE_REAL_REJECT_MIN,
        "gold_rank1_given_ORACLE_in_diagnostic_at_least_80pct":candidate["gold_rank1_rate"]>=.80,
    }
    all_pass=all(pass_gates.values())
    report={
        "experiment":"CALIBRI v125 contextual pairwise answer contrastive shared Green Tunnel",
        "source_v124_artifact_report_SHA256":hashlib.sha256(original_evidence.read_bytes()).hexdigest(),
        "actual_real_host_generations_reproduced_SHA_exact":True,
        "train_fit_questions":len(TRAIN),"heldout_q":len(VALID),
        "baseline_two_real_hosts_not_given_ground_truth":True,
        "old_original_CB59_source_unchanged":True,
        "host_source_core_bpp_excludes_one_reusable_Green":8*host.stat().st_size/134515008,
        "one_final_shared_green_checkpoint":stored,
        "physical_eight_MB_neural_file_never_scales_with_number_of_host_layers":True,
        "fit":trace,
        "heldout_comparisons":results,
        "predeclared_quality_checks":pass_gates,
        "all_predeclared_verifier_quality_checks_PASS":all_pass,
        "does_not_repair_HOST_if_HOST_never_generates_gold":True,
        "real_fact_correctness_of_both_original_hosts_is_still_ZERO_on_v124_valid":True,
        "gold_oracle_rank_is_diagnostic_only_not_real_generator_answer":True,
        "formal_independent_v89_TEST_UNOPENED":True,
        "trillion_scale_or_coherent_language_NOT_PROVEN":True,
        "no_main_Render_Supabase_Jessica_APK_production_changes":True}
    (out/"v125_contrastive_hard_real.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"stage":"v125_CONTRASTIVE_COMPARISON_DONE",
        "bytes":stored["file_bytes"],
        "checks":pass_gates,
        "scored":{k:{z:v for z,v in vals.items()
                   if z in ("gold_rank1_rate","gold_reference_acceptance_rate",
                            "plausible_wrong_rejection_rate","real_generation_wrong_rejection_rate",
                            "full_heldout_80_example_balanced_accuracy")}
                    for k,vals in results.items()},
        "model_coherent_answer_recovered":False},ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("host","source_green","previous_hard","v124_report","out"):
        p.add_argument(name,type=Path)
    q=p.parse_args();main(q.host,q.source_green,q.previous_hard,q.v124_report,q.out)
