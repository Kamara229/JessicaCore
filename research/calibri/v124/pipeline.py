"""One frozen <=10MB neural Green Tunnel trained on REAL failed responses.

Do not conflate "rejects untruthful answers" with "repairs language".
Results show if negative-source awareness improves abstention on 14M + 135M.
"""
import argparse,hashlib,json,random,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from v120.critic import GreenTunnel
from .protocol import TRAIN,VALID,SEED
from .host_sources import generate_hard_negatives
from .hard_examples import examples,hard_samples,positive_and_negative,canonical
from .training import (restore_previous,simple_synthetic,train,infer,grade,
                       file_checkpoint)

def evaluate_all(models,valid_rows):
    report={}
    for name,model in models.items():
        predictions=infer(model,valid_rows)
        groups={"all":grade(valid_rows,predictions),
                "reference_gold":grade(valid_rows,predictions,{"independently_correct_reference"}),
                "plausible_wrong":grade(valid_rows,predictions,{"plausible_false_reference"})}
        for host in ("CB59_v119","Pythia14M"):
            subset=f"actual_generated_{host}"
            records=[(r,p) for r,p in zip(valid_rows,predictions) if r[3]==subset]
            bad=[p for (q,a,y,k),p in records if not y]
            good=[p for (q,a,y,k),p in records if y]
            groups[host]={
                "real_candidate_count":len(records),
                "genuinely_wrong_generated_count":len(bad),
                "correct_host_answers_count":len(good),
                "wrong_generated_rejected":sum(p<.5 for p in bad),
                "wrong_generated_false_accepted":sum(p>=.5 for p in bad),
                "wrong_generated_rejection_rate":sum(p<.5 for p in bad)/len(bad) if bad else None,
                "correct_generated_accepted":sum(p>=.5 for p in good),
                "mean_wrong_generated_confidence":sum(bad)/len(bad) if bad else None,
                "real_candidate_responses":[{"question":r[0],"generated":r[1][:130],
                    "is_true_answer":bool(r[2]),"score":p,"would_accept":p>=.5}
                    for r,p in records],
            }
        report[name]=groups
    return report

def run(source,critic_source,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    # Gold is NOT in host generator call; grades and critic TRAIN receive it.
    train_candidates,train_source=generate_hard_negatives(TRAIN,source,out)
    valid_candidates,valid_source=generate_hard_negatives(VALID,source,out)
    train_rows=examples(TRAIN,train_candidates)
    valid_rows=examples(VALID,valid_candidates)
    assert len(train_rows)==4*len(TRAIN) and len(valid_rows)==4*len(VALID)
    assert not {q for q,a,label,k in train_rows}&{q for q,a,label,k in valid_rows}
    baseline=restore_previous(critic_source).eval()
    control=restore_previous(critic_source)
    control_curve=train(control,simple_synthetic(train_rows),seed=SEED)
    proper=restore_previous(critic_source)
    curve=train(proper,train_rows,seed=SEED)
    checkpoint=out/"GreenTunnel_v124_REAL_host_failures.safetensors"
    physical=file_checkpoint(proper,checkpoint)
    restored=GreenTunnel()
    restored.load_state_dict(load_file(str(checkpoint)),strict=True)
    if infer(proper,valid_rows)!=infer(restored,valid_rows):
        raise AssertionError("loaded neural weights changed predictions")
    metrics=evaluate_all({"original_v120":baseline,
                          "synthetic_negative_only":control,
                          "real_host_hard_negative":restored},valid_rows)
    predeclared_decision=(
        metrics["real_host_hard_negative"]["CB59_v119"]["wrong_generated_rejection_rate"] >
        metrics["synthetic_negative_only"]["CB59_v119"]["wrong_generated_rejection_rate"]
        and
        metrics["real_host_hard_negative"]["Pythia14M"]["wrong_generated_rejection_rate"] >
        metrics["synthetic_negative_only"]["Pythia14M"]["wrong_generated_rejection_rate"]
    )
    fact_specific_positive=metrics["real_host_hard_negative"]["reference_gold"]["positive_acceptance"]
    report={
        "experiment":"v124 one 8MB Green Tunnel trained on actual generated failures of TWO LLM architectures",
        "fixed_same_neural_weights_for_both_HOSTS":True,
        "actual_host_source_metadata":{"train":train_source,"validation":valid_source},
        "question_split":{"train_q":len(TRAIN),"holdout_q":len(VALID),
                          "never_overlap":True},
        "positive_train_candidates_are_ground_truth_NOT_teacher_logits":True,
        "generated_negative_text_never_replaced_with_gold":True,
        "real_candidates_used_in_train":True,
        "real_heldout_candidates_not_used_for_gradient":True,
        "source_real_candidate_training_sha256":hashlib.sha256(
            json.dumps(train_candidates,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        "source_real_candidate_validation_sha256":hashlib.sha256(
            json.dumps(valid_candidates,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        "training_counts":{
            "total_REAL_host_hard_examples":len(train_rows),
            "real_host_wrong_answers_vs_correct_answers":positive_and_negative(train_rows),
            "synthetic_only_training_examples":len(simple_synthetic(train_rows))},
        "fit_trace":{"synthetic_only":control_curve,"actual_hard_negative":curve},
        "one_final_10MB_green_model":physical,
        "valid_all_source_breakdown":metrics,
        "fully_source_HOST_only_bpp_excluding_reusable_green":
            8*len(source.read_bytes())/134515008,
        "one_REUSABLE_green_is_not_included_in_CORE_bpp_by_user_request":True,
        "one_single_neural_model_stored_once":True,
        "neural_discards_bad_answers_not_generate_new_correct_ones":True,
        "new_valid_quality_improvement_is_NOT_assumed":True,
        "hard_negatives_better_both_hosts_than_synthetic_baseline":predeclared_decision,
        "actual_gold_positive_acceptance":fact_specific_positive,
        "SCIENCE_GATES_not_predeclared_pass_if_no_competent_answer_generation":False,
        "unseen_1T_host_not_tested":True,
        "v89_independent_TEST_unopened":True,
        "production_unmodified":True,
        "elapsed_seconds":time.perf_counter()-start}
    (out/"v124_host_hard_negative.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v124_TWO_HOST_REAL_NEGATIVES_DONE",
        "one_neural_file_bytes":physical["file_bytes"],
        "original_135M_core_bpp":report["fully_source_HOST_only_bpp_excluding_reusable_green"],
        "valid_hard_negative_grade":{name:{
            k:val.get("wrong_generated_rejection_rate")
            for k,val in scores.items() if k in ("Pythia14M","CB59_v119")}
            for name,scores in metrics.items()},
        "gold_acceptance":fact_specific_positive,
        "both_hosts_hard_training_better":predeclared_decision,
        "scientific_ability_to_generate_coherent_answer":False},
        ensure_ascii=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("host",type=Path)
    p.add_argument("previous_neural",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.host,a.previous_neural,a.out)
