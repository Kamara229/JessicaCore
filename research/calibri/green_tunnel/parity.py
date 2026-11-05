"""STRICT source-parity acceptance, separate from Green Tunnel TRAIN signals.

Never confuse loss reduction, source bpp success, first-token matches,
"all Yes", a loose GOLD substring parser, or GitHub CI GREEN with
independently verified quality of FULL model answers.

The v192 grader remains frozen. A domain with many ungraded outputs is
NOT evidence of success, and a development set can never certify parity.
"""
from collections import defaultdict
from typing import Mapping, Any
from v192.grade import summarize as frozen_grade
from .contract import PhysicalSource


def compare_machine_parity(
    cases:list[dict],
    original_fp32_answers:Mapping[int,str],
    recovered_packed_answers:Mapping[int,str],
    physical:PhysicalSource,
    *,
    paired_native_cpu_ratio:float|None,
    qa_independent_of_training_and_model_selection:bool,
    minimum_independent_questions:int=96,
    both_checkpoints_frozen_before_QA:bool,
    native_only_paid_weights:bool,
    native_eos_count:int|None=None,
    full_answer_distinct_fraction:float|None=None,
)->dict:
    """Conservative *machine* score, not human review or universal parity proof.

    Require every previously machine-correct original FP32 case to remain
    machine-correct, not just equal aggregate counts. Compare contradictory
    pair 2/2 and all domains. To call the model ready to scale require
    >=96 completely fresh questions independent of TRAIN+model choice,
    real physical source under <=.700 and matched 90% CPU benchmark.
    """
    if not cases or len(set(c["case_id"] for c in cases))!=len(cases):
        raise ValueError("A full unique frozen question set is required")
    expected={c["case_id"] for c in cases}
    if set(original_fp32_answers)!=expected or set(recovered_packed_answers)!=expected:
        raise ValueError("Missing/supplemented original or compressed actual responses")
    if not physical.valid_rate:
        raise ValueError("Complete physically paid compressed source >0.700 bpp")
    baseline=frozen_grade(cases,original_fp32_answers)
    packed=frozen_grade(cases,recovered_packed_answers)
    if baseline["grader_version"]!=packed["grader_version"]:
        raise AssertionError("Original FP32 and CALIBRI scored by different grammars")
    old={x["case_id"]:x for x in baseline["verbatim_answers_and_grades"]}
    new={x["case_id"]:x for x in packed["verbatim_answers_and_grades"]}
    fp32_correct={cid for cid,x in old.items()
                  if x["machine_status"]=="machine_correct"}
    calibri_correct={cid for cid,x in new.items()
                  if x["machine_status"]=="machine_correct"}
    lost=sorted(fp32_correct-calibri_correct)
    new_correct=sorted(calibri_correct-fp32_correct)
    per_domain={}
    domain_names=set(baseline["domain_totals"])|set(packed["domain_totals"])
    for d in sorted(domain_names):
        b=baseline["domain_totals"][d]
        q=packed["domain_totals"][d]
        if b["n"]!=q["n"]:
            raise AssertionError("Source and compressed domains not same actual question set")
        per_domain[d]={"n":b["n"],"original_confirmed":b["machine_correct"],
           "compressed_confirmed":q["machine_correct"],
           "original_ungraded":b["ungraded"],"compressed_ungraded":q["ungraded"],
           "correct_answers_not_fewer_than_original":q["machine_correct"]>=b["machine_correct"],
           "ungraded_not_greater_than_original":q["ungraded"]<=b["ungraded"]}
    pairs={}
    for domain in sorted(set(baseline["opposite_pairs"])|set(packed["opposite_pairs"])):
        b=baseline["opposite_pairs"].get(domain,{"total":0,"both_correct":0})
        q=packed["opposite_pairs"].get(domain,{"total":0,"both_correct":0})
        if b["total"]!=q["total"]:
            raise AssertionError("Pair-completeness mismatch")
        pairs[domain]={"total":b["total"],"original_both_correct":b["both_correct"],
          "compressed_both_correct":q["both_correct"],
          "full_opposition_at_least_original":q["both_correct"]>=b["both_correct"]}
    qa_domains_pass=all(v["correct_answers_not_fewer_than_original"] and
                        v["ungraded_not_greater_than_original"]
                        for v in per_domain.values())
    all_baseline_correct_retained=(len(lost)==0)
    pair_pass=all(v["full_opposition_at_least_original"] for v in pairs.values())
    recovered_parity_on_evaluated_machine_cases=(
        all_baseline_correct_retained and qa_domains_pass and pair_pass and
        packed["machine_confirmed"]>=baseline["machine_confirmed"] and
        packed["ungraded_not_known_wrong"]<=baseline["ungraded_not_known_wrong"])
    speed_known=paired_native_cpu_ratio is not None
    speed_pass=bool(speed_known and
        isinstance(paired_native_cpu_ratio,(int,float)) and
        0.90<=paired_native_cpu_ratio<100)
    health_pass=True
    if native_eos_count is not None:
        if not 0<=native_eos_count<=len(cases):
            raise ValueError("Actual EOS count outside QA response range")
        health_pass=health_pass and native_eos_count>0
    if full_answer_distinct_fraction is not None:
        if not 0<=full_answer_distinct_fraction<=1:
            raise ValueError("Actual generated-token diversity fraction invalid")
        health_pass=health_pass and full_answer_distinct_fraction>=0.25
    independent_sufficient=(
        qa_independent_of_training_and_model_selection and
        both_checkpoints_frozen_before_QA and
        len(cases)>=minimum_independent_questions)
    can_certify_135m_pre_scale_parity=(
        physical.valid_rate and native_only_paid_weights and
        recovered_parity_on_evaluated_machine_cases and
        speed_pass and health_pass and independent_sufficient)
    reasons=[]
    if not all_baseline_correct_retained:reasons.append("original_FP32_correct_answers_lost")
    if not qa_domains_pass:reasons.append("domain_correctness_or_ungraded_regressed")
    if not pair_pass:reasons.append("opposing_pair_complete_correctness_regressed")
    if not speed_pass:reasons.append("paired_native_CPU_speed_below_90pct_or_unmeasured")
    if not health_pass:reasons.append("generation_EOS_or_diversity_unhealthy")
    if not native_only_paid_weights:reasons.append("unpaid_original_FP32_or_sidecar_in_runtime")
    if not independent_sufficient:reasons.append("not_presealed_sufficient_independent_QA")
    return {
       "strict_machine_grader":baseline["grader_version"],
       "same_frozen_question_IDs_for_original_and_CALIBRI":True,
       "entire_source_physically_paid_bpp":physical.actual_bpp,
       "source_SHA256":physical.physical_sha256,
       "original_machine_confirmed":baseline["machine_confirmed"],
       "calibri_machine_confirmed":packed["machine_confirmed"],
       "original_ungraded":baseline["ungraded_not_known_wrong"],
       "calibri_ungraded":packed["ungraded_not_known_wrong"],
       "previously_FP32_correct_cases_retained":len(fp32_correct&calibri_correct),
       "FP32_correct_cases_LOST_in_CALIBRI":lost,
       "new_machine_correct_case_IDs_not_in_original":new_correct,
       "by_domain":per_domain,"opposite_pair_behavior":pairs,
       "observed_development_MACHINE_parity_not_human":recovered_parity_on_evaluated_machine_cases,
       "native_90pct_median_speed_pass":speed_pass,
       "native_EOS_diversity_probes_pass":health_pass,
       "independent_presealed_96plus_QA":independent_sufficient,
       "CERTIFIED_PARITY_TO_ORIGINAL_FP32_FOR_SCALING":can_certify_135m_pre_scale_parity,
       "hold_135M_no_larger_model_until_certified":not can_certify_135m_pre_scale_parity,
       "FAIL_OR_UNDETERMINED_REASONS":reasons,
       "no_KL_NLL_PPL_acceptance":True,
       "training_loss_plays_no_role_in_pass":True,
       "machine_parity_cannot_replace_semantic_human_review":True
    }
