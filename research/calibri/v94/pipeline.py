"""v94: public TRAIN-heldout actual task accuracy + genuine generation + full speed.

Original v92 complete physically rewritten same-budget source vs frozen CB59.
Never read v89 independent TEST. No comparison to FP32 probability distributions.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v92.validate import swap_one_block
from .provenance import verify_sources
from .qa import build_boolq,build_sciq,score_cases
from .answers import paired
from .generation import paired_greedy
from .policy import evaluate
from .repair_plan import plan

def run(original_path:Path,candidate_path:Path,report_path:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    source=verify_sources(original_path,candidate_path,report_path)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    boolq,src_boolq,yn=build_boolq(tok)
    sciq,src_sciq,abcd=build_sciq(tok)
    student,meta,_=make_model("baseline",SimpleNamespace(original=original_path,folder=out))
    if not meta["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("direct packed complete native model required")
    parent,leaf,old,new=swap_one_block(student,original_path.read_bytes(),
        candidate_path.read_bytes(),source["rewritten_complete_operator"])
    domain_reports={}
    try:
        for domain,items,answers,origin in (
            ("BoolQ",boolq,yn,src_boolq),("SciQ",sciq,abcd,src_sciq)):
            setattr(parent,leaf,old)
            baseline=score_cases(student,items,answers)
            setattr(parent,leaf,new)
            trial=score_cases(student,items,answers)
            domain_reports[domain]=paired(domain,baseline,trial,len(answers))
            domain_reports[domain]["original_public_train_holdout_provenance"]=origin
            domain_reports[domain]["original_paired_answers"]={"CB59":baseline,"rewritten":trial}
        generation=paired_greedy(student,tok,parent,leaf,old,new)
    finally:setattr(parent,leaf,old)
    decision=evaluate(domain_reports,generation,source,semantic_quality_verified=False)
    next_plan=plan(domain_reports,generation,source,decision)
    payload={"experiment":"CALIBRI_v94_outcome_first_multidomain_honest_quality_preservation",
        "original_real_pretrained_revision":MODEL_SHA,"complete_native_archive":source,
        "independent_from_all_prior_v90_v92 TRAIN_SELECTION":True,
        "domain_results":domain_reports,
        "generation_and_full_CPU_decode":generation,
        "behavior_first_automatic_decision":decision,
        "facade_based_automatic_repair_planner":next_plan,
        "real_correctness_quality_and_repetition_are_objectives":True,
        "KL_NLL_and_FP32_similarity_not_measured_or_used":True,
        "all_new_sources_original_public_train_split_only":True,
        "v89_independent_TEST_and_main_Render_Supabase_Jessica_untouched":True,
        "new_model_weights_not_trained_or_deployed":True,
        "study_limitations":"This is additional independent TRAIN-heldout comparison on a formerly "
           "selected research candidate. SciQ/BoolQ next-token constrained choices are "
           "task correctness signals, not an estimate of all human answer quality. "
           "Generation is 32 greedy tokens on 16 public synthetic prompts; human "
           "semantic and factual review is still required."}
    path=out/"v94_original_true_answers_multi_domain_full_greedy_cpu.json"
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    digest={"stage":"v94_measured_functional_preservation_no_distribution_metrics",
        "BoolQ":{key:domain_reports["BoolQ"][key]["correct"] for key in ("baseline","rewritten")},
        "SciQ":{key:domain_reports["SciQ"][key]["correct"] for key in ("baseline","rewritten")},
        "generation_unique_ratio":{kind:generation[kind]["mean_distinct_ratio"]
                                   for kind in ("original","candidate")},
        "full_model_cpu_decode_speed_ratio":generation["paired_full_model_greedy_decode_speed_ratio"],
        "status":decision["status"],"next_repair":next_plan["next_action"],
        "official_TEST_untouched":True}
    print(json.dumps(digest,ensure_ascii=False),flush=True)
    return payload
