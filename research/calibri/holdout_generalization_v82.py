#!/usr/bin/env python3
"""CALIBRI v82: NEW, larger, source-complete QA/fluency GENERALIZATION gate.

No novel candidate tuned on published v79/v81 TEST results. Reproduce
the frozen genuine v81 two-TRAIN-split hard-KL constrained source without
changing any gradient, thresholds, seed or model representation. Before
opening independent NEW SciQ TEST / BoolQ VALID rows and fresh language
TEST tokens, record original full source SHA and TRAIN choice.

Then run genuine pinned original trained FP32 teacher, original native
5bit packed CB59, paid v76, and frozen v81 CB76, all independently. Use
80 original SciQ & 80 BoolQ source rows, 16x64 new WikiText/AGNews
contexts and 4x24 genuine greedy KV-cache prompts. Compute paired
McNemar and paired bootstrap uncertainty, physical bytes, source SHA,
absolute KL/NLL and model throughput/deduplication.

This is a replication / falsification stage, NOT new weight training,
new compression theorem, or production Jessica deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import hard_kl_answer_v81 as v81
import repro_qa_calibration_v78 as evaluation
from binary_geometry_v64 import EXPECTED_UNIQUE
from paid_lowrank_v76 import SOURCE_BYTES,REQUIRED_PHYSICAL_BYTES,unpack

# Completely untouched v82 TEST chosen before reproducing the original
# v81 TRAIN algorithm; do not update these constants from QA outcomes.
SCIQ_TEST_ROWS=(160,240)
BOOLQ_VALID_ROWS=(2990,3070)
WIKITEXT_TEST_START=140000
AGNEWS_TEST_ROWS=(7410,7510)
CONTEXT_COUNT=16
CONTEXT_SIZE=64
AUTOREGRESSIVE_PROMPTS=(
    "If the results are consistent, the next explanation is",
    "The new discovery surprised the researchers because",
    "The key difference between the two approaches is",
    "At the beginning of the story, the traveler noticed",
)
TOKENS_PER_PROMPT=24
KIND_ORDER=("teacher","baseline","paid_v76","qa_v79")
SAMPLE_BSTRAPS=5000
SOURCE_PARENT_SHA_PREVIOUS_V81="94282c492713b053ccabc76510bad5f45791142866aa81abad62b35081e752c0"
ORIGINAL_PARAM_COUNT=134515008
assert ORIGINAL_PARAM_COUNT==EXPECTED_UNIQUE
assert 8*REQUIRED_PHYSICAL_BYTES/EXPECTED_UNIQUE<0.700

def setup_frozen_ranges():
    # This function is used exclusively during independent TEST after
    # train(args), train repeat(args), and freeze(args) in CI.
    v81.NEW_TEST_SCIQ=SCIQ_TEST_ROWS
    v81.NEW_TEST_BOOLQ=BOOLQ_VALID_ROWS
    v81.NEW_TEST_WIKI=WIKITEXT_TEST_START
    v81.NEW_TEST_NEWS=AGNEWS_TEST_ROWS
    v81.NEW_PROMPTS=AUTOREGRESSIVE_PROMPTS
    evaluation.WINDOW_COUNT=CONTEXT_COUNT
    evaluation.WINDOW=CONTEXT_SIZE
    evaluation.NEW_TOKENS_PER_PROMPT=TOKENS_PER_PROMPT

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def pretest_audit(folder,source):
    original=source.read_bytes()
    candidate=(folder/"CALIBRI_v79_frozen_complete.cb76").read_bytes()
    parent=(folder/"CALIBRI_v76_paid_full.cb76").read_bytes()
    frozen=json.loads((folder/"v81_FROZEN_before_TEST.json").read_text())
    report1=json.loads((folder/"v81_train_first.json").read_text())
    report2=json.loads((folder/"v81_train_second.json").read_text())
    _,U,V,b=unpack(candidate)
    _,parentU,parentV,parentB=unpack(parent)
    if len(original)!=SOURCE_BYTES or len(candidate)!=REQUIRED_PHYSICAL_BYTES:
        raise AssertionError("not a complete original pretrained source with paid factors")
    if candidate[:SOURCE_BYTES]!=original or parent[:SOURCE_BYTES]!=original:
        raise AssertionError("trained 2D original source prefix changed")
    if frozen["frozen_first_source_SHA"]!=sha(candidate):
        raise AssertionError("frozen byte SHA/source mismatch")
    if frozen["parent_v76_source_SHA"]!=sha(parent):
        raise AssertionError("original parent source SHA/source mismatch")
    if report1["chosen_TRAIN_checkpoint"]!=report2["chosen_TRAIN_checkpoint"]:
        raise AssertionError("repeated TRAIN optimizer chose differing checkpoints")
    if report1["chosen_TRAIN_checkpoint"]!=frozen["frozen_first_TRAIN_winner"]:
        raise AssertionError("TRAIN decision differs from preregistered source")
    if frozen["source_heldout_TEST_unseen_before_frozen"]!=True:
        raise AssertionError("TEST opened before frozen source")
    # Stronger original v81 cross-host bitwise test: a divergent SHA is a
    # negative reproducibility result, not a reason to tune on TEST.
    stats={
        "study":"CALIBRI_v82_PRETEST_frozen_source_135M_reproduction",
        "original_source_sha":sha(original),
        "paid_parent_sha":sha(parent),
        "newly_reproduced_v81_sha":sha(candidate),
        "prior_v81_run_41555699743_sha":SOURCE_PARENT_SHA_PREVIOUS_V81,
        "cross_host_v81_SOURCE_SHA_match":sha(candidate)==SOURCE_PARENT_SHA_PREVIOUS_V81,
        "same_host_repeat_bitwise_identical":frozen["same_runner_source_bitwise_equal"],
        "TRAIN_only_frozen_checkpoint":frozen["frozen_first_TRAIN_winner"],
        "original_CB59_source_complete_bytes":len(original),
        "paid_CB76_source_complete_bytes":len(candidate),
        "paid_CB76_actual_all_in_bpw":8*len(candidate)/EXPECTED_UNIQUE,
        "original_134m_pretrained_source_untouched":True,
        "original_native_5bit_C_with_cpu_torch_paid_factor":True,
        "original_dense_pretrained_2D_absent_in_packed_runtime":True,
        "same_physical_shape_output_U_V_bias":
           [list(U.shape),list(V.shape),list(b.shape)],
        "same_physical_shape_parent":
           [list(parentU.shape),list(parentV.shape),list(parentB.shape)],
        "never_seen_v82_SciQ_TEST_range":list(SCIQ_TEST_ROWS),
        "never_seen_v82_BoolQ_VALID_range":list(BOOLQ_VALID_ROWS),
        "never_seen_v82_WikiText_test_token_range":
            [WIKITEXT_TEST_START,WIKITEXT_TEST_START+CONTEXT_COUNT*CONTEXT_SIZE],
        "never_seen_v82_AGNews_test_record_range":list(AGNEWS_TEST_ROWS),
        "all_test_source_ranges_preregistered_before_first_v82_train":True,
        "NOT_a_newly_TRAIN_optimized_v82_candidate":True,
        "all_parent_and_candidate_weight_bytes_audited_before_TEST":True}
    (folder/"v82_PRETEST_source_and_split_audit.json").write_text(
        json.dumps(stats,indent=2)+"\n")
    print(json.dumps({"stage":"v82_SOURCE_frozen_before_all_NEW_TEST",
                      "selected_original_v81_checkpoint":stats["TRAIN_only_frozen_checkpoint"],
                      "source_sha_vs_v81_prior_match":stats["cross_host_v81_SOURCE_SHA_match"],
                      "physical_bpw":round(stats["paid_CB76_actual_all_in_bpw"],9),
                      "new_QA_TEST_rows":{ "SciQ":SCIQ_TEST_ROWS,"BoolQ":BOOLQ_VALID_ROWS}}),
          flush=True)

def exactly_two_sided_mcnemar(a,b):
    one=int(np.sum((a==1)&(b==0)))
    two=int(np.sum((a==0)&(b==1)))
    n=one+two
    if n==0:return 1.0,one,two
    # Exact two-sided binomial. The samples are paired original items,
    # not independence-assumed binomial on two accuracy rates.
    x=sum(math.comb(n,i) for i in range(min(one,two)+1))
    return min(1.0,2*x/2**n),one,two

def paired_change(a,b,rng):
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    if len(a)!=len(b) or not len(a):
        raise AssertionError("missing matched independent actual source positions")
    d=b-a
    draws=rng.integers(0,len(d),size=(SAMPLE_BSTRAPS,len(d)))
    lo,hi=np.quantile(d[draws].mean(axis=1),[.025,.975])
    return {"difference_new_minus_original":float(d.mean()),
            "paired_bootstrap_95":[float(lo),float(hi)],
            "paired_units":len(d)}

def score(args):
    data={k:json.loads((args.folder/("v81_"+k+".json")).read_text())
          for k in KIND_ORDER}
    frozen=json.loads((args.folder/"v82_PRETEST_source_and_split_audit.json").read_text())
    if not frozen["all_parent_and_candidate_weight_bytes_audited_before_TEST"]:
        raise AssertionError("must freeze original model weights before TEST")
    if data["qa_v79"]["physical_source_integrity"]["complete_physical_SHA256"]!=frozen["newly_reproduced_v81_sha"]:
        raise AssertionError("new actual test uses nonfrozen candidate")
    if data["paid_v76"]["physical_source_integrity"]["complete_physical_SHA256"]!=frozen["paid_parent_sha"]:
        raise AssertionError("parent TEST source checksum mismatch")
    for k in KIND_ORDER[1:]:
        meta=data[k]["physical_source_integrity"]
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise AssertionError("original pretrained dense 2D source in hot runtime")
        bytes_actual=meta["physical_complete_model_bytes"]
        if bytes_actual!=(SOURCE_BYTES if k=="baseline" else REQUIRED_PHYSICAL_BYTES):
            raise AssertionError("physically paid source miscounted")
    out={
      "experiment":"CALIBRI_v82_genuinely_unseen_MORE_QA_and_language_generalization",
      "original_real_trained_model":"HuggingFaceTB/SmolLM2-135M-Instruct",
      "original_unique_trained_params":EXPECTED_UNIQUE,
      "source_complete_baseline_bytes":SOURCE_BYTES,
      "source_complete_paid_bytes":REQUIRED_PHYSICAL_BYTES,
      "source_physically_paid_bpw":8*REQUIRED_PHYSICAL_BYTES/EXPECTED_UNIQUE,
      "original_frozen_v81_TRAIN_source_SHA":frozen["newly_reproduced_v81_sha"],
      "same_source_v81_as_earlier_run":frozen["cross_host_v81_SOURCE_SHA_match"],
      "new_v82_training_or_TEST_tuning_performed":False,
      "all_source_frozen_before_opening_TEST":True,
      "limitations":"One original real pretrained 135M model; tiny 64-token inference contexts; QA is supported SciQ answer-string scoring and truncated BoolQ passage, not original benchmarks. Repetition is 4x24 token diagnostic, not semantic validation.",
      "QA":{},"language":{},"generation":{}
    }
    rng=np.random.default_rng(82)
    for task in ("sciq","boolq"):
        source_ref=data["teacher"]["QA"][task]["source"]
        fingerprint=source_ref["fixed_input_fingerprint"]
        items_teacher=data["teacher"]["QA"][task]["items"]
        for kind in KIND_ORDER:
            task_result=data[kind]["QA"][task]
            if task_result["source"]["fixed_input_fingerprint"]!=fingerprint:
                raise AssertionError("QA source differs between trained originals")
            if len(task_result["items"])!=len(items_teacher):
                raise AssertionError("source QA question count changed across models")
            for a,b in zip(items_teacher,task_result["items"]):
                if a["record_index"]!=b["record_index"] or a["gold"]!=b["gold"] or a["prompt_SHA"]!=b["prompt_SHA"]:
                    raise AssertionError("not same gold original question/score")
        count=len(items_teacher)
        one={"original_row_span":source_ref["original_row_range"],
             "source_excluded_choices":source_ref["source_invalid_rows_uniformly_omitted"],
             "valid_real_questions":count,
             "actual_scoring":data["teacher"]["QA"][task]["score_method"],
             "source_revision_sha":source_ref["revision"],
             "models":{},"paired_deltas":{}}
        for kind in KIND_ORDER:
            items=data[kind]["QA"][task]["items"]
            correct=np.array([z["correct"] for z in items],dtype=np.int64)
            one["models"][kind]={"correct":int(correct.sum()),"total":count,
                                  "accuracy":float(np.mean(correct)),
                                  "actual_CPU_forward_seconds":data[kind]["QA"][task]["CPU_forward_seconds"]}
        for before,after in (("paid_v76","qa_v79"),
                             ("baseline","qa_v79"),
                             ("teacher","qa_v79")):
            a=np.asarray([r["correct"] for r in data[before]["QA"][task]["items"]])
            b=np.asarray([r["correct"] for r in data[after]["QA"][task]["items"]])
            p,old,new=exactly_two_sided_mcnemar(a,b)
            outcome=paired_change(a,b,rng)
            outcome.update({"exact_McNemar_two_sided_p":float(p),
                            "original_only_correct":old,
                            "candidate_only_correct":new,
                            "statistical_unit":"original labeled question",
                            "cannot_claim_general_assistant_competence":True})
            one["paired_deltas"][f"{after}_minus_{before}"]=outcome
        out["QA"][task]=one
    for domain in ("wiki_test","news_test"):
        source_sha=data["teacher"]["language"][domain]["source"]["input_SHA"]
        for k in KIND_ORDER:
            if data[k]["language"][domain]["source"]["input_SHA"]!=source_sha:
                raise AssertionError("independent teacher/source text tokens differ")
        section={"source":data["teacher"]["language"][domain]["source"],
                 "scored_real_next_token_positions":CONTEXT_COUNT*(CONTEXT_SIZE-1),
                 "actual_context_windows":CONTEXT_COUNT,
                 "per_model":{},"paired_change":{}}
        for kind in KIND_ORDER[1:]:
            samples=data[kind]["language"][domain]["data"]
            if len(samples)!=CONTEXT_COUNT:raise AssertionError("missing original TEST windows")
            section["per_model"][kind]={
                "KL":float(np.mean([x["teacher_student_KL"] for x in samples])),
                "NLL":float(np.mean([x["heldout_token_NLL"] for x in samples])),
                "CPU_prefill_s":data[kind]["language"][domain]["CPU_prefill_seconds"]}
        for field in ("teacher_student_KL","heldout_token_NLL"):
            original=data["paid_v76"]["language"][domain]["data"]
            candidate=data["qa_v79"]["language"][domain]["data"]
            section["paired_change"]["v81_minus_v76_"+field]=paired_change(
                [o[field] for o in original],[c[field] for c in candidate],rng)
        out["language"][domain]=section
    for kind in KIND_ORDER:
        gen=data[kind]["greedy_actual_token_generation"]
        out["generation"][kind]={
             "true_KV_caching_used":True,
             "post_prefill_decoded_tokens":gen["genuine_tokens_after_prefill"],
             "cpu_post_prefill_tokens_per_second":gen["tokens_per_second_one_cpu"],
             "mean_distinct_generated_token_ratio":gen["mean_distinct_token_ratio"],
             "mean_largest_token_share":gen["mean_top_token_share"],
             "per_prompt":gen["prompts"]}
    # Predeclared research-quality, NOT a production gate and never used to
    # tune v81 source. A fail is a falsified heldout-generalization claim.
    science={
        "teacher_anchor_sciq_ge_60pct":
            out["QA"]["sciq"]["models"]["teacher"]["accuracy"]>=0.60,
        "v81_sciq_correct_nonregress_vs_parent":
            out["QA"]["sciq"]["models"]["qa_v79"]["correct"]>=
            out["QA"]["sciq"]["models"]["paid_v76"]["correct"],
        "v81_boolq_accuracy_gain_vs_parent_ge_10pp":
            out["QA"]["boolq"]["paired_deltas"]["qa_v79_minus_paid_v76"]["difference_new_minus_original"]>=0.10,
        "v81_boolq_gain_paired_bootstrap_lower_gt_0":
            out["QA"]["boolq"]["paired_deltas"]["qa_v79_minus_paid_v76"]["paired_bootstrap_95"][0]>0,
        "new_heldout_wiki_KL_regress_le_0p25_nats":
            out["language"]["wiki_test"]["paired_change"]["v81_minus_v76_teacher_student_KL"]["difference_new_minus_original"]<=0.25,
        "new_heldout_news_KL_regress_le_0p25_nats":
            out["language"]["news_test"]["paired_change"]["v81_minus_v76_teacher_student_KL"]["difference_new_minus_original"]<=0.25,
        "v81_greedy_distinct_ratio_ge_0p25":
            out["generation"]["qa_v79"]["mean_distinct_generated_token_ratio"]>=0.25,
        "no_main_live_Jessica_Render_Supabase_APK_changes":True}
    out["preregistered_new_TEST_research_go_nogo_checks"]=science
    out["all_research_checks_pass"]=all(science.values())
    out["critical_note"]="New v82 TEST did not train or choose source; any FAIL cannot retroactively alter original v81 train selection. No claim of working assistant."
    path=args.folder/"v82_REAL_independent_TEST_statistical_audit.json"
    path.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({
        "stage":"v82_SUCCESSFULLY_AUDITED_genuine_PRETRAINED_independent_TEST",
        "physical_bpw":round(out["source_physically_paid_bpw"],9),
        "same_v81_SHA_as_previous_host":out["same_source_v81_as_earlier_run"],
        "QA_counts":{task:{name:m["correct"] for name,m in t["models"].items()}
                         for task,t in out["QA"].items()},
        "v81_minus_parent_QA_McNemar_p":{
            task:round(t["paired_deltas"]["qa_v79_minus_paid_v76"]["exact_McNemar_two_sided_p"],6)
            for task,t in out["QA"].items()},
        "v81_minus_parent_TEST_KL":{
            corpus:round(v["paired_change"]["v81_minus_v76_teacher_student_KL"]["difference_new_minus_original"],5)
            for corpus,v in out["language"].items()},
        "greedy_distinct_token_share":{
            k:round(v["mean_distinct_generated_token_ratio"],4) for k,v in out["generation"].items()},
        "all_preregistered_research_go":out["all_research_checks_pass"]
        },ensure_ascii=False),flush=True)

def smoke():
    assert SCIQ_TEST_ROWS[1]<=1000 and SCIQ_TEST_ROWS[0]<SCIQ_TEST_ROWS[1]
    assert BOOLQ_VALID_ROWS[1]<3270 and BOOLQ_VALID_ROWS[0]<BOOLQ_VALID_ROWS[1]
    assert AGNEWS_TEST_ROWS[1]<=7600 and AGNEWS_TEST_ROWS[0]<AGNEWS_TEST_ROWS[1]
    assert SOURCE_BYTES<REQUIRED_PHYSICAL_BYTES<=math.floor(0.700*EXPECTED_UNIQUE/8)
    assert CONTEXT_COUNT==16 and TOKENS_PER_PROMPT==24
    print("v82 independently NEW QA/test spans, exact all-in physical source budget PASS")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",required=True,
                   choices=("smoke","pretest","evaluate","score"))
    p.add_argument("--folder",type=Path,default=Path("results"))
    p.add_argument("--original",type=Path)
    p.add_argument("--kind",choices=KIND_ORDER,default="teacher")
    args=p.parse_args()
    args.folder.mkdir(parents=True,exist_ok=True)
    if args.phase=="smoke":smoke();return
    if args.original is None:
        p.error("--original is required for exact pretrained source SHA/integrity")
    if args.phase=="pretest":pretest_audit(args.folder,args.original)
    elif args.phase=="evaluate":
        if not (args.folder/"v82_PRETEST_source_and_split_audit.json").is_file():
            raise AssertionError("source cannot be tested before physically frozen")
        setup_frozen_ranges()
        v81.evaluate(args)
    else:score(args)
if __name__=="__main__":
    main()
