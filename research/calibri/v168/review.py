"""Exploratory, source-SHA-locked adjudication of v167 ACTUAL FREE ANSWERS.

Owner's research criterion is response correctness/coherence and speed,
NOT token distribution resemblance, NLL, KL, or PPL.

The v167 strict-exact comparator labeled e.g. '7 + 5 = 12.' as wrong.
This retrospective review is intentionally labeled POST-HOC and is NOT
an independent hidden TEST or predeclared generalization benchmark.
"""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter
from pathlib import Path

PRIOR_RUN=41966247057
PRIOR_ARTIFACT=13388681824
PRIOR_REPORT_SHA="80a0006e93db53c4d8e4e5073b037cd04b68f9ff8a7e2a90e175556a809d9146"
ARMS=("FP32","CB59","CB161")
KNOWN_CORRECT_FP32=frozenset((
    0,1,3,4,7,   # arithmetic: five unambiguous correct results
    8,9,11,12,13,14,15, # factual: one incorrect 'week=52'
    16,18,19,20,22,23,24,25,26,27,28,29,30,31, # binding: two incorrect swaps
    32,34,35,37, # reasoning: four correct, two incomplete and two false
))
KNOWN_INCORRECT_FP32={
    2:"incomplete calculation; no correct result",
    5:"incomplete calculation; no correct result",
    6:"confidently gives 59 instead of 49",
    10:"confuses days in a week with weeks in a year",
    17:"incorrectly repeats blue after swapping subject to red",
    21:"contradictory: Lina yellow badge is green",
    33:"incomplete, does not give number of chairs",
    36:"yes to false numerical comparison",
    38:"half of 14 incorrectly given as 6",
    39:"refuses to answer simple weekday question",
}
FREE_FACTUAL_RANGE=range(40)
DOMAINS=("arithmetic","factual","binding","reasoning")
NON_FACTUAL=("instruction","creative")

def validate_annotations():
    expected=set(FREE_FACTUAL_RANGE)
    if KNOWN_CORRECT_FP32 & set(KNOWN_INCORRECT_FP32):
        raise AssertionError("contradictory adjudication")
    if KNOWN_CORRECT_FP32 | set(KNOWN_INCORRECT_FP32)!=expected:
        raise AssertionError("unlabeled question or out-of-range adjudication")
    if len(KNOWN_CORRECT_FP32)!=30 or len(KNOWN_INCORRECT_FP32)!=10:
        raise AssertionError("review count changed")
    return True

def physical_review(path:Path):
    validate_annotations()
    blob=path.read_bytes()
    sha=hashlib.sha256(blob).hexdigest()
    if sha!=PRIOR_REPORT_SHA:
        raise ValueError("incorrect v167 actual outputs; manual labels must be reaudited")
    j=json.loads(blob)
    for k in ("NO_KL_NLL_PPL_USED","protected_official_independent_TEST_not_opened"):
        if j.get(k) is not True:
            raise ValueError("source benchmark protection mismatch")
    records=j["real_48_free_generation_case_records"]
    if tuple(records)!=ARMS:
        raise ValueError("missing paired actual FP32/CB59/CB161 generated outputs")
    all_annotations={}
    totals={}
    for arm in ARMS:
        rows=records[arm]
        if len(rows)!=48 or [x["id"] for x in rows]!=list(range(48)):
            raise ValueError("changed/omitted original generated answer")
        all_annotations[arm]=[]
        for x in rows:
            idx=x["id"]
            if idx>=40:
                judgement=None
                reason="open response: requires separate human review of quality/fluency, not forced factual accuracy"
            elif arm=="FP32":
                judgement=idx in KNOWN_CORRECT_FP32
                reason=("unambiguous correct final factual answer present in the original FP32 text"
                        if judgement else KNOWN_INCORRECT_FP32[idx])
            else:
                text=x["verbatim_text"].lower().strip()
                # Fail closed if compressed outputs no longer consist of the
                # observed repetitive nonanswers: those must be human-reviewed.
                if arm=="CB59":
                    tokens=("speci","fevers","compon","specirofen")
                    if not all(z in " ".join(tokens)+" " for z in text.split()):
                        # a few emitted morphs are glued to tokens; use
                        # fully traced actual output guards below.
                        if not all(c.isalpha() or c.isspace() for c in text):
                            raise ValueError(f"unexpected CB59 output #{idx}; manually inspect")
                else:
                    if not all(c.isalpha() or c.isspace() for c in text):
                        raise ValueError(f"unexpected CB161 output #{idx}; manually inspect")
                judgement=False
                reason="actual model output is a repeating unrelated subword string, not an answer"
            all_annotations[arm].append({
                "id":idx,"domain":x["domain"],
                "question":x["question"],"gold":x["gold_reference_if_applicable"],
                "verbatim_answer":x["verbatim_text"],
                "answer_SHA256":hashlib.sha256(x["verbatim_text"].encode()).hexdigest(),
                "factual_correct_review":judgement,"annotation":reason,
                "strict_exact_v167":x["strict_exact_gold"],
                "source_prompt_SHA256":x["prompt_sha"],
                "posthoc_review_not_independent":True
            })
        by={}
        for dom in DOMAINS:
            selected=[a for a in all_annotations[arm] if a["domain"]==dom]
            by[dom]={"correct":sum(a["factual_correct_review"] for a in selected),
                     "questions":len(selected)}
        totals[arm]={"per_domain":by,
                     "factual_correct":sum(v["correct"] for v in by.values()),
                     "factual_questions":40,
                     "instructions_and_creative_not_fake_graded":8,
                     "strict_exact_was":sum(bool(x["strict_exact_gold"]) for x in rows[:40])}
    paper=j["actual_288_choice_answer_accuracy_and_fp32_ceiling"]["by_domain"]
    bias={}
    for dom in paper:
        bias[dom]={arm:{
            "correct":paper[dom][arm]["correct"],
            "number":paper[dom][arm]["questions"],
            "chosen_labels":paper[dom][arm]["chosen_label_histogram"]
        } for arm in ARMS}
    no_evidence={
        "BoolQ":"ALL three models returned Yes on all 96 questions; 56/96 is constant-Yes baseline, not demonstrated passage comprehension",
        "SciQ":"FP32 used only A or D; CB161 only D in 96 examples; similar percent correct is not capability preservation",
        "Arithmetic":"forced choice original FP32 always A, CB161 mostly D; 25 percent may be option-prior baseline",
        "Binding":"the original four-choice synthetic benchmark only put gold in A or B; measured 0/32 on packed D is not properly balanced, and must NOT be treated as isolated proof of binding loss",
        "Logic":"original FP32 overwhelmingly chose A, packed D; chance-level choice accuracy is not proof of reasoning"
    }
    return {
        "stage":"CALIBRI_v168_CORRECTED_RESPONSE_FIRST_V167_ACTUAL_TEXT_ADJUDICATION",
        "source":{"v167_run":PRIOR_RUN,"v167_artifact":PRIOR_ARTIFACT,
                  "complete_JSON_SHA256":PRIOR_REPORT_SHA},
        "preliminary_POSTHOC_manual_factual_review":totals,
        "all_48_original_questions_gold_outputs_and_reasons_per_arm":all_annotations,
        "v167_FORCED_CHOICE_RESULTS_BUT_EXPLAINED_BIAS":bias,
        "quality_assessment_caveats":no_evidence,
        "confirmed_behavioral_observation":"FP32 generated many meaningful answers. Both physically packed variants produced overwhelmingly repetitive non-answers. Low forced-choice scores do not imply FP32 was incapable.",
        "speed_comparison_v167_measured_original_cpu":j["actual_free_generation_and_cpu_decode_speed"]["by_arm"],
        "MUST_NOT_USE_KL_NLL_PPL_TO_SELECT_CODEC":True,
        "posthoc_adjudication_not_independent_test":True,
        "no_new_weights_no_source_refitting_no_prod_changes":True,
        "research_action":"Next test properly balanced free-response contrastive QA, plus physical speed and text, before deciding how to improve physical codebook."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("frozen_v167_complete_json",type=Path)
    p.add_argument("outdir",type=Path)
    a=p.parse_args()
    a.outdir.mkdir(parents=True,exist_ok=True)
    result=physical_review(a.frozen_v167_complete_json)
    out=a.outdir/"v168_v167_source_corrected_behaviour_regrade.json"
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":result["stage"],
        "posthoc_original_factual_review":result["preliminary_POSTHOC_manual_factual_review"],
        "choice_bias_summary":result["quality_assessment_caveats"],
        "report":str(out)},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
