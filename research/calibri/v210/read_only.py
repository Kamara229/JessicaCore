"""v210 independent READ-ONLY retrospective original-135M sealed v206→v208 QA comparison.

PUBLIC DEV is already repeatedly observed: it MUST NOT inform source changes,
checkpoint selection, new TRAIN sample selection, or learning parameters.
Never interpret ungraded as wrong or green CI as full-answer parity.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from v205.read_only import _read,_grades
from v207.read_only import audit_steps

def analyze(q206,q208,a206,a208,s206,s208):
    if q206["cases"]!=q208["cases"]:
        raise AssertionError("Frozen original 16 questions/gold changed")
    if q206["preexisting_v192_frozen_grader_sha256"]!=q208["preexisting_v192_frozen_grader_sha256"]:
        raise AssertionError("Original machine grader changed")
    f206,f208=(_grades(q,"FP32_existing_135M") for q in (q206,q208))
    if f206!=f208:raise AssertionError("Original frozen FP32 verbatim answers changed")
    prev=_grades(q206,"GT206")
    baseline=_grades(q208,"CB206")
    if prev!=baseline:raise AssertionError("Physically sealed v206 baseline answers or grading drift")
    after=_grades(q208,"GT208")
    if set(prev)!=set(after) or len(after)!=16:
        raise AssertionError("Not the same 16 locked PUBLIC DEV questions")
    t206,t208=audit_steps(a206,48),audit_steps(a208,48)
    if set(t206["operators"])&set(t208["operators"]):
        raise AssertionError("v208 repeated v206 paid operators instead of next 48")
    for key,s in [("v206",s206),("v208",s208)]:
        if not s["completed"] or s["whole_original_source_bpp"]>.7:
            raise AssertionError(key+" incomplete or above physical bpp budget")
        if s["native_CPU_speed_ratio_CB59"]<=0:
            raise AssertionError("Missing native CPU speed")
    if s206["source_SHA256"]==s208["source_SHA256"]:
        raise AssertionError("Distinct verified physical sources expected")
    if s206["source_SHA256"]!="5a6d5f07f9e49fa6063dd772d7e2aee53beeb3527d10d5a429715593b68939de":
        raise AssertionError("v208 has incorrect signed input v206")
    if s208["source_SHA256"]!="cd0947dcd34b1befee8aa2d58821b869a8476e8156a78a2a9747ea565f5b90ec":
        raise AssertionError("v208 did not produce signed expected physical source")
    transitions=Counter()
    domain_changes=defaultdict(Counter)
    rows=[]
    for i in sorted(prev):
        a,b=prev[i],after[i]
        if a["domain"]!=b["domain"] or a["domain"]!=f206[i]["domain"]:
            raise AssertionError("Question domain drift")
        t=a["machine_status"]+" -> "+b["machine_status"]
        transitions[t]+=1
        domain_changes[a["domain"]][t]+=1
        rows.append({"case_id":i,"domain":a["domain"],
                     "FP32_status":f206[i]["machine_status"],
                     "v206_status":a["machine_status"],"v208_status":b["machine_status"],
                     "v206_full_answer":a["raw"],"v208_full_answer":b["raw"],
                     "response_changed":a["raw"]!=b["raw"]})
    if sum(transitions.values())!=16:raise AssertionError("Partial sealed QA review")
    return {"experiment":"v210_read_only_v206_v208_original135M_PUBLIC_DEV_diagnostics",
      "NO_model_execution_training_optimizer_or_gradients":True,
      "PUBLIC_DEV_not_blind_and_NOT_checkpoint_training_parameter_selector":True,
      "FP32_confirmed":sum(x["machine_status"]=="machine_correct" for x in f206.values()),
      "v206_confirmed":sum(x["machine_status"]=="machine_correct" for x in prev.values()),
      "v208_confirmed":sum(x["machine_status"]=="machine_correct" for x in after.values()),
      "v206_recognized_wrong":sum(x["machine_status"]=="machine_wrong" for x in prev.values()),
      "v208_recognized_wrong":sum(x["machine_status"]=="machine_wrong" for x in after.values()),
      "v206_ungraded":sum(x["machine_status"]=="ungraded" for x in prev.values()),
      "v208_ungraded":sum(x["machine_status"]=="ungraded" for x in after.values()),
      "statuses_including_ungraded":dict(sorted(transitions.items())),
      "domains":{d:dict(sorted(c.items())) for d,c in sorted(domain_changes.items())},
      "all_16_full_answers_unchanged_grader":rows,
      "v206_TRAIN_48_step_audit":t206,"v208_TRAIN_next48_audit":t208,
      "v206_physical_SHA256":s206["source_SHA256"],
      "v208_physical_SHA256":s208["source_SHA256"],
      "v206_paid_bpp":s206["whole_original_source_bpp"],
      "v208_paid_bpp":s208["whole_original_source_bpp"],
      "v206_native_CPU_vs_CB59":s206["native_CPU_speed_ratio_CB59"],
      "v208_native_CPU_vs_CB59":s208["native_CPU_speed_ratio_CB59"],
      "quality_parity_not_certified":True,"NO_LIVE_main_Jessica_Render_Supabase_APK":True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("v206_artifact");ap.add_argument("v208_artifact");ap.add_argument("out")
    a=ap.parse_args()
    l,r=Path(a.v206_artifact),Path(a.v208_artifact)
    result=analyze(
        _read(l/"REAL_GREEN_TUNNEL_new_public_DEV_16_full_QA.json"),
        _read(r/"REAL_GREEN_TUNNEL_new_public_DEV_16_full_QA.json"),
        _read(l/"ACTUAL_guarded_TRAIN_48_step_audit.json"),
        _read(r/"ACTUAL_guarded_TRAIN_48_step_audit.json"),
        _read(l/"v206_real_NATIVE_SOURCE_QA_summary.json"),
        _read(r/"v208_real_NATIVE_SOURCE_QA_summary.json"))
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=True)
    (folder/"READ_ONLY_SEALED_v206_v208_original135M_full_answer_differences.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"stage":"v210_sealed_v206_to_v208_read_only_completed",
        "FP32":result["FP32_confirmed"],"v206":result["v206_confirmed"],
        "v208":result["v208_confirmed"],
        "transitions":result["statuses_including_ungraded"],
        "v206_accepted":len(result["v206_TRAIN_48_step_audit"]["accepted"]),
        "v208_accepted":len(result["v208_TRAIN_next48_audit"]["accepted"]),
        "PUBLIC_DEV_NOT_used_to_tune":True},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
