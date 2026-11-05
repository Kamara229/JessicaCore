"""Real v196 physical recovered 135M vs immutable original FP32 parity audit.

This script does NOT train. It downloads no checkpoints; consumes
completed v196 native generated text and authenticated final 11MB
physical binary, and original independently generated v193 FP32
135M QA. NEVER treat historical PUBLIC DEVELOPMENT parity as final
independent QA proof. Emit strict PASS/FAIL reasons and raw evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

from direct_codebook_v59 import CB59
from green_tunnel.contract import PhysicalSource,verify_physical_source
from green_tunnel.parity import compare_machine_parity
from v193.precommit import grading_fingerprint
def save(path,record):
    """Atomic result without loading any model or changing its weights."""
    path=Path(path)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\\n",encoding="utf8")
    tmp.replace(path)



def audit(experiment_dir,original_v193_json,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    experiment_dir=Path(experiment_dir)
    old=json.loads(Path(original_v193_json).read_text())
    if not old.get("complete"):
        raise AssertionError("No original FP32 135M answer source")
    raw=old["actual_unfiltered_FP32_generations_96_each"]["135M"]
    v193cases=old["frozen_before_any_model_generation"]["full_questions_golds"]
    grader=old["frozen_before_any_model_generation"]["grading_source_frozen_before_inference"]["sha256"]
    if grader!=grading_fingerprint()["sha256"] or len(raw)!=96 or len(v193cases)!=96:
        raise AssertionError("Historical precommitted QA or original scorer changed")

    latest=json.loads((experiment_dir/"REAL_GREEN_TUNNEL_new_public_DEV_16_full_QA.json").read_text())
    if not latest.get("complete"):
        raise AssertionError("Native Green Tunnel actual QA unfinished")
    cases=latest["cases"]
    original={c["case_id"]:raw[c["case_id"]]["complete_generation_original"]["verbatim_text"] for c in cases}
    if (len(cases)!=16 or
        [c["question"] for c in cases]!=[v193cases[c["case_id"]]["question"] for c in cases]):
        raise AssertionError("Native paid source tested different question texts")
    saved_original={x["case_id"]:x["verbatim_text"]
        for x in latest["original_actual_135M_FP32_from_previous_sealed_v193"]}
    if saved_original!=original:raise AssertionError("Native QA original FP32 control drifted")

    seal=json.loads((experiment_dir/"PHYSICAL_GREEN_TUNNEL_SOURCE_SEALED_BEFORE_ANY_QA.json").read_text())
    archive_path=experiment_dir/"CALIBRI_v196_green_tunnel_all_vocab_2cycles_complete_135M.cb59"
    binary=archive_path.read_bytes()
    blob=CB59(binary)
    source=PhysicalSource(
        codec="CB59_K32V8 complete 135M",
        original_unique_parameters=134_515_008,
        physical_whole_bytes=seal["bytes"],
        physical_sha256=seal["sha256"],
        aliases=tuple(sorted(blob.meta["source_alias_keys"].items())))
    source_check=verify_physical_source(
        source,binary,decode=CB59,
        verify_alias=lambda a,aliases:(
            a.meta["original_unique_parameter_count"]==134_515_008 and
            tuple(sorted(a.meta["source_alias_keys"].items()))==aliases and
            a.handles["model.embed_tokens.weight"] is a.handles["lm_head.weight"]))
    if source_check["physical_size_bytes"]!=11_229_303:
        raise AssertionError("Green Tunnel entire source should stay within same physical budget")
    speed=json.loads((experiment_dir/"paired_identical_work_CPU_speed_GT196_vs_CB59.json").read_text())
    measured_ratio=speed["median_speed_ratio_vs_CB59"]
    arms=latest["native_compact_actual_QA"]
    if set(arms)!={"CB59","CB186","GT196"}:
        raise AssertionError("Missing native packed original v186 and Green Tunnel controls")
    judged={}
    for arm,full in arms.items():
        if len(full)!=len(cases):raise AssertionError("Missing actual native question answers")
        answers={r["case_id"]:r["actual_output"]["verbatim_text"] for r in full}
        if set(answers)!={c["case_id"] for c in cases}:
            raise AssertionError("Native response case ID mismatch")
        eos=sum(bool(r["actual_output"]["eos"]) for r in full)
        diversity=sum(r["actual_output"]["distinct_token_fraction"] for r in full)/len(full)
        score=compare_machine_parity(
            cases,original,answers,source,
            paired_native_cpu_ratio=measured_ratio if arm=="GT196" else None,
            qa_independent_of_training_and_model_selection=False,
            both_checkpoints_frozen_before_QA=True,
            native_only_paid_weights=True,
            native_eos_count=eos,
            full_answer_distinct_fraction=diversity)
        judged[arm]={"original_135M_FP32_machine_correct":score["original_machine_confirmed"],
            "actual_135M_source_machine_correct":score["calibri_machine_confirmed"],
            "actual_135M_SOURCE_FP32_regressions":score["FP32_correct_cases_LOST_in_CALIBRI"],
            "actual_strict_rate":score["entire_source_physically_paid_bpp"],
            "actual_NATIVE_generation_EOS_count":eos,
            "actual_NATIVE_token_diversity_fraction":diversity,
            "machine_parity_gate":score,
            "actual_full_native_verbatim_answers":answers}
    complete={"verified_complete_source":source_check,
        "v196_preregistered_v193_original_FP32_135M_questions":cases,
        "all_actual_packed_controls_and_new_recovered_source":judged,
        "native_speed_GT196_vs_CB59_median":measured_ratio,
        "actual_v192_source_grader_SHA":grader,
        "successful_model_training_not_a_semantic_ACCEPTANCE":True,
        "public_16_DEV_CANNOT_certify_original_FP32_parity":True,
        "no_other_models_allowed_until_unseen_strict_parity":True,
        "no_private_v89_TEST":True}
    save(output/"v197_verified_v196_all_arms_REAL_16QA_strict_parity_gates.json",complete)
    print(json.dumps({"stage":"v197_SOURCE_PAID_parity_audit",
        "physically_paid_bytes":source_check["physical_size_bytes"],
        "actual_bpp":source_check["actual_complete_bpp"],
        "original_135M_FP32_confirmed":judged["GT196"]["original_135M_FP32_machine_correct"],
        "original_v186_packed_confirmed":judged["CB186"]["actual_135M_source_machine_correct"],
        "new_GT196_packed_confirmed":judged["GT196"]["actual_135M_source_machine_correct"],
        "native_CPU_speed_vs_CB59":measured_ratio,
        "may_scale_to_360M":False,
        "reason":"Only public 16 DEV cases, not an independent 96+ parity set"}),flush=True)
    return complete


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("v196_completed_physical_and_QA_artifact",type=Path)
    p.add_argument("v193_original_FP32_answers",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    audit(a.v196_completed_physical_and_QA_artifact,a.v193_original_FP32_answers,a.output)
