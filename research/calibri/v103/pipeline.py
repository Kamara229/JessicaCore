"""v103: scientifically valid original TRAIN ground truth, no teacher worship.

Head-only source recode candidate from completed v102 is independently
evaluated with the original model's real chat format; **no model fitting**.
Both immutable original CB59 and v102 actual same-byte archive run through
the same original native C direct-on-demand runtime.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from .spec import ASSERTIONS
from .qa import cases,report
from .text import matched_generation

def run(original:Path,candidate:Path,v102_report:Path,locked:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    lock=json.loads(locked.read_text())
    prior=json.loads(v102_report.read_text())
    base=original.read_bytes()
    recoded=candidate.read_bytes()
    frozen=lock["frozen_reference_sources"]["CB59"]
    base_sha=hashlib.sha256(base).hexdigest()
    candidate_sha=hashlib.sha256(recoded).hexdigest()
    if lock.get("contract_version")!="CALIBRI_BASELINE_LOCK_2026-11-04":
        raise AssertionError("source physical lock unexpectedly absent")
    if (len(base)!=frozen["complete_archive_bytes"] or base_sha!=frozen["sha256"]
        or len(recoded)!=len(base) or candidate_sha!=prior["new_separate_source_sha256"]
        or prior["original_immutable_reference_sha256"]!=base_sha
        or prior["reencoded_complete_archive_bytes"]!=len(base)
        or MODEL_SHA!=lock["original_model_revision"]
        or 8*len(recoded)/EXPECTED_UNIQUE>.7):
        raise AssertionError("v102 model not frozen physically to exact same-byte source")
    for raw in (base,recoded):
        archive=CB59(raw)
        if archive.handles["lm_head.weight"] is not archive.handles["model.embed_tokens.weight"]:
            raise AssertionError("broken input/output embedding shared codebook")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    questions,provenance,answers=cases(tok)
    native_base_root=out/"CB59";native_base_root.mkdir(exist_ok=True)
    native_v102_root=out/"v102";native_v102_root.mkdir(exist_ok=True)
    fp32=original_teacher()
    original_model,base_info,_=make_model("baseline",SimpleNamespace(
        original=original,folder=native_base_root))
    recoded_model,candidate_info,_=make_model("baseline",SimpleNamespace(
        original=candidate,folder=native_v102_root))
    if not all(x["zero_dense_pretrained_2D_hot_weights"]
               for x in (base_info,candidate_info)):
        raise AssertionError("full trained dense source leaked into native model")
    groups=(("original_FP32",fp32),("original_CB59",original_model),
            ("same_byte_v102",recoded_model))
    scores={}
    with torch.inference_mode():
        for name,model in groups:
            scores[name]=report(model,questions,answers)
            print(json.dumps({"stage":"v103_public_original_TRAIN_BoolQ_ground_truth",
                "model":name,
                "correct":scores[name]["summary"]["correct"],
                "total":scores[name]["summary"]["count"],
                "balanced_accuracy":scores[name]["summary"]["balanced_accuracy"],
                "predicted_counts":scores[name]["summary"]["predicted_counts"]}),
                flush=True)
        actual=[]
        for domain,request,correct in ASSERTIONS:
            attempts={}
            for name,model in groups:
                attempts[name]=matched_generation(model,tok,request,correct)
            actual.append({"domain":domain,
                "gold_original_answer":correct,
                "chat_template_prompt_sha256":attempts["original_FP32"][
                    "chat_template_rendered_prompt_sha256"],
                "texts":attempts})
            if len({x["chat_template_rendered_prompt_sha256"]
                    for x in attempts.values()})!=1:
                raise AssertionError("native teacher used incompatible chat format")
            print(json.dumps({"stage":"v103_chat_format_visible_truth_prefix",
                "domain":domain,
                "teacher_prefix_gold":attempts["original_FP32"][
                    "exact_label_prefix_matches_truth"],
                "original_native_prefix_gold":attempts["original_CB59"][
                    "exact_label_prefix_matches_truth"],
                "new_recode_prefix_gold":attempts["same_byte_v102"][
                    "exact_label_prefix_matches_truth"]}),flush=True)
    report_out={
        "experiment":"CALIBRI_v103_fixed_same_byte_v102_pretrained_chat_genuine_gold_answer_audit",
        "original_public_TRAIN_ground_truth_provenance":provenance,
        "original_pretrained_model_revision":MODEL_SHA,
        "original_original_CB59_archive_sha256":base_sha,
        "same_byte_recode_v102_archive_sha256":candidate_sha,
        "original_complete_archive_bytes":len(base),
        "candidate_complete_archive_bytes":len(recoded),
        "all_in_bpp":8*len(recoded)/EXPECTED_UNIQUE,
        "original_tied_head_RMSE":prior["old_original_tied_head_weight_RMSE"],
        "v102_tied_head_RMSE":prior["new_tied_head_weight_RMSE"],
        "real_original_public_labeled_BoolQ_train_results":scores,
        "genuine_original_model_chat_template_texts_and_labels":actual,
        "trained_source_changes_or_optimizer_steps_during_v103":0,
        "new_candidate_selected_using_this_QA":False,
        "teacher_prediction_is_NOT_ground_truth":True,
        "first_text_answer_prefix_only_is_NOT_complete_semantic_accuracy":True,
        "independent_official_v89_TEST_unopened":True,
        "no_modification_to_Live_Jessica_main_Render_Supabase_APK":True,
        "scientific_quality_PASS_not_assumed_from_CI":True,
        "interpretation":"Evaluate whether prepaid physical head source recoding helps truly "
          "labeled real task correctness and coherent generated text. A word/numeric prefix "
          "is strictly indicative, not semantic proof. Original FP32 may also err. No model "
          "is selected, trained or promoted using this public TRAIN evidence."}
    target=out/"v103_same_byte_head_candidate_chat_ground_truth_and_visible_text.json"
    target.write_text(json.dumps(report_out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v103_finished_ground_truth_no_model_selection",
        "real_gold_BoolQ_correct":{n:v["summary"]["correct"] for n,v in scores.items()},
        "original_archive_untouched":True,
        "native_source_bytes":len(recoded),
        "official_TEST_untouched":True}),flush=True)
    return report_out
