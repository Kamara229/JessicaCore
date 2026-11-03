"""CALIBRI v95: protected-size behavior-guided automatic full block candidate search.

Only public original *train* split used. Choose one candidate by real
question correctness on FIT, then open a separate untouched TRAIN_VALID
once. Keep original CB59/CB88/v92/Live immutable. Source bpp and native
runtime pinned by real physical bytes, no KL/NLL or FP32 distribution loss.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v94.provenance import verify_sources
from v92.validate import swap_one_block
from .spec import TARGET,TRAIN_GENERATION,VALID_GENERATION,verify
from .qa import build
from .features import train_energies
from .variants import variants
from .scoring import eval_answers,eval_greedy,fit_rank
from .gates import decisions

def run(original_path:Path,v92_path:Path,v92_report:Path,locked:Path,out:Path):
    verify()
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    lock=json.loads(locked.read_text())
    if lock.get("contract_version")!="CALIBRI_BASELINE_LOCK_2026-11-04":
        raise AssertionError("immutable physical compression baseline absent")
    provenance=verify_sources(original_path,v92_path,v92_report)
    source=original_path.read_bytes();previous=v92_path.read_bytes()
    expected=lock["frozen_reference_sources"]
    if (hashlib.sha256(source).hexdigest()!=expected["CB59"]["sha256"] or
        hashlib.sha256(previous).hexdigest()!=expected["v92_whole_block_inplace_candidate"]["sha256"] or
        len(source)!=expected["CB59"]["complete_archive_bytes"] or
        provenance["rewritten_complete_operator"]!=TARGET):
        raise AssertionError("v95 must preserve exact pinned original size and operator")
    model,info,_=make_model("baseline",SimpleNamespace(original=original_path,folder=out))
    if not info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("source native CPU runtime unexpectedly dense")
    pretrained=original_teacher()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases,originals,options=build(tok)
    print(json.dumps({"stage":"v95_train_only_sources_loaded","splits":originals},default=list),flush=True)
    energy,feature_sources=train_energies(pretrained,TARGET,cases["BoolQ"]["TRAIN"])
    w=pretrained.get_submodule(TARGET).weight.detach().cpu().float().numpy()
    archives,physical=variants(source,previous,w,energy)
    # Train-only search, no access to independent VALID before fit selection.
    fit={}
    original_parent,leaf,original_module,_=swap_one_block(model,source,source,TARGET)
    try:
        for name,blob in archives.items():
            parent,leaf,old,candidate=swap_one_block(model,source,blob,TARGET)
            setattr(parent,leaf,candidate)
            fit[name]={"QA":eval_answers(model,cases,options,"TRAIN"),
                       "greedy":eval_greedy(model,tok,TRAIN_GENERATION)}
            print(json.dumps({"stage":"v95_train_candidate",
                "name":name,"BoolQ_correct":fit[name]["QA"]["BoolQ"]["score"]["correct"],
                "SciQ_correct":fit[name]["QA"]["SciQ"]["score"]["correct"],
                "distinct":fit[name]["greedy"]["mean_distinct_ratio"],
                "physical_bytes":len(blob)}),flush=True)
        ordered,ranks=fit_rank(fit)
        winner=ordered[0]
        setattr(original_parent,leaf,original_module)
        # No re-ranking on VALID: read original public held-out answers exactly once.
        baseline_QA=eval_answers(model,cases,options,"TRAIN_VALID")
        baseline_GEN=eval_greedy(model,tok,VALID_GENERATION)
        teacher_QA=eval_answers(pretrained,cases,options,"TRAIN_VALID")
        parent,leaf,old,new=swap_one_block(model,source,archives[winner],TARGET)
        setattr(parent,leaf,new)
        candidate_QA=eval_answers(model,cases,options,"TRAIN_VALID")
        candidate_GEN=eval_greedy(model,tok,VALID_GENERATION)
    finally:setattr(original_parent,leaf,original_module)
    native_ratio=candidate_GEN["tokens_per_second"]/max(baseline_GEN["tokens_per_second"],1e-9)
    decision=decisions(candidate_QA,baseline_QA,teacher_QA,
                       len(archives[winner]),len(source),native_ratio,baseline_GEN,candidate_GEN)
    # No automated release even if tiny VALID succeeds: semantic review and larger
    # independent sample plus reproducible CPU runs are still outstanding.
    decision["exploratory_only"]=True
    decision["needs_human_review_of_factual_generated_text"]=True
    decision["no_release_without_robust_speed_replication_and_new_unopened_TEST"]=True
    payload={
        "experiment":"CALIBRI_v95_zero_extra_bit_answer_guided_multi_operator_search",
        "frozen_model_original_revision":MODEL_SHA,
        "baseline_lock_contract":lock["contract_version"],
        "pinned_original_source_sha":provenance["CB59_SHA256"],
        "historical_v92_pinned_source_sha":provenance["candidate_SHA256"],
        "new_original_public_TRAIN_only_provenance":originals,
        "source_activation_FIT_only":feature_sources,
        "candidate_physical_sources":physical,
        "FIT_candidate_quality":fit,
        "FIT_model_ranking_without_teacher_matching":ranks,
        "FIT_selected_variant":winner,
        "VALID_unseen_ground_truth_correct_answers":{
            "CB59":baseline_QA,"selected":candidate_QA,"original_FP32":teacher_QA},
        "VALID_original_public_TRAIN_only_greedy":{
            "CB59":baseline_GEN,"selected":candidate_GEN},
        "full_model_native_CPU_speed_ratio":native_ratio,
        "automatic_QUALITY_FIRST_immutable_budget_decision":decision,
        "all_variants_physically_complete_original_archives_unchanged_in_size":True,
        "official_v89_TEST_never_loaded":True,
        "CB59_CB88_v92_live_Jessica_Render_Supabase_main_APK_unchanged":True,
        "limitations":"Single 32+24 FIT and 64+32 independent original train-heldout "
            "scored items, 2 FIT/8 VALID greedy prompts. QUAL outcome is limited; "
            "no external model release, missing semantic human review, CPU speed noisy."}
    path=out/"v95_same_byte_behavior_first_trial.json"
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    if winner not in ("CB59_LOCKED","v92_LOCKED"):
        (out/"v95_selected_RESEARCH_ONLY.cb59").write_bytes(archives[winner])
    print(json.dumps({"stage":"v95_finished_research_only",
      "winner":winner,
      "FIT_correct":{x:{d:fit[x]["QA"][d]["score"]["correct"] for d in ("BoolQ","SciQ")} for x in fit},
      "VALID_CB59_correct":{d:baseline_QA[d]["score"]["correct"] for d in baseline_QA},
      "VALID_selected_correct":{d:candidate_QA[d]["score"]["correct"] for d in candidate_QA},
      "VALID_FP32_correct":{d:teacher_QA[d]["score"]["correct"] for d in teacher_QA},
      "rate_bytes":len(archives[winner]),
      "CPU_full_speed_ratio":native_ratio,
      "decision":decision["decision"],
      "old_v89_TEST_untouched":True},ensure_ascii=False),flush=True)
    return payload
