"""Actual v196 Green Tunnel 2-cycle restoration of the REAL trained 135M v186 source.

1. Verify authentic initial v186 physical 11.2MB archive and original CB59;
2. freeze public TRAIN and all QA/VALID cases before teacher capture;
3. train ALL 211 already-paid source codebooks on 49,152-token
   vocabulary, paired contrasts and already-paid 5bit code groups;
4. seal a NEW physical <=0.700 source BEFORE touching any QA/VALID;
5. evaluate actual native packed C answers, paired VALID and CPU speed
   vs original CB59, original FP32 and previous native v186.
No new inference-time weights, no private v89 TEST, no Jessica tests.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.protocol import freeze_cases
from v174.intervene import extract_original
from v174.run import save
from v175.evaluate import generate
from v184.run import first_tokens,actual_answers
from v185.codec import replace_discrete
from v185.run import physical_numerical_control
from v186.run import validation_pairs,speed_pairs
from v185.optim import make_student
from paid_lowrank_v76 import make_model
from v192.grade import summarize as grade_new_qa
from v193.precommit import grading_fingerprint
from v193.questions import cases as new_v193_cases

from . import protocol as P
from .policy import POLICY
from .data import training_inputs,full_vocabulary_references
from .recover import build_trainable_from_paid_source,recover


@torch.inference_mode()
def newly_frozen_public_dev_16(models,tokenizer,source_v193_json,output):
    old=json.loads(Path(source_v193_json).read_text())
    freeze=old["frozen_before_any_model_generation"]
    if not old.get("complete") or (
        freeze["grading_source_frozen_before_inference"]["sha256"]!=
        grading_fingerprint()["sha256"]):
        raise AssertionError("Incorrect independently frozen public original QA source/grader")
    frozen_cases=freeze["full_questions_golds"]
    if frozen_cases!=new_v193_cases():
        raise AssertionError("Historical public questions/gold rewritten after checkpoint training")
    ids=(0,1,2,3,24,25,26,27,48,49,50,51,72,73,74,75)
    selected=[frozen_cases[i] for i in ids]
    old_raw=old["actual_unfiltered_FP32_generations_96_each"]["135M"]
    original={i:old_raw[i]["complete_generation_original"]["verbatim_text"] for i in ids}
    actual={"FP32_existing_135M":original}
    raw={"FP32_existing_135M":[{"case_id":i,"verbatim_text":original[i]} for i in ids]}
    for name,model in models.items():
        rows=[];answers={}
        for case in selected:
            val=generate(model,tokenizer,case["question"],cap=32,stop=True)
            if val["prompt_sha"]!=old_raw[case["case_id"]]["complete_generation_original"]["prompt_sha"]:
                raise AssertionError("Native 135M no longer uses exact original tokenizer/chat prefix")
            rows.append({"case_id":case["case_id"],"question":case["question"],
                         "gold":case["gold"],"actual_output":val})
            answers[case["case_id"]]=val["verbatim_text"]
        raw[name]=rows
        actual[name]=answers
    checked={name:grade_new_qa(selected,answers) for name,answers in actual.items()}
    for name,x in checked.items():
        if x["n"]!=16:raise AssertionError("Partial original 135M public development QA")
    compared={"complete":True,"cases":selected,
        "original_actual_135M_FP32_from_previous_sealed_v193":raw.pop("FP32_existing_135M"),
        "native_compact_actual_QA":raw,
        "preexisting_v192_frozen_grader_sha256":grading_fingerprint()["sha256"],
        "scores":checked,
        "not_new_blind_test_public_dev_only":True,
        "score_changes_not_used_to_tune_the_training_that_just_ran":True,
        "private_v89_TEST_unread":True}
    save(output/"REAL_GREEN_TUNNEL_new_public_DEV_16_full_QA.json",compared)
    print(json.dumps({"stage":"v196_REAL_public_DEV_16",
        "score_by_source":{k:{"correct":v["machine_confirmed"],
            "wrong":v["machine_recognized_wrong"],
            "ungraded":v["ungraded_not_known_wrong"]}
            for k,v in checked.items()}}),flush=True)
    return compared


def run(v186_archive,v161_source,v193_original_control,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(P.SEED);torch.set_num_threads(2)
    raw=Path(v186_archive).read_bytes()
    if (len(raw)!=P.SOURCE_BYTES or
        hashlib.sha256(raw).hexdigest()!=P.INITIAL_135M_SHA256):
        raise AssertionError("Not the verified authentic TRAINED v186 physical start")
    started=CB59(raw)
    if (started.meta["original_unique_parameter_count"]!=P.FROZEN_SOURCE_135M_UNIQUE or
        started.handles["model.embed_tokens.weight"] is not started.handles["lm_head.weight"]):
        raise AssertionError("Not an entire tied original 135M native compressed checkpoint")
    original_archive=extract_original(Path(v161_source))
    if (len(original_archive.blob)!=P.SOURCE_BYTES or
        original_archive.hash()==""):
        raise AssertionError("Original physical CB59 rate baseline unavailable")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    if len(tok)!=P.VOCAB_SIZE:raise AssertionError("Original output vocabulary changed")
    selected,train_pairs,valid_pairs,source_origin=training_inputs(tok)
    dev=freeze_cases(all_cases())
    if set(x["question"] for x in dev)&set(x["question"] for x in new_v193_cases()):
        raise AssertionError("Historical and new public DEV sets unexpectedly intersect")
    if set(x["question"] for x in dev)&set(x.get("question") for x in selected):
        raise AssertionError("New original source training overlaps historic public DEV")
    frozen={
        "experiment":"v196_green_tunnel_continued_135M_v186_full_vocabulary_all_layers",
        "physical_SOURCE_start_SHA256":P.INITIAL_135M_SHA256,
        "original_v186_entire_source_bytes":P.SOURCE_BYTES,
        "original_fp32_unique_parameters":P.FROZEN_SOURCE_135M_UNIQUE,
        "prepaid_complete_file_bpp_before_training":8*len(raw)/P.FROZEN_SOURCE_135M_UNIQUE,
        "full_archive_physical_0p700_max_bytes":P.MAX_BYTES,
        "source_model":MODEL,"original_revision":MODEL_SHA,
        "source_211_paid_codebooks_trainable_every_gradient_step":True,
        "all_49152_vocabulary_logits_teacher_supervision":True,
        "teacher_full_original_fp32_used_TRAIN_ONLY":True,
        "no_new_Green_Tunnel_inference_weight_sidecar":True,
        "public_TRAIN_disjoint_records":source_origin,
        "contradictory_public_TRAIN_pairs":[x["pair_id"] for x in train_pairs],
        "untouched_public_v186_VALID_pairs":[x["pair_id"] for x in valid_pairs],
        "historical_DEV_exact_questions":[x["question"] for x in dev],
        "newer_v193_public_DEV_96_question_sha256":
            [x["question_sha256"] for x in new_v193_cases()],
        "fixed_96_layer_training_targets":list(P.ORIGINAL_LAYER_SCHEDULE),
        "fixed_recovery_cycles":POLICY.cycles,
        "total_steps":POLICY.cycles*POLICY.steps_per_cycle,
        "learning_rate":POLICY.learning_rate,
        "centroid_source_guard":POLICY.fp32_centroid_guard,
        "paired_TRAIN_every_n_steps":POLICY.pair_every_n_steps,
        "word_distribution_teacher_temperature":POLICY.teacher_soft_temperature,
        "all_213?":"NO. Actual unique source matrices count is exactly 211",
        "validation_never_selects_best_epoch":True,
        "train_objective_not_a_quality_or_utility_metric":True,
        "no_KL_NLL_PPL_as_answer_acceptance":True,
        "no_private_v89_TEST":True,
        "no_live_main_Jessica_Render_Supabase_APK":True}
    save(out/"FROZEN_green_tunnel_protocol_BEFORE_FP32_teacher_or_training.json",frozen)
    print(json.dumps({"stage":"v196_FROZEN_all_words_all_layers",
         "source_bpp":frozen["prepaid_complete_file_bpp_before_training"],
         "all_vocab":P.VOCAB_SIZE,"model_matrix_count":P.CODEBOOK_MATRIX_COUNT,
         "cycles":POLICY.cycles,"steps":frozen["total_steps"]}),flush=True)

    teacher=original_teacher().cpu().eval()
    teacher_refs=full_vocabulary_references(
        teacher,selected,POLICY.teacher_soft_temperature)
    teacher_original=teacher.state_dict()
    teacher_weights={name:teacher_original[name].detach().cpu().numpy().copy()
                     for name in set(P.ORIGINAL_LAYER_SCHEDULE)}
    student,sources=build_trainable_from_paid_source(started)
    training=recover(student,sources,teacher_refs,train_pairs,teacher_weights,out)
    save(out/"full_192_step_green_tunnel_all_level_TRAIN_audit.json",training)
    from v176.operators import clear
    clear(sources)
    physical_blob,seal=replace_discrete(started,sources,strict=False)
    if len(physical_blob)!=P.SOURCE_BYTES or len(physical_blob)>P.MAX_BYTES:
        raise AssertionError("Green Tunnel no longer fits actual physical <=0.700 budget")
    dest=out/"CALIBRI_v196_green_tunnel_all_vocab_2cycles_complete_135M.cb59"
    dest.write_bytes(physical_blob)
    sha=hashlib.sha256(dest.read_bytes()).hexdigest()
    if sha!=seal["sha256"] or sha==P.INITIAL_135M_SHA256:
        raise AssertionError("No authentic physically paid changed compressed 135M source")
    if seal["actually_changed_FP16_codebook_cells"]<=0:
        raise AssertionError("All-level Green Tunnel did not alter stored FP16 dictionary")
    seal["trained_by_complete_49152_output_vocabulary"]=True
    seal["all_211_source_matrix_F16_centroids_updated"]=True
    seal["completed_recovery_cycles"]=POLICY.cycles
    seal["based_on_verified_v186_SOURCE_SHA256"]=P.INITIAL_135M_SHA256
    seal["complete_native_QA_not_yet_done_at_seal"]=True
    save(out/"PHYSICAL_GREEN_TUNNEL_SOURCE_SEALED_BEFORE_ANY_QA.json",seal)
    print(json.dumps({"stage":"v196_GREEN_TUNNEL_SOURCE_PHYSICALLY_SEALED",
        "complete_actual_bpp":seal["complete_bpp"],
        "complete_file_bytes":seal["bytes"],
        "SHA256":seal["sha256"],
        "changed_original_5bit_indices":seal["actual_changed_5bit_group_indices"],
        "FP16_dictionary_cells_changed":seal["actually_changed_FP16_codebook_cells"]}),flush=True)

    with TemporaryDirectory(dir=out) as temp:
        temp=Path(temp)
        oldsource=temp/"original_cb59";oldsource.write_bytes(original_archive.blob)
        path186=temp/"source186";path186.write_bytes(raw)
        for folder in ("baseline","v186","v196"):(temp/folder).mkdir()
        cb59,_,_=make_model("baseline",SimpleNamespace(
             original=oldsource,folder=temp/"baseline"))
        cb186,_,_=make_model("baseline",SimpleNamespace(
             original=path186,folder=temp/"v186"))
        cb196,_,_=make_model("baseline",SimpleNamespace(
             original=dest,folder=temp/"v196"))
        if any(x.ndim>=2 for x in cb196.parameters()):
            raise AssertionError("UNPAID trained 2D source at native Green Tunnel inference")
        # Source consistency before any QA outcome touches the run.
        verify=physical_numerical_control(
             student,cb196,sources,train_pairs[0]["left"]["prompt_ids"][:8][None])
        save(out/"FULL_source_native_vs_green_tunnel_training_numerical_control.json",verify)
        models={"FP32":teacher,"CB59":cb59,"CB186":cb186,"GT196":cb196}
        checked_valid=validation_pairs(models,tok,valid_pairs)
        save(out/"HELDOUT_v186_VALID_pair_behavior_full_green_tunnel.json",checked_valid)
        print(json.dumps({"stage":"v196_HELDOUT_pairs",
            "summary":checked_valid["summary"]}),flush=True)
        first=first_tokens(models,tok,dev)
        save(out/"historical_20_first_token_sanity.json",first)
        # Matched identical-work 32-token speed vs ORIGINAL CB59 and GT196
        speed=speed_pairs({"CB59":cb59,"CB186":cb196},tok,dev)
        speed["read_result_as_GT196_vs_original_CB59_not_CB186"]=True
        save(out/"paired_identical_work_CPU_speed_GT196_vs_CB59.json",speed)
        print(json.dumps({"stage":"v196_native_speed_compared_CB59",
            "median":speed["median_speed_ratio_vs_CB59"]}),flush=True)
        actual=actual_answers(models,tok,dev,out)
        from v175.evaluate import summarize as historical_summary
        judged=historical_summary(actual)
        save(out/"historical_24_FULL_NATIVE_FP32_CB59_v186_GT196_QA.json",{
             "full_answers":actual,"conservative_previous_grades":judged,
             "not_new_blind_test":True})
        print(json.dumps({"stage":"v196_historical_24_QA",
             "summary":{key:val["all"] for key,val in judged.items()}}),flush=True)
        frozen_new=newly_frozen_public_dev_16(
             {"CB59":cb59,"CB186":cb186,"GT196":cb196},
             tok,v193_original_control,out)
    if hashlib.sha256(dest.read_bytes()).hexdigest()!=seal["sha256"]:
        raise AssertionError("Physical 135M v196 source mutated since sealed pre QA")
    final={"complete":True,"frozen":frozen,"real_SOURCE":seal,
        "real_training":training,"native_v196_consistency":verify,
        "actual_VALID_pairs":checked_valid,
        "actual_historical_full_QA":judged,
        "actual_new_public_16QA":frozen_new["scores"],
        "native_CPU_matched_CB59_speed":speed,
        "actual_135M_complete_archive_bytes":len(physical_blob),
        "green_tunnel_requires_NO_UNPAID_SEPARATE_INFERENCE_WEIGHTS":True,
        "private_v89_TEST_unread":True,
        "Jessica_never_ran_model_testing":True,
        "diagnostics_and_teacher_TRAIN_only":True}
    save(out/"v196_GREEN_TUNNEL_COMPLETE_IF_ACTUALLY_VERIFIED.json",final)
    print(json.dumps({"stage":"v196_GREEN_TUNNEL_COMPLETE",
       "bpp":seal["complete_bpp"],
       "source_sha256":seal["sha256"],
       "historical_24_machine_confirmed":{k:x["all"]["machine_correct"] for k,x in judged.items()},
       "newer_16_machine_confirmed":{k:x["machine_confirmed"] for k,x in frozen_new["scores"].items()},
       "heldout_opposing_pairs_correct":{k:x["both_opposing_answers_correct_pairs"]
                                 for k,x in checked_valid["summary"].items()},
       "speed_vs_original_CB59":speed["median_speed_ratio_vs_CB59"]}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("real_v186_trained_source",type=Path)
    ap.add_argument("original_v161_cb153_source",type=Path)
    ap.add_argument("original_v193_original_FP32_public_QA",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    run(args.real_v186_trained_source,args.original_v161_cb153_source,
        args.original_v193_original_FP32_public_QA,args.output)
