"""Real v186 paired behavioral source training => physical native CPU QA."""
import argparse,hashlib,json,statistics
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.protocol import freeze_cases
from v174.intervene import extract_original
from v174.run import save
from v175.evaluate import generate,summarize
from v184.run import first_tokens,actual_answers
from v185.optim import make_student
from v185.codec import replace_discrete
from v185.run import physical_numerical_control
from paid_lowrank_v76 import make_model
from . import protocol as p
from .pairs import raw_pairs,build_pairs,manifest
from .fit import train


@torch.inference_mode()
def validation_pairs(models,tokenizer,valid):
    if len(valid)!=p.VALID_PAIR_COUNT:raise AssertionError("Incomplete heldout VALID")
    result={name:[] for name in models}
    for pair in valid:
        for name,model in models.items():
            items=[]
            for side in ("left","right"):
                row=pair[side]
                ids=row["prompt_ids"][None]
                logits=model(input_ids=ids,use_cache=False).logits[0,-1]
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite paired VALID logits")
                first=int(logits.argmax())
                items.append({"side":side,"question":row["question"],
                    "gold":row["answer"],"gold_first_token":row["first_expected_id"],
                    "predicted_first_token":first,
                    "first_token_correct":first==row["first_expected_id"],
                    "prompt_sha256":row["prompt_sha"]})
            result[name].append({
                "pair_id":pair["pair_id"],"domain":pair["domain"],"sides":items,
                "both_first_tokens_correct":all(x["first_token_correct"] for x in items),
                "predicted_tokens_different":items[0]["predicted_first_token"]!=
                                              items[1]["predicted_first_token"]})
    compact={}
    for name,items in result.items():
        flat=[side for item in items for side in item["sides"]]
        compact[name]={"paired_heldout_questions":len(flat),
            "both_opposing_answers_correct_pairs":sum(x["both_first_tokens_correct"] for x in items),
            "genuine_prediction_changes_per_pair":sum(x["predicted_tokens_different"] for x in items),
            "individual_first_token_correct":sum(x["first_token_correct"] for x in flat),
            "unique_first_token_ids":len(set(x["predicted_first_token"] for x in flat)),
            "all_predictions_one_token":len(set(x["predicted_first_token"] for x in flat))==1,
            "by_domain":{domain:{
                "pairs":len([x for x in items if x["domain"]==domain]),
                "both_correct":sum(x["both_first_tokens_correct"] for x in items if x["domain"]==domain)}
                for domain in p.TRAIN_DOMAINS}}
    return {"results":result,"summary":compact,
        "heldout_VALID_not_for_training_selection":True}


@torch.inference_mode()
def validation_sample_text(models,tok,valid):
    selected=valid[::3]  # deterministic: four pairs, one each domain
    if len(selected)!=4 or set(x["domain"] for x in selected)!=set(p.TRAIN_DOMAINS):
        raise AssertionError("VALID text samples not all domains")
    outputs=[]
    for pair in selected:
        for side in ("left","right"):
            row=pair[side]
            answers={}
            for name,model in models.items():
                result=generate(model,tok,row["question"],cap=16,stop=True)
                answers[name]={"verbatim_text":result["verbatim_text"],
                    "token_ids":result["token_ids"],"eos":result["eos"],
                    "cap_hit_without_eos":result["cap_hit_without_eos"],
                    "distinct_fraction":result["distinct_token_fraction"]}
            outputs.append({"domain":pair["domain"],"pair_id":pair["pair_id"],
                "question":row["question"],"expected":row["answer"],
                "outputs":answers})
    return outputs


@torch.inference_mode()
def speed_pairs(models,tok,cases):
    chosen=[x for x in cases if x["question_id"] in p.SPEED_IDS]
    if len(chosen)!=4:raise AssertionError("Missing frozen CPU speed cases")
    for name in ("CB59","CB186"):
        generate(models[name],tok,chosen[0]["prompt"],cap=8,stop=False)
    rows=[]
    for repeat in range(2):
        for index,case in enumerate(chosen):
            order=("CB59","CB186") if (repeat+index)%2==0 else ("CB186","CB59")
            seconds={}
            for name in order:
                result=generate(models[name],tok,case["prompt"],cap=32,stop=False)
                if len(result["token_ids"])!=32:
                    raise AssertionError("CPU benchmark not same work")
                seconds[name]=result["wall_seconds"]
            rows.append({"repeat":repeat,"case_id":case["question_id"],
                "wall_seconds":seconds,"speed_ratio_CB186_vs_CB59":
                    seconds["CB59"]/seconds["CB186"]})
    return {"paired_32_token_runs":rows,
        "median_speed_ratio_vs_CB59":statistics.median(
            x["speed_ratio_CB186_vs_CB59"] for x in rows),
        "8_paired_runs":len(rows)==8,
        "includes_prefill":True,"not_quality_gate":True}


def run(path,out):
    torch.set_num_threads(2)
    torch.manual_seed(p.SEED)
    out.mkdir(parents=True,exist_ok=True)
    archive=extract_original(path)
    if archive.hash()!=p.SOURCE_SHA or len(archive.blob)!=p.SOURCE_BYTES:
        raise ValueError("Source hash mismatch: no experimental data accepted")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    dev=freeze_cases(all_cases())
    dev_q={item["question"] for item in dev}
    train_pairs=build_pairs(tok,"TRAIN",dev_q)
    valid_pairs=build_pairs(tok,"VALID",dev_q)
    train_q={x[s]["question"] for x in train_pairs for s in ("left","right")}
    valid_q={x[s]["question"] for x in valid_pairs for s in ("left","right")}
    if train_q & valid_q or train_q & dev_q or valid_q & dev_q:
        raise AssertionError("TRAIN/VALID/DEV question collision")
    if len(train_q)!=48 or len(valid_q)!=24:
        raise AssertionError("Incomplete paired split")
    locked={"experiment":"v186 original 5-bit codes balanced paired behavior TRAIN",
        "original_complete_CB59_sha":p.SOURCE_SHA,
        "original_complete_CB59_bytes":p.SOURCE_BYTES,
        "original_unique_parameters":p.TOTAL_UNIQUE,
        "max_complete_physical_0p7_bytes":p.MAX_BYTES,
        "model":MODEL,"model_revision":MODEL_SHA,
        "seed":p.SEED,"steps":p.STEPS,
        "total_24_opposing_TRAIN_pairs":manifest(train_pairs),
        "untouched_12_opposing_VALID_pairs":manifest(valid_pairs),
        "frozen_24_historical_DEV_question_IDs":list(p.CASE_IDS),
        "frozen_first_token_DEV_ids":list(p.SMOKE_IDS),
        "fixed_32_token_CPU_speed_question_IDs":list(p.SPEED_IDS),
        "all_96_train_target_source_operators":list(p.SCHEDULE),
        "learning_rate_codebooks":p.CENTROID_LR,
        "opposing_answer_margin":p.MARGIN,
        "contrastive_margin_weight":p.PAIR_MARGIN_WEIGHT,
        "max_5bit_swaps_each_step":p.MAX_INDEX_SWAPS_PER_STEP,
        "optimization_only_known_answer_PUBLIC_TRAIN":True,
        "train_pair_balance_across_four_domains":True,
        "no_original_FP32_teacher_behavior_pseudolabels":True,
        "valid_pair_or_DEV_scores_never_for_fitting":True,
        "no_private_v89_TEST_access":True,
        "no_Jessica_model_tests_or_live_changes":True,
        "reconstructed_model_not_used_for_native_inference":True,
        "train_CE_NOT_a_quality_acceptance":True}
    save(out/"protocol_locked_before_ANY_training.json",locked)
    print(json.dumps({"stage":"frozen_pairs","TRAIN":len(train_pairs),
        "VALID":len(valid_pairs),"historical_DEV":24}),flush=True)
    teacher=original_teacher()
    # Preserved original *actual source weights* act as a numerical prior
    # for changed codes, but NEVER as answer targets or QA ground truth.
    pretrained=teacher.state_dict()
    teacher_weights={key:pretrained[key].detach().cpu().numpy()
                     for key in set(p.SCHEDULE)}
    student,sources=make_student(archive)
    def partial(step,curve):
        save(out/"TRAIN_status_before_DEV.json",{
            "completed_gradient_steps":step,"last_four":curve[-4:],
            "accepted_code_changes_so_far":sum(x["accepted_indices"] for x in curve)})
    training=train(student,sources,train_pairs,teacher_weights,partial)
    save(out/"TRAIN_full_audit.json",training)
    source_bytes,physical=replace_discrete(archive,sources,strict=True)
    dest=out/"CALIBRI_v186_balanced_5bit_full_source.cb59"
    dest.write_bytes(source_bytes)
    if hashlib.sha256(dest.read_bytes()).hexdigest()!=physical["sha256"]:
        raise AssertionError("Packed full source post-file SHA drift")
    physical["physical_closed_before_any_development_or_VALID_measurement"]=True
    save(out/"SEALED_source_before_any_QA.json",physical)
    print(json.dumps({"stage":"physical_sealed","bytes":physical["bytes"],
        "bpp":physical["complete_bpp"],
        "actual_reassigned_groups":physical["actual_changed_5bit_group_indices"],
        "changed_dictionary_values":physical["actually_changed_FP16_codebook_cells"],
        "sha256":physical["sha256"]}),flush=True)
    with TemporaryDirectory(dir=out) as folder:
        folder=Path(folder)
        original=folder/"original.cb59";original.write_bytes(archive.blob)
        (folder/"source").mkdir()
        (folder/"candidate").mkdir()
        cb59,_,_=make_model("baseline",SimpleNamespace(
            original=original,folder=folder/"source"))
        cb186,_,_=make_model("baseline",SimpleNamespace(
            original=dest,folder=folder/"candidate"))
        if any(t.ndim>=2 for t in cb186.parameters()):
            raise AssertionError("Actual native runtime has unpaid dense source")
        numeric=physical_numerical_control(
            student,cb186,sources,train_pairs[0]["left"]["prompt_ids"][:8][None])
        save(out/"source_NATIVE_vs_96step_TRAIN_surrogate.json",numeric)
        print(json.dumps({"stage":"native_match",**numeric}),flush=True)
        models={"FP32":teacher,"CB59":cb59,"CB186":cb186}
        valid=validation_pairs(models,tok,valid_pairs)
        save(out/"VALID_balanced_pair_behaviors.json",valid)
        print(json.dumps({"stage":"paired_heldout_VALID","summary":valid["summary"]}),flush=True)
        texts=validation_sample_text(models,tok,valid_pairs)
        save(out/"VALID_full_answer_sample.json",texts)
        first=first_tokens(models,tok,dev)
        save(out/"historical_DEV_20_first_tokens.json",first)
        print(json.dumps({"stage":"historical_first_tokens",
            "match_FP32":{k:x["FP32_first_matches"] for k,x in first.items()},
            "distinct_first_ids":{k:x["distinct_first_tokens"] for k,x in first.items()}}),flush=True)
        speed=speed_pairs(models,tok,dev)
        save(out/"paired_CPU_speed_8_runs.json",speed)
        print(json.dumps({"stage":"native_CPU_speed",
            "median":speed["median_speed_ratio_vs_CB59"]}),flush=True)
        answers=actual_answers(models,tok,dev,out)
        graded=summarize(answers)
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=physical["sha256"]:
            raise AssertionError("Physical source mutated after evaluation")
        completed={"completed":True,"locked_protocol":locked,
            "complete_source":physical,"train_audit":training,
            "native_numerics":numeric,"independent_VALID_pairs":valid,
            "VALID_full_text_sample":texts,"historical_DEV_first_tokens":first,
            "historical_DEV_24_full_raw_answers":answers,
            "historical_DEV_grades":graded,"native_CPU_speed":speed,
            "private_TEST_untouched":True,"Jessica_did_not_run_tests":True,
            "native_source_measured_not_simulated":True}
        save(out/"v186_complete.json",completed)
        print(json.dumps({"stage":"complete","complete_bpp":physical["complete_bpp"],
            "physical_sha256":physical["sha256"],
            "changed_5bit_codes":physical["actual_changed_5bit_group_indices"],
            "speed_vs_CB59":speed["median_speed_ratio_vs_CB59"],
            "VALID":valid["summary"],
            "DEV":{k:v["all"] for k,v in graded.items()}}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    run(args.source,args.out)
