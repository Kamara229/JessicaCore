"""NRA3 S105 ONE-TIME genuine Calibri+GT prospective Taiga TRAIN-c evaluation.

This is the FIRST & ONLY scoring of the strictly S103 SHA-precommitted
Taiga TRAIN-c256 external UD morphology + multi-positive grammar labels.
S104 original Calibri CB59/GT 12-layer activations and original616 J7
must be physically SHA-frozen BEFORE this script can read UD labels.

Two frozen, physically trained IN-Calibri neural child checkpoints only:
  S99 (trained Taiga TRAIN-b) and S102 (trained Taiga TRAIN-d).
Hashes are precommitted before Taiga TRAIN-c FIRSTREAD in S103.
No model updates, no graph finetuning, no threshold optimization,
no reruns; after evaluation Taiga TRAIN-c becomes irrevocably SPENT.
All model.forward receive ONLY original 12-layer BPE source and masks,
NOT UD event/role/original-word labels. UD VERB event positions are
given ONLY to metrics AFTER both forward passes. Unknown events and
unannotated roles NEVER counted negatives or factual QA.

Scores are UD-annotator teacher-anchored top1/top3 argument token
matching and UPOS/Case/Number/Voice on a new upstream TRAIN-only
file; NOT independent human-verified factuality or true event
recognition. CC BY-SA 4.0 attribution/share-alike; no production
or commercial derivative without legal clearance. All gate0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyInCalibriWeights,UD_UPOS,UD_CASE,UD_NUMBER,UD_VOICE,
)
from research.calibri_nra3.s105_TaigaC_ONE_TIME_teacher_labels_AFTER_true12_SHA import (
    load_source_VERIFY_sha_THEN_OPEN_gold,SOURCE_SHA,GOLD_SHA,STAGE as S105_MAPPING,
)

STAGE="CALIBRI_NRA3_S105_PROSPECTIVE_ONE_SHOT_TAIGA_C_S99_VS_S102_FROZEN_NO_TRAIN"
PREFIX="model.nra3_neural_weights."
EXPECTED={
    "S99":"c77a6964ae4e02cb3aac98d69978a25cec2b491a9624cc9668b3b307699bff8a",
    "S102":"f40dddbc03282a6c45c4f807746275d76a4ba776aa613247e73fbdf7f43329df",
}
N_PARAMS=28887
BOOTSTRAP_REPLICATES=1000
BOOTSTRAP_SEED=1052026
OUTPUT="NRA3_S105_ONE_SHOT_TAIGA_C_S99_VS_S102_FROZEN_FIRSTREAD_METRICS.json"


def require(ok,reason):
    if not bool(ok):
        raise ValueError("NRA3 S105 ONE-SHOT frozen prospective TaigaC: "+reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_one_frozen_child(path:Path,name:str):
    require(path.is_file() and name in EXPECTED and sha(path)==EXPECTED[name],
            "model freeze SHA pinned before S103 Taiga C one-shot label precommit")
    data=load_file(str(path),device="cpu")
    require(data and all(z.startswith(PREFIX) for z in data)
            and sum(t.numel() for t in data.values())==N_PARAMS
            and all(t.dtype==torch.float16
                    and torch.isfinite(t).all() for t in data.values()),
            "ONLY 28,887 actual in-Calibri FP16 train-b/train-d frozen child weights")
    model=NRA3MorphologyInCalibriWeights()
    model.load_state_dict({
        key[len(PREFIX):]:tensor.float().detach().clone()
        for key,tensor in data.items()},strict=True)
    model.eval()
    require(model.output_gate==0
            and not model.candidate_accepted
            and not model.activation_allowed
            and model.budget()["NRA2_independent_mini_areas_preserved"]==8
            and model.budget()["all_12_original_calibri_layers"]==12
            and model.budget()["new_neural_parameters"]==N_PARAMS,
            "strict original Calibri embedded neural 8-mini-area child gate0")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def empty_counter():
    return {"correct":0,"correct_top3":0,"support":0}


@torch.inference_mode()
def evaluate_one_sha_frozen(model,source,aligned):
    """One source-only neural forward per eligible row for each frozen child."""
    hidden=source["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    valid=source["native_BPE_source_valid_mask"]
    require(hidden.ndim==4 and hidden.shape[1]==12 and hidden.shape[-1]==320
            and hidden.dtype==torch.float16 and torch.isfinite(hidden).all()
            and not hidden.requires_grad
            and len(hidden)==len(valid)==len(aligned.eligible),
            "true physical original Calibri GT all12 and source-native BPE only")
    morph={
        k:empty_counter() for k in ("upos","case","number","voice")
    }
    per_morph_class={
        k:{} for k in morph
    }
    by_role={name:empty_counter() for name in ROLE_NAMES}
    row_correct=[]
    row_total=[]
    classified_rows=0
    for idx in torch.where(aligned.eligible)[0].tolist():
        # *** SOURCE-ONLY model forward, absolutely no teacher inputs ***
        out=model(hidden[idx:idx+1].float(),valid[idx:idx+1])
        classified_rows+=1
        for key,classes in (
            ("upos",UD_UPOS),("case",UD_CASE),
            ("number",UD_NUMBER),("voice",UD_VOICE)):
            gold=aligned.morphology[key][idx]
            known=gold!=-100
            if not bool(known.any()):
                continue
            pred=getattr(out,key+"_logits")[0][known].argmax(-1)
            true=gold[known]
            require(bool((true>=0).all())
                    and bool((true<len(classes)).all()),
                    "all supervised offline morphological classes valid")
            n=int(known.sum())
            hits=int((pred==true).sum())
            morph[key]["support"]+=n
            morph[key]["correct"]+=hits
            for label in range(len(classes)):
                mask=true==label
                if not bool(mask.any()):
                    continue
                entry=per_morph_class[key].setdefault(
                    classes[label],{"correct":0,"support":0})
                entry["support"]+=int(mask.sum())
                entry["correct"]+=int((pred[mask]==true[mask]).sum())
        masks=aligned.role.argument_gold_positive[idx]
        known=aligned.role.event_role_supervised[idx]
        count=0;correct=0
        for role_index,name in enumerate(ROLE_NAMES):
            for predicate in torch.where(known[role_index])[0].tolist():
                true_args=masks[role_index,predicate]
                require(bool(true_args.any())
                        and not bool((true_args&~valid[idx]).any()),
                        "teacher-anchored role slot must have positive valid ORIGINAL BPE")
                logits=out.role_arc_logits[0,role_index,predicate]
                prediction=int(logits.argmax())
                topk=logits.topk(min(3,int(valid[idx].sum()))).indices
                hit=int(bool(true_args[prediction]))
                hit3=int(bool(true_args[topk].any()))
                entry=by_role[name]
                entry["support"]+=1
                entry["correct"]+=hit
                entry["correct_top3"]+=hit3
                count+=1;correct+=hit
        row_total.append(count)
        row_correct.append(correct)
    require(classified_rows>=128
            and len(row_total)==classified_rows
            and sum(row_total)>=200
            and sum(x["support"] for x in by_role.values())==sum(row_total),
            "only strictly aligned one-shot UD-annotated event ROLE slots counted")
    def summarize(e):
        return {**e,
                "top1":e["correct"]/e["support"] if e["support"] else None,
                "top3":e["correct_top3"]/e["support"] if e["support"] else None}
    per_role={k:summarize(v) for k,v in by_role.items() if v["support"]}
    aggregate=empty_counter()
    for z in per_role.values():
        for k in aggregate:aggregate[k]+=z[k]
    for key in morph:
        morph[key]=summarize(morph[key])
        per_morph_class[key]={
            k:{**v,"accuracy":v["correct"]/v["support"]}
            for k,v in per_morph_class[key].items()
        }
    return {
        "n_eligible_source_rows":classified_rows,
        "teacher_anchored_role_slots":summarize(aggregate),
        "by_UD_predicate_argument_role":per_role,
        "original_native_BPE_UD_morph_tag_accuracy":morph,
        "by_UD_morph_tag_class":per_morph_class,
        "per_original_SOURCE_ROW_role_correct_counts":row_correct,
        "per_original_SOURCE_ROW_role_total_counts":row_total,
        "never_model_forward_gold_event_or_arg":True,
        "not_autonomous_event_detection_or_user_factuality":True,
    }


def paired_row_bootstrap(s99,s102):
    """Predeclared paired row-cluster bootstrap, not adaptive filtering."""
    a=s99["per_original_SOURCE_ROW_role_correct_counts"]
    b=s102["per_original_SOURCE_ROW_role_correct_counts"]
    support=s99["per_original_SOURCE_ROW_role_total_counts"]
    require(support==s102["per_original_SOURCE_ROW_role_total_counts"]
            and len(a)==len(b)==len(support)>=128,
            "identical one-shot valid source cohort and teacher role slots")
    source=torch.tensor(support,dtype=torch.float64)
    score_a=torch.tensor(a,dtype=torch.float64)
    score_b=torch.tensor(b,dtype=torch.float64)
    require(bool((source>=0).all()) and int(source.sum())>=200,
            "strict teacher-anchored one-shot UD edge/row support")
    g=torch.Generator(device="cpu")
    g.manual_seed(BOOTSTRAP_SEED)
    values=[]
    n=len(a)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled=torch.randint(0,n,(n,),generator=g)
        denom=float(source[sampled].sum())
        if denom<=0:
            continue
        values.append(float((score_b[sampled]-score_a[sampled]).sum())/denom)
    require(len(values)>=.99*BOOTSTRAP_REPLICATES,
            "bootstrap valid source row cluster coverage")
    values.sort()
    return {
        "method":"paired_sentence_cluster_bootstrap_fixed_1000",
        "seed":BOOTSTRAP_SEED,
        "n_replicates":len(values),
        "delta_S102_minus_S99_paired_role_top1":(
            sum(b)/sum(support)-sum(a)/sum(support)),
        "lower_95_percentile":values[int(.025*len(values))],
        "upper_95_percentile":values[int(.975*len(values))],
        "NOT_claim_of_independent_human_factuality":True,
    }


def run_one_time(source_path,source_report,teacher_folder,
                 frozen_S99,frozen_S102,out_path):
    require(not out_path.exists(),"FIRSTREAD score JSON already exists; no reruns allowed")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    # *** FROZEN S99 and S102 checkpoint SHAs BEFORE even opening GOLD. ***
    models={
        "S99":load_one_frozen_child(frozen_S99,"S99"),
        "S102":load_one_frozen_child(frozen_S102,"S102"),
    }
    baseline_shas={k:sha(x) for k,x in (
        ("S99",frozen_S99),("S102",frozen_S102))}
    # Strict original protected616 provenance, ORIGINAL12 tensor SHA,
    # THEN (and only then) separately open protected S103 GOLD.
    source,aligned,audit=load_source_VERIFY_sha_THEN_OPEN_gold(
        source_path,source_report,teacher_folder)
    first=evaluate_one_sha_frozen(models["S99"],source,aligned)
    second=evaluate_one_sha_frozen(models["S102"],source,aligned)
    compare=paired_row_bootstrap(first,second)
    require(all(sha(p)==EXPECTED[k] for k,p in (
              ("S99",frozen_S99),("S102",frozen_S102)))
            and audit["output_gate"]==0 and not audit["candidate_accepted"],
            "frozen S99/S102 child exact precommitted SHA remains bit-identical")
    # Stored metrics aggregate by row/role/class, but no sentence text,
    # original UD third party gold or trained model leaves private CI.
    for d in (first,second):
        del d["per_original_SOURCE_ROW_role_correct_counts"]
        del d["per_original_SOURCE_ROW_role_total_counts"]
    manifest={
        "stage":STAGE,
        "source_gold_alignment_stage":S105_MAPPING,
        "S103_precommitted_prospective_source_SHA":SOURCE_SHA,
        "S103_precommitted_prospective_gold_SHA":GOLD_SHA,
        "original_real12_S104_verified_source_SHA":
            audit["physical_original12_S101_SHA"],
        "frozen_FP16_neural_child_SHAs_precommitted_BEFORE_firstread":baseline_shas,
        "frozen_S99_firstread_UD_metrics":first,
        "frozen_S102_firstread_UD_metrics":second,
        "predeclared_sentence_cluster_paired_difference":compare,
        "strict_original_native_BPE_source_gold_coverage":audit,
        "Taiga_C_prospective_FIRSTREAD_BURNED_never_retrain_retune_rerun":True,
        "no_model_update_optimizer_training_or_selection_in_this_stage":True,
        "original_S99_S102_child_same_in_Calibri_architecture_28887":True,
        "eight_NRA2_neural_mini_areas_untouched":True,
        "original43M_CB59_GT_616J7_protected":True,
        "source_all12_forward_completely_goldblind":True,
        "teacher_event_anchor_ONLY_in_metrics_not_in_neural_forward":True,
        "this_is_offline_UD_taiga_not_independent_human_factuality_test":True,
        "no_claim_new_candidate_616_J7_all_in_bpp_CPU_or_causal_generation":True,
        "CC_BY_SA_4_0_commercial_production_use_BLOCKED_pending_legal_review":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }
    out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_text(json.dumps(
        manifest,sort_keys=True,ensure_ascii=False,indent=2)+"\n",
        encoding="utf8")
    print("NRA3_S105_PROSPECTIVE_TAIGA_C_ONE_SHOT_FROZEN_S99_VS_S102",
          json.dumps({
            "S99_role_top1":first["teacher_anchored_role_slots"]["top1"],
            "S102_role_top1":second["teacher_anchored_role_slots"]["top1"],
            "difference_CI":compare,
            "n_valid_source":first["n_eligible_source_rows"],
            "source_sha":manifest["original_real12_S104_verified_source_SHA"],
            "C_FIRSTREAD_SPENT_FOREVER":True,
            "output_gate":0,
          },sort_keys=True),flush=True)
    return manifest


def main():
    ap=argparse.ArgumentParser()
    for name in ("source","source_report","teacher","s99","s102","out"):
        ap.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=ap.parse_args()
    run_one_time(a.source,a.source_report,a.teacher,a.s99,a.s102,a.out)


if __name__=="__main__":
    main()
