"""NRA3 S105: strictly ONE prospective firstread of frozen S99 vs S102.

Preregistered BEFORE any Taiga-C independent test labels are opened:
- S99 original CALIBRI child FP16 SHA frozen at S103.
- S102 original CALIBRI child FP16 SHA frozen at S103.
- S103 official Taiga TRAIN-c source SHA & separate UD role GOLD SHA.
- S104 original Calibri+GT all12 protected616 genuine BPE SOURCE SHA
  physically validated BEFORE any Taiga-C UD gold read.
- Exactly ONE deterministic no-gradient evaluation: top1 argument
  belonging to ANY annotated positive UD role edge, with gold VERB
  anchor used only POST forward as a metric; and explicit native-BPE
  UPOS, Case, Number, Voice accuracy (unknown=-100 excluded).
- Fixed Wilson 95% binomial intervals and paired ROW bootstrap
  S102-minus-S99 role accuracy interval, seed=1051119, 2000 draws.
  No hyperparameter, threshold, architecture, checkpoint selection.
- Separate metadata for structural grouping; targeted stratified
  UD-Taiga TRAIN-c, not naturalistic user factuality or QA.

After this one firstread TRAIN-c is permanently SPENT for ALL later
training/model selection; no retry, no integration or status activation.
CC BY-SA 4.0 unreviewed third-party research only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from safetensors.torch import load_file
from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyInCalibriWeights,
)
from research.calibri_nra3.s105_prospective_TaigaC_offline_gold_AFTER_original12_SHA import (
    load_source_VERIFY_sha_THEN_OPEN_gold,UNKNOWN,
    S104_SOURCE,S104_REPORT,SOURCE_SHA,GOLD_SHA,
)

STAGE="CALIBRI_NRA3_S105_PROSPECTIVE_ONE_SHOT_TAIGA_C_COMPARE_S99_S102_GATE0"
S99_FILE="NRA3_S99_TAIGA_B_IN_CALIBRI_TRAINED_MORPH_RELATION_28887_FP16.safetensors"
S102_FILE="NRA3_S102_TAIGA_D_IN_CALIBRI_ALLVERB_ROLE_28887_TRAINED_FP16.safetensors"
S99_SHA="c77a6964ae4e02cb3aac98d69978a25cec2b491a9624cc9668b3b307699bff8a"
S102_SHA="f40dddbc03282a6c45c4f807746275d76a4ba776aa613247e73fbdf7f43329df"
PREFIX="model.nra3_neural_weights."
EXPECTED_PAID=28887
BOOTSTRAP_SEED=1051119
BOOTSTRAP_DRAWS=2000
RESULT_NAME="NRA3_S105_ONE_TIME_PROSPECTIVE_TAIGAC_S99_VS_S102_REPORT.json"
PREFIX_LOG="NRA3_S105_NO_GOLD_IN_FORWARD_FIRSTREAD_ONCE"


def require(condition,why):
    if not bool(condition):
        raise ValueError("NRA3 S105 irrevocable Taiga-C firstread: "+why)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen_child(path,expected_name,expected_sha):
    """Check physical neural checkpoint BEFORE accessing held test gold."""
    require(path.name==expected_name
            and path.is_file() and sha(path)==expected_sha,
            "frozen pretrained FP16 model checkpoint hash mismatch")
    raw=load_file(str(path),device="cpu")
    require(raw and sum(t.numel() for t in raw.values())==EXPECTED_PAID
            and all(z.startswith(PREFIX) for z in raw)
            and all(t.dtype==torch.float16 and bool(torch.isfinite(t).all())
                    for t in raw.values()),
            "same 28887 in-Calibri neural child FP16 state_dict only")
    m=NRA3MorphologyInCalibriWeights().eval()
    m.load_state_dict(
        {k[len(PREFIX):]:v.float().detach().clone()
         for k,v in raw.items()},strict=True)
    require(m.budget()["new_neural_parameters"]==EXPECTED_PAID
            and m.budget()["NRA2_independent_mini_areas_preserved"]==8
            and m.budget()["all_12_original_calibri_layers"]==12
            and m.output_gate==0
            and not m.candidate_accepted and not m.activation_allowed,
            "frozen actual Calibri child has eight NRA2 microheads gate0")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def empty_summary():
    return {
        "role":{"correct":0,"slots":0},
        "roles":{k:{"correct":0,"slots":0} for k in ROLE_NAMES},
        "morph":{k:{"correct":0,"support":0} for k in
                 ("upos","case","number","voice")},
    }


def add_metric(a,b):
    a["role"]["correct"]+=b["role"]["correct"]
    a["role"]["slots"]+=b["role"]["slots"]
    for r in ROLE_NAMES:
        for k in ("correct","slots"):
            a["roles"][r][k]+=b["roles"][r][k]
    for m in a["morph"]:
        for k in ("correct","support"):
            a["morph"][m][k]+=b["morph"][m][k]


def score_row(model,original_all12,valid,aligned,i):
    """NO gold passed into model.forward. Teacher used only to score."""
    with torch.inference_mode():
        o=model(original_all12[i:i+1].float().detach(),valid[i:i+1])
    metric=empty_summary()
    for name in metric["morph"]:
        gold=aligned.morphology[name][i]
        known=gold!=UNKNOWN
        n=int(known.sum())
        if not n:
            continue
        guess=getattr(o,name+"_logits")[0][known].argmax(-1)
        metric["morph"][name]["correct"]=int((guess==gold[known]).sum())
        metric["morph"][name]["support"]=n
    for role_id,name in enumerate(ROLE_NAMES):
        for predicate in torch.where(
                aligned.role.event_role_supervised[i,role_id])[0].tolist():
            positive=aligned.role.argument_gold_positive[i,role_id,predicate]
            require(bool(positive.any()),
                    "every supervised held event role slot has positive native BPE argument")
            guess=int(o.role_arc_logits[0,role_id,predicate].argmax())
            hit=int(bool(positive[guess]))
            metric["role"]["correct"]+=hit
            metric["role"]["slots"]+=1
            metric["roles"][name]["correct"]+=hit
            metric["roles"][name]["slots"]+=1
    return metric


def wilson95(success,total):
    if not total:
        return None
    z=1.959963984540054
    p=success/total
    denom=1+z*z/total
    center=(p+z*z/(2*total))/denom
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return [max(0,center-half),min(1,center+half)]


def summarize(rows):
    aggregate=empty_summary()
    for r in rows:
        add_metric(aggregate,r)
    for d in [aggregate["role"],*aggregate["roles"].values(),
              *aggregate["morph"].values()]:
        denom=d.get("slots",d.get("support",0))
        correct=d["correct"]
        d["accuracy"]=correct/denom if denom else None
        d["wilson95"]=wilson95(correct,denom)
    return aggregate


def paired_row_bootstrap_s102_minus_s99(rows_s99,rows_s102):
    require(len(rows_s99)==len(rows_s102) and len(rows_s99)>=96,
            "precommitted paired prospective held test source rows")
    rnd=random.Random(BOOTSTRAP_SEED)
    n=len(rows_s99)
    diffs=[]
    for _ in range(BOOTSTRAP_DRAWS):
        chosen=(rnd.randrange(n) for _ in range(n))
        a_h=a_n=b_h=b_n=0
        for i in chosen:
            a=rows_s99[i]["role"]
            b=rows_s102[i]["role"]
            require(a["slots"]==b["slots"],
                    "identical same original native BPE UD role evidence both models")
            a_h+=a["correct"];a_n+=a["slots"]
            b_h+=b["correct"];b_n+=b["slots"]
        diffs.append(b_h/b_n-a_h/a_n if a_n and b_n else 0)
    diffs.sort()
    return {
        "method":"paired SOURCE-ROW bootstrap 2000 draws fixed seed, role-slot accuracy diff",
        "seed":BOOTSTRAP_SEED,"draws":BOOTSTRAP_DRAWS,
        "S102_minus_S99_95_percentile_CI":[
            diffs[int(0.025*(BOOTSTRAP_DRAWS-1))],
            diffs[int(0.975*(BOOTSTRAP_DRAWS-1))],
        ],
        "diagnostic_only_no_hyperparameter_or_threshold_selection":True,
    }


def ONE_FIRSTREAD_s99_vs_s102_original12(
        *,s104_source,s104_report,s103_gold_folder,
        s99_weights,s102_weights,out):
    require(not out.exists(),"one-shot evaluator can NEVER overwrite or rerun outputs")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    require(s104_source.name==S104_SOURCE
            and s104_report.name==S104_REPORT,
            "only actual S104 genuine original firstread source artifact")
    report=json.loads(s104_report.read_text())
    require(report["stage"]==
                "CALIBRI_NRA3_S104_FIRSTREAD_TAIGA_TRAINC256_REAL43MGT_ALL12_SOURCEONLY_AFTER616"
            and report["S103_TAIGA_C_FIRSTREAD_SOURCE256_SHA256"]==SOURCE_SHA
            and report["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and report["source_ONLY_no_S103_FIRSTREAD_C_UD_gold_loaded_in_original_CPU"]
            and report["NRA3_morph_neural_28887_weights_REGISTERED_INSIDE_actual_Calibri_model"]
            and report["all_original_43M_GT_CB59_params_sha_pre_and_post_identical"]
            and not report["candidate_accepted"]
            and report["new_NRA3_train_steps"]==0
            and sha(s104_source)==report["ALL12_CPU_SAFETENSORS_SHA256"],
            "verify genuine gold-blind original12 protected616 physical source SHA FIRST")
    # Absolutely BEFORE opening Taiga-C held gold:
    model99=frozen_child(s99_weights,S99_FILE,S99_SHA)
    model102=frozen_child(s102_weights,S102_FILE,S102_SHA)
    print("NRA3_S105_GOLD_STILL_CLOSED_ORIGINAL12_SHA_OLD616_AND_TWO_FROZEN_CHILD_SHAS_OK",
          json.dumps({"original12_SHA":sha(s104_source),
                      "S99_child_SHA":sha(s99_weights),
                      "S102_child_SHA":sha(s102_weights)},sort_keys=True),
          flush=True)
    # The FIRST and ONLY access to prospective Taiga-C gold is below.
    original,aligned,audit=load_source_VERIFY_sha_THEN_OPEN_gold(
        s104_source,s104_report,s103_gold_folder)
    h=original["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"].detach()
    valid=original["native_BPE_source_valid_mask"]
    require(h.ndim==4 and h.shape[1]==12 and h.shape[-1]==320
            and h.dtype==torch.float16 and bool(torch.isfinite(h).all())
            and not h.requires_grad
            and int(aligned.eligible.sum())>=128
            and aligned.counts["positive_event_argument_edges"]>=200,
            "S104 true original12 held dataset meets preregistered native BPE coverage")
    valid_rows=torch.where(aligned.eligible)[0].tolist()
    rows99=[]
    rows102=[]
    per_category={str(z):{"S99":[],"S102":[]} for z in range(4)}
    # No gradients, no model selection, no mutation, exactly once
    # evaluating every PRECOMMITTED original source row eligible.
    for i in valid_rows:
        x99=score_row(model99,h,valid,aligned,i)
        x102=score_row(model102,h,valid,aligned,i)
        rows99.append(x99)
        rows102.append(x102)
        category=str(int(aligned.original_source_row_indices[i])//64)
        per_category[category]["S99"].append(x99)
        per_category[category]["S102"].append(x102)
    a=summarize(rows99)
    b=summarize(rows102)
    require(a["role"]["slots"]==b["role"]["slots"]
            and a["role"]["slots"]>=128,
            "same protected firstread one-time UD graph measured for both frozen models")
    strata={k:{"n_source":len(v["S99"]),
               "S99":summarize(v["S99"]),
               "S102":summarize(v["S102"])} for k,v in per_category.items()}
    bootstrap=paired_row_bootstrap_s102_minus_s99(rows99,rows102)
    diff=b["role"]["accuracy"]-a["role"]["accuracy"]
    result={
        "stage":STAGE,
        "original_CALIBRI_true12_C_firstread_physical_SHA":sha(s104_source),
        "precommitted_Taiga_C_SOURCE_SHA":SOURCE_SHA,
        "precommitted_SEPARATE_UD_GOLD_SHA":GOLD_SHA,
        "S99_PINNED_in_Calibri_trained_FP16_SHA":S99_SHA,
        "S102_PINNED_in_Calibri_trained_FP16_SHA":S102_SHA,
        "upstream_Taiga_train_c_git_blob":"16ddc9c25dc1ab56d809c4b155f4eb18942b0e8f",
        "CC_BY_SA_4_license_and_commercial_prod_USE_blocked":True,
        "n_original_precommitted_source":256,
        "n_original_source_C_valid_BPE":int(h.shape[0]),
        "n_UD_role_morphology_exact_aligned_firstread":len(valid_rows),
        "n_gold_UD_events_firstread":aligned.counts["known_verb_predicate_BPE_positions"],
        "n_gold_positive_argument_edges_firstread":aligned.counts["positive_event_argument_edges"],
        "source_exclusions_only_if_unsafe_original_native_BPE":audit["row_exclusions"],
        "S99_one_shot_C_firstread":a,
        "S102_one_shot_C_firstread":b,
        "S102_minus_S99_firstread_UD_argument_top1_accuracy":diff,
        "paired_row_bootstrap_difference":bootstrap,
        "source_stratified_64_by_fixed_original_UD_categories":strata,
        "all_8_original_NRA2_neural_miniareas_retained":True,
        "both_child_weights_28_887_params_REGISTERED_in_CALIBRI":True,
        "original_CB59_GreenTunnel_S59_S65_and_protected616_UNTOUCHED":True,
        "all_tests_gold_blind_until_source_and_models_SHA_verified":True,
        "teacher_EVENT_anchor_only_POST_FORWARD_metric_NOT_autonomous":True,
        "no_event_precision_or_real_world_factuality_accuracy":True,
        "S96_S99_S102_same_corpus_developer_diagnostics_SPENT_never_tuned":True,
        "Taiga_C_firstread_PERMANENTLY_SPENT_for_training_tuning_RETEST":True,
        "metric_protocol_fixed_BEFORE_one_shot_firstread":True,
        "no_retraining_no_threshold_no_checkpoint_or_architecture_selection":True,
        "no_new_integrated_616_J7_or_online_autoregressive_cert":True,
        "no_all_in_bpp_le07_CPU_ge90_cb59_cert":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/RESULT_NAME).write_text(
        json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("NRA3_S105_PROSPECTIVE_ONCE_ONLY_TAIGAC_S99_VS_S102_RESULT",
          json.dumps({
              "n_eligible_test":len(valid_rows),
              "n_role_slots":a["role"]["slots"],
              "S99_held_teacher_arg_top1":a["role"]["accuracy"],
              "S102_held_teacher_arg_top1":b["role"]["accuracy"],
              "S102_minus_S99":diff,
              "bootstrap_CI":bootstrap["S102_minus_S99_95_percentile_CI"],
              "held_morph_UPOS_S99":a["morph"]["upos"]["accuracy"],
              "held_morph_UPOS_S102":b["morph"]["upos"]["accuracy"],
              "output_gate":0,"firstread_SPENT":True,
          },sort_keys=True),flush=True)
    return result


def main():
    a=argparse.ArgumentParser()
    for key in ("s104_source","s104_report","s103_gold_folder",
                "s99_weights","s102_weights","out"):
        a.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    opt=a.parse_args()
    ONE_FIRSTREAD_s99_vs_s102_original12(
        s104_source=opt.s104_source,s104_report=opt.s104_report,
        s103_gold_folder=opt.s103_gold_folder,
        s99_weights=opt.s99_weights,s102_weights=opt.s102_weights,
        out=opt.out)


if __name__=="__main__":
    main()
