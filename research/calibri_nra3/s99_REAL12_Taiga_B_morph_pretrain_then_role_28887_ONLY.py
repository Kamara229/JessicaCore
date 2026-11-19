"""NRA3 S99: genuine original Calibri12/GT morphology → relational TRAIN.

In-model NEW neural child only (28,887 params), not a standalone
parser. Parent Calibri CB59/GreenTunnel and S59/S65 kept frozen.
S98 physically replayed original616 J7 and captured ALL12 genuine
source layers from NEW licensed UD-Taiga TRAIN-b; source features
SHA-pinned BEFORE opening separate S97 morphology/role UD labels.

Fixed two-stage schedule: morphology-only 10 epochs, joint explicit
morphology plus teacher-anchored UD event→argument/mini-area 24
epochs. Fixed 48 TRAIN /16 same TRAIN-b internal-development sources
per morphology stratum. Train/dev disjoint. Diagnostics never used
for tuning or checkpoint selection. No old S96 internal dev scores,
S79/S86 spent48, Taiga upstream dev/test or PROD datasets.

Metric is teacher-offline POS/Case/Number/Voice tag per real original
BPE and UD predicate-anchored top1 role argument, NOT independent
real-world QA, factuality, knowledge retention or event detection.
CC BY-SA 4.0 non-commercial research pending license compliance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file,save_file

from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyInCalibriWeights,ROLE_NAMES,
)
from research.calibri_nra3.s93_offline_ud_grammatical_role_train_loss_ONLY import (
    NRA3TrainOnlyTargets,graph_train_only_loss,IGNORE,
)
from research.calibri_nra3.s99_nativeBPE_UD_morphology_labels_AFTER_frozen_original import (
    physical_source_then_separate_gold,LABELS_NAME,
)
from research.calibri_nra3.s99_offline_morphology_native_BPE_TRAIN_loss import (
    offline_native_morphology_supervision_loss,
)

STAGE="CALIBRI_NRA3_S99_NEW_TAIGA_TRAIN_B_REAL12_MORPH_PRETRAIN_THEN_UD_ROLE_28887_FP16"
PAID=28887
SEED=991119
MORPH_EPOCHS=10
JOINT_EPOCHS=24
LR=0.0015
WEIGHT_DECAY=0.01
MORPH_LOSS_JOINT_WEIGHT=0.7
PREFIX="model.nra3_neural_weights."
START_WEIGHTS="NRA3_S98_IN_CALIBRI_UNTRAINED_MORPH_28887_FP16.safetensors"
TRAINED_WEIGHTS="NRA3_S99_TAIGA_B_IN_CALIBRI_TRAINED_MORPH_RELATION_28887_FP16.safetensors"
MAPPED_LABELS="NRA3_S99_TAIGA_B_ORIGINAL_BPE_OFFLINE_MORPH_RELATION_LABELS.safetensors"
MANIFEST="NRA3_S99_REAL12_MORPH_PRETRAIN_AND_GRAMMATICAL_INTERNAL_DIAGNOSTICS.json"


def require(value,reason):
    if not bool(value):
        raise ValueError("NRA3 S99 source-first genuine12 morph+role TRAIN: "+reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_8source_stratified_batches(row_ids,mask):
    cohorts=[]
    for stratum in range(4):
        selected=torch.where((row_ids>=64*stratum)
                             &(row_ids<64*stratum+48)&mask)[0]
        require(24<=len(selected)<=48,
                "strict predeclared 48/16 stratum but too little aligned TRAIN")
        cohorts.append(selected)
    n=min(len(a)//2 for a in cohorts)
    require(n>=12,"need fixed balanced batches from every Taiga TRAIN-b stratum")
    return [torch.cat([block[2*i:2*i+2] for block in cohorts])
            for i in range(n)]


def target_subset(aligned,ix):
    truth=aligned.targets
    return NRA3TrainOnlyTargets(
        mention=truth.mention[ix],
        role_argument_token=truth.role_argument_token[ix],
        state=truth.state[ix],
        mini_area=truth.mini_area[ix],
        original_native_valid=truth.original_native_valid[ix],
        evidence=truth.evidence,
    )


def offline_morph_loss(graph,aligned,ix,valid):
    return offline_native_morphology_supervision_loss(
        graph,aligned.upos[ix],aligned.case[ix],
        aligned.number[ix],aligned.voice[ix],valid[ix])


@torch.inference_mode()
def score_teacher_development_once(model,h,valid,aligned,selection):
    """Only offline UD label token matching; NO oracle in model forward."""
    from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
        UD_UPOS,UD_CASE,UD_NUMBER,UD_VOICE,
    )
    source_indices=torch.where(selection)[0]
    require(len(source_indices)>0,"nonempty fixed source-only train/dev slice")
    scores={
        "upos":{"correct":0,"support":0},
        "case":{"correct":0,"support":0},
        "number":{"correct":0,"support":0},
        "voice":{"correct":0,"support":0},
    }
    roles={role:{"correct":0,"support":0} for role in ROLE_NAMES}
    for original_idx in source_indices.tolist():
        net=model(h[original_idx:original_idx+1].float().detach(),
                  valid[original_idx:original_idx+1])
        for name,true,logit in (
            ("upos",aligned.upos[original_idx],net.upos_logits[0]),
            ("case",aligned.case[original_idx],net.case_logits[0]),
            ("number",aligned.number[original_idx],net.number_logits[0]),
            ("voice",aligned.voice[original_idx],net.voice_logits[0]),
        ):
            mask=true!=IGNORE
            pred=logit[mask].argmax(-1)
            n=int(mask.sum())
            scores[name]["support"]+=n
            scores[name]["correct"]+=int((pred==true[mask]).sum())
        truth=aligned.targets.role_argument_token[original_idx]
        for role_id,name in enumerate(ROLE_NAMES):
            predicate_indices=torch.where(truth[role_id]!=IGNORE)[0]
            for p in predicate_indices.tolist():
                correct=int(net.role_arc_logits[0,role_id,p].argmax()==
                            truth[role_id,p])
                roles[name]["correct"]+=correct
                roles[name]["support"]+=1
    for val in scores.values():
        val["accuracy"]=val["correct"]/val["support"] if val["support"] else None
    for val in roles.values():
        val["accuracy"]=val["correct"]/val["support"] if val["support"] else None
    return {
        "n_source_rows":len(source_indices),
        "per_original_BPE_UD_morph_tag_accuracy":scores,
        "per_teacher_anchored_event_to_UD_arg_role_top1":{
            k:z for k,z in roles.items() if z["support"]},
        "this_is_internal_development_not_independent_firstread":True,
        "model_forward_only_frozen_source12_no_UD_teacher":True,
        "no_authentic_event_or_QA_or_factuality_accuracy_claim":True,
    }


def run_once(s98_source,s98_report,s98_child,s97_dir,out):
    require(not out.exists(),"trained S99 checkpoint must not overwrite prior")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    # ORIGINAL source SHA + 616 J7 verified FIRST, ONLY THEN open labels.
    physical,aligned,source_audit=physical_source_then_separate_gold(
        s98_source,s98_report,s97_dir)
    manifest=json.loads(s98_report.read_text())
    require(s98_child.name==START_WEIGHTS
            and sha(s98_child)==manifest["IN_CALIBRI_28887_MORPH_FP16_WEIGHTS_SHA256"]
            and manifest["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and manifest["NRA3_morph_neural_28887_weights_REGISTERED_INSIDE_actual_Calibri_model"]
            and manifest["new_NRA3_train_steps"]==0
            and not manifest["candidate_accepted"],
            "only SHA-pinned S98 genuine Calibri+GT 12layer morph child after original616")
    h=physical["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"].detach()
    valid=physical["native_BPE_source_valid_mask"]
    require(h.ndim==4 and h.shape[1]==12 and h.shape[-1]==320
            and h.dtype==torch.float16 and torch.isfinite(h).all()
            and not h.requires_grad and valid.shape==(h.shape[0],h.shape[2]),
            "frozen true original 12layer native BPE source")
    raw=load_file(str(s98_child),device="cpu")
    require(sum(z.numel() for z in raw.values())==PAID
            and all(k.startswith(PREFIX) for k in raw)
            and all(z.dtype==torch.float16 for z in raw.values()),
            "exactly new 28887 FP16 trained-in-future neural child keys")
    neural=NRA3MorphologyInCalibriWeights()
    neural.load_state_dict({
        k[len(PREFIX):]:w.detach().float().clone() for k,w in raw.items()
    },strict=True)
    require(neural.budget()["new_neural_parameters"]==PAID
            and not neural.is_trained and neural.output_gate==0
            and not neural.candidate_accepted and not neural.activation_allowed,
            "research-only in-Calibri morphological child at gate0")

    neural.eval()
    original_baseline=score_teacher_development_once(
        neural,h,valid,aligned,aligned.same_corpus_development)
    batches=fixed_8source_stratified_batches(
        aligned.original_source_indices,aligned.train)
    opt=torch.optim.AdamW(
        neural.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    trace=[]
    for epoch in range(MORPH_EPOCHS+JOINT_EPOCHS):
        neural.train()
        losses=[]
        for ix in batches:
            opt.zero_grad(set_to_none=True)
            graph=neural(h[ix].float().detach(),valid[ix])
            morph,_=offline_morph_loss(graph,aligned,ix,valid)
            if epoch<MORPH_EPOCHS:
                loss=morph
            else:
                role,_=graph_train_only_loss(
                    graph,target_subset(aligned,ix))
                loss=role+MORPH_LOSS_JOINT_WEIGHT*morph
            require(torch.isfinite(loss),"original source-only morph/UD train loss finite")
            loss.backward()
            nn.utils.clip_grad_norm_(neural.parameters(),1.)
            require(all(p.grad is None or torch.isfinite(p.grad).all()
                        for p in neural.parameters()),
                    "only new neural child gradients, never old CB59/GT")
            opt.step()
            losses.append(float(loss.detach()))
        if epoch in (0,MORPH_EPOCHS-1,MORPH_EPOCHS,
                     MORPH_EPOCHS+9,MORPH_EPOCHS+JOINT_EPOCHS-1):
            trace.append({
                "stage":"morph_only" if epoch<MORPH_EPOCHS else "joint_roles",
                "epoch":epoch,
                "TRAIN_mean_loss":sum(losses)/len(losses),
            })
    neural.eval()
    neural.is_trained=True
    internal=score_teacher_development_once(
        neural,h,valid,aligned,aligned.same_corpus_development)
    train=score_teacher_development_once(
        neural,h,valid,aligned,aligned.train)
    require(sha(s98_child)==manifest["IN_CALIBRI_28887_MORPH_FP16_WEIGHTS_SHA256"],
            "original untrained S98 in-Calibri child physical SHA unchanged")
    saved={PREFIX+k:w.detach().half().cpu().contiguous()
           for k,w in neural.state_dict().items()}
    require(sum(z.numel() for z in saved.values())==PAID
            and all(torch.isfinite(z).all() for z in saved.values()),
            "exactly new 28887 in-Calibri FP16 trained neural weights valid")
    out.mkdir(parents=True,exist_ok=False)
    save_file(saved,str(out/TRAINED_WEIGHTS))
    reread=load_file(str(out/TRAINED_WEIGHTS))
    require(reread.keys()==saved.keys()
            and all(torch.equal(reread[k],v) for k,v in saved.items()),
            "new S99 in-Calibri trained neural FP16 weights readback")
    save_file({
        "offline_UPOS_original_BPE_UNKNOWN_NEG100":aligned.upos.contiguous(),
        "offline_Case_original_BPE_UNKNOWN_NEG100":aligned.case.contiguous(),
        "offline_Number_original_BPE_UNKNOWN_NEG100":aligned.number.contiguous(),
        "offline_Voice_original_BPE_UNKNOWN_NEG100":aligned.voice.contiguous(),
        "offline_UD_argument_roles_original_BPE_NEG100unknown":
            aligned.targets.role_argument_token.contiguous(),
        "offline_UD_event_positive_only_original_BPE_NEG100unknown":
            aligned.targets.mention.contiguous(),
        "offline_NRA2_eight_mini_area_only_positive_else_unknown":
            aligned.targets.mini_area.contiguous(),
        "offline_UD_event_state_only_positive_else_unknown":
            aligned.targets.state.contiguous(),
        "source_valid_original_Calibri_BPE":
            aligned.targets.original_native_valid.contiguous(),
        "source_original_Taiga_B256_row_index":
            aligned.original_source_indices.int().contiguous(),
        "fixed_TRAIN_48_per_64_source_stratum":aligned.train.contiguous(),
        "fixed_INTERNAL_DEV_16_per_64_source_stratum":
            aligned.same_corpus_development.contiguous(),
    },str(out/MAPPED_LABELS))
    report={
        "stage":STAGE,
        "S98_original_true12_source_SHA":manifest["ALL12_CPU_SAFETENSORS_SHA256"],
        "S98_original_untrained_morph_28887_child_SHA":
            manifest["IN_CALIBRI_28887_MORPH_FP16_WEIGHTS_SHA256"],
        "S99_new_trained_morph_28887_child_SHA":sha(out/TRAINED_WEIGHTS),
        "S99_train_ONLY_native_BPE_teacher_gold_SHA":sha(out/MAPPED_LABELS),
        "S97_source_SHA":source_audit["S94_SOURCE_SHA256"],
        "S97_separate_offline_UD_GOLD_SHA":source_audit["S94_separate_gold_SHA256"],
        "source_S99_alignment_coverage":source_audit,
        "new_neural_in_Calibri_trainable_parameters":PAID,
        "new_nominal_FP16_weight_bytes":PAID*2,
        "new_in_calibri_neural_UPOS_Case_Number_Voice_heads":True,
        "eight_old_NRA2_neural_mini_areas_preserved":True,
        "learned_router_all_original12_Calibri_layers":True,
        "in_weight_relation_memory_slots_not_proof_of_knowledge_retention":True,
        "only_NEW_child_neural_weights_trained_BASE_Calibri_GT_frozen":True,
        "original_Calibri_616_J7_frozen_precheck":True,
        "gold_loaded_ONLY_after_S98_true_original12_physical_SHA_verified":True,
        "license_CC_BY_SA_attribution_review_no_commercial_use":True,
        "S96_old_same_Taiga_train_a_internal_dev_SPENT_not_reused":True,
        "S79_S86_old_firstreads_SPENT_not_reused":True,
        "Taiga_original_upstream_dev_test_NEVER_loaded":True,
        "fixed_stage_schedule_no_hparam_or_checkpoint_tuning":True,
        "metrics_NOT_independent_factuality_or_user_event_QA_test":True,
        "protected_active_candidate_616_NOT_validated":True,
        "all_in_bpp_cpu_online_causal_NOT_certified":True,
        "train_config_FIXED":{
            "seed":SEED,"morphology_epochs":MORPH_EPOCHS,
            "morph_and_role_joint_epochs":JOINT_EPOCHS,"lr":LR,
            "weight_decay":WEIGHT_DECAY,
            "joint_morph_loss_weight":MORPH_LOSS_JOINT_WEIGHT,
            "balanced_8source_batches":len(batches),
        },
        "TRAIN_only_loss_trace":trace,
        "S98_random_untrained_same_Taiga_B_INTERNAL_DIAG":original_baseline,
        "S99_trained_same_Taiga_B_INTERNAL_DIAG":internal,
        "S99_trained_Taiga_B_TRAIN_ONLY":train,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (out/MANIFEST).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("NRA3_S99_TRUE_ORIGINAL12_IN_CALIBRI_MORPH_TO_UD_TRAIN",
          json.dumps({
              "aligned_source":source_audit["n_native_BPE_mapping_eligible"],
              "morphology_teacher_UD_BPE":source_audit["n_annotated_morphology_originalBPE_by_head"],
              "same_training_corpus_dev_UPOS_acc_before":
                  original_baseline["per_original_BPE_UD_morph_tag_accuracy"]["upos"]["accuracy"],
              "same_training_corpus_dev_UPOS_acc_after":
                  internal["per_original_BPE_UD_morph_tag_accuracy"]["upos"]["accuracy"],
              "same_training_corpus_dev_teacher_UD_actor_acc_after":
                  internal["per_teacher_anchored_event_to_UD_arg_role_top1"].get("actor",{}).get("accuracy"),
              "new_in_model_neural_params":PAID,"output_gate":0,
          },ensure_ascii=False,sort_keys=True),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser()
    for key in ("s98_source","s98_report","s98_child","s97_dir","out"):
        ap.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    args=ap.parse_args()
    run_once(args.s98_source,args.s98_report,args.s98_child,args.s97_dir,args.out)


if __name__=="__main__":
    main()
