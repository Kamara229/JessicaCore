"""S96 real 12-layer Calibri + GT, ONLY new NRA3 neural-child UD training.

Genuine S95 original Calibri CB59 hidden states are SHA-frozen after
protected616 J7. Offline Taiga UD source->native BPE labels are opened
only after physical original-source hash verification. The learned
23k-scale NRA3 neural child remains part of Calibri state_dict, NOT an
external grammar service. Eight NRA2 mini-areas and four all12 routers
remain within the registered child. No old weights updated.

One predeclared 48/16 per 64 UD stratum TRAIN/internal-diagnostic
split. Internal diagnostics are NOT blind first-read or naturalistic
QA, and teacher event anchors are used ONLY in metrics, never forward.
No S79/S86 spent firstreads, output gate0, no production license grant.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file,save_file

from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    NRA3InCalibriNeuralWeights,ROLE_NAMES,
)
from research.calibri_nra3.s93_offline_ud_grammatical_role_train_loss_ONLY import (
    NRA3TrainOnlyTargets,graph_train_only_loss,IGNORE,
)
from research.calibri_nra3.s96_taiga_originalBPE_offline_UD_targets import (
    physical_source_then_separate_gold,LABELS_NAME,
)

STAGE="CALIBRI_NRA3_S96_TAIGA_UD_ORIGINAL12_NEW22311_FP16_TRAIN_SHADOW"
SEED,EPOCHS,LR,WD=961119,28,0.002,0.01
PAID=22311
START_WEIGHTS="NRA3_S95_IN_CALIBRI_UNTRAINED_22311_NEURAL_WEIGHTS_FP16.safetensors"
WEIGHTS="NRA3_S96_TAIGA_UD_TRAINED_IN_CALIBRI_22311_FP16.safetensors"
MANIFEST="NRA3_S96_TAIGA_UD_TRAINED_RELATIONS_INTERNAL_DIAG.json"
PREFIX="model.nra3_neural_weights."


def require(ok,why):
    if not bool(ok):
        raise ValueError("NRA3 S96 original12 Taiga TRAIN-only: "+why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def subset(original:NRA3TrainOnlyTargets,ix:torch.Tensor):
    return NRA3TrainOnlyTargets(
        mention=original.mention[ix],
        role_argument_token=original.role_argument_token[ix],
        state=original.state[ix],
        mini_area=original.mini_area[ix],
        original_native_valid=original.original_native_valid[ix],
        evidence=original.evidence,
    )


def fixed_stratum_batches(original_ids,train):
    groups=[]
    for k in range(4):
        ix=torch.where((original_ids>=k*64)
                       &(original_ids<k*64+48)&train)[0]
        require(24<=len(ix)<=48,
                "S96 needs >=24 exact original BPE TRAIN alignments per group")
        groups.append(ix)
    batches=min(len(z)//2 for z in groups)
    require(batches>=12,"not enough balanced UD TRAIN batches")
    return [torch.cat([group[i*2:i*2+2] for group in groups])
            for i in range(batches)]


@torch.inference_mode()
def teacher_anchored_role_top1(net,h,valid,targets,mask):
    """Gold event position for score ONLY, not inferred event detection."""
    total=0
    hits=0
    by_role={x:{"correct":0,"support":0} for x in ROLE_NAMES}
    for i in torch.where(mask)[0].tolist():
        logits=net(h[i:i+1].float().detach(),valid[i:i+1]).role_arc_logits
        for role in range(len(ROLE_NAMES)):
            positions=torch.where(targets.role_argument_token[i,role]!=IGNORE)[0]
            for predicate in positions.tolist():
                arg=int(targets.role_argument_token[i,role,predicate])
                pred=int(logits[0,role,predicate].argmax())
                score=int(pred==arg)
                total+=1;hits+=score
                slot=by_role[ROLE_NAMES[role]]
                slot["correct"]+=score
                slot["support"]+=1
    require(total>0,"internal UD role evaluation has labelled argument edges")
    return {
        "annotated_teacher_event_role_edges":total,
        "n_correct_original_BPE_role_argmax":hits,
        "teacher_anchored_native_BPE_role_top1_accuracy":hits/total,
        "by_role":{k:v for k,v in by_role.items() if v["support"]>0},
        "oracle_event_anchor_used_ONLY_for_metric":True,
        "not_end_to_end_event_or_factuality_quality":True,
    }


def train_new_in_calibri_neural_only(s95_source,s95_report,s95_child,s94,out):
    require(not out.exists(),"no overwrite/retest of saved neural checkpoint")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    source,aligned,audit=physical_source_then_separate_gold(
        s95_source,s95_report,s94)
    source_manifest=json.loads(s95_report.read_text())
    require(s95_child.name==START_WEIGHTS
            and sha(s95_child)==source_manifest["IN_CALIBRI_22311_NEW_FP16_WEIGHTS_SHA256"]
            and source_manifest["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and source_manifest["NRA3_neural_22311_weights_REGISTERED_INSIDE_actual_Calibri_model"]
            and not source_manifest["candidate_accepted"],
            "pinned original S95 protected616 and REAL in-model neural child only")
    hidden=source["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"].detach()
    valid=source["native_BPE_source_valid_mask"]
    require(hidden.ndim==4 and hidden.shape[1]==12 and hidden.shape[-1]==320
            and hidden.shape[0]==len(aligned.original_source_indices)
            and hidden.dtype==torch.float16 and not hidden.requires_grad
            and torch.isfinite(hidden).all(),
            "frozen original Calibri all-12 physical BPE hidden states")
    raw=load_file(str(s95_child),device="cpu")
    require(raw and sum(z.numel() for z in raw.values())==PAID
            and all(k.startswith(PREFIX) and v.dtype==torch.float16
                    for k,v in raw.items()),
            "exact 22311 new child weights saved IN Calibri state dict")
    model=NRA3InCalibriNeuralWeights()
    model.load_state_dict({k[len(PREFIX):]:v.detach().float()
                           for k,v in raw.items()},strict=True)
    require(model.budget()["new_neural_parameters"]==PAID
            and model.output_gate==0 and not model.candidate_accepted
            and not model.is_trained,
            "NRA3 model child starts untrained and unaccepted")
    model.eval()
    initial=teacher_anchored_role_top1(
        model,hidden,valid,aligned.targets,aligned.same_corpus_development)
    batches=fixed_stratum_batches(aligned.original_source_indices,aligned.train)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    traces=[]
    for epoch in range(EPOCHS):
        model.train()
        losses=[]
        for ix in batches:
            opt.zero_grad(set_to_none=True)
            graph=model(hidden[ix].float().detach(),valid[ix])
            loss,parts=graph_train_only_loss(graph,subset(aligned.targets,ix))
            require(torch.isfinite(loss),"finite UD source-only TRAIN loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            require(all(p.grad is None or torch.isfinite(p.grad).all()
                        for p in model.parameters()),
                    "only the NEW Calibri neural child has finite gradients")
            opt.step()
            losses.append(float(loss.detach()))
        if epoch in (0,3,9,19,EPOCHS-1):
            traces.append({"epoch":epoch,"TRAIN_only_mean_loss":
                           sum(losses)/len(losses)})
    model.eval()
    model.is_trained=True
    train_score=teacher_anchored_role_top1(
        model,hidden,valid,aligned.targets,aligned.train)
    diag=teacher_anchored_role_top1(
        model,hidden,valid,aligned.targets,aligned.same_corpus_development)
    require(sha(s95_child)==source_manifest["IN_CALIBRI_22311_NEW_FP16_WEIGHTS_SHA256"],
            "real S95 child checkpoint protected, never overwritten")
    trained={PREFIX+k:v.detach().half().cpu().contiguous()
             for k,v in model.state_dict().items()}
    require(sum(z.numel() for z in trained.values())==PAID
            and all(torch.isfinite(z).all() for z in trained.values()),
            "exactly 22311 trained FP16 IN Calibri child")
    out.mkdir(parents=True,exist_ok=False)
    save_file(trained,str(out/WEIGHTS))
    reread=load_file(str(out/WEIGHTS))
    require(reread.keys()==trained.keys()
            and all(torch.equal(reread[k],v) for k,v in trained.items()),
            "physical trained child FP16 readback")
    save_file({
        "S96_teacher_UD_event_nativeBPE_POSITIVE_only_NEG100unknown":
            aligned.targets.mention.contiguous(),
        "S96_teacher_UD_role_argument_nativeBPE_NEG100unknown":
            aligned.targets.role_argument_token.contiguous(),
        "S96_teacher_UD_event_state_POSITIVE_only_minus1_unknown":
            aligned.targets.state.contiguous(),
        "S96_NRA2_eight_UD_miniareas_POSITIVE_only_minus1_unknown":
            aligned.targets.mini_area.contiguous(),
        "S96_original_native_BPE_valid":
            aligned.targets.original_native_valid.contiguous(),
        "S96_source_original_row_indices":
            aligned.original_source_indices.int().contiguous(),
        "S96_predeclared_48of64_TRAIN_mask":aligned.train.contiguous(),
        "S96_predeclared_16of64_same_corpus_DEV_mask":
            aligned.same_corpus_development.contiguous(),
    },str(out/LABELS_NAME))
    report={
        "stage":STAGE,
        "S95_frozen_original12_native_H_SHA":
            source_manifest["ALL12_CPU_SAFETENSORS_SHA256"],
        "S95_original_untrained_in_model_child_SHA":
            source_manifest["IN_CALIBRI_22311_NEW_FP16_WEIGHTS_SHA256"],
        "S96_trained_in_model_child_SHA":sha(out/WEIGHTS),
        "S96_offline_Taiga_UD_BPE_labels_SHA":sha(out/LABELS_NAME),
        "S94_Taiga_source_SHA":audit["S94_SOURCE_SHA256"],
        "S94_separate_UD_gold_SHA":audit["S94_separate_gold_SHA256"],
        "S96_original_native_BPE_mapping_audit":audit,
        "new_trainable_NRA3_neural_parameters":PAID,
        "new_nominal_FP16_weight_bytes":PAID*2,
        "only_NEURAL_CHILD_inscribed_in_Calibri_weights_trained":True,
        "all_base_Calibri_GT_S59_S65_neural_weights_UNCHANGED":True,
        "protected_original_616_J7_replayed_before_S95_source":True,
        "eight_separate_NRA2_neural_miniareas_preserved":True,
        "four_learned_routes_over_original12_layers":True,
        "six_relation_states_and_in_weight_memory_proto_not_retention_test":True,
        "UD_label_only_after_real12_original_source_forward":True,
        "one_FIXED_48_16_source_stratum_split_before_metrics":True,
        "same_corpus_internal_dev_NOT_blind_firstread":True,
        "TRAIN_only_model_no_tuner_or_checkpoint_selection":True,
        "unlabelled_event_mentions_and_role_bits_NOT_assumed_negative":True,
        "original_S79_S86_firstread48_NOT_used":True,
        "no_616_candidate_or_bpp_CPU_online_causal_proof":True,
        "licensed_CC_BY_SA_4_0_no_commercial_production_without_review":True,
        "training_fixed":{"seed":SEED,"epochs":EPOCHS,
                          "lr":LR,"weight_decay":WD,"grad_clip":1.0,
                          "n_balanced_batches":len(batches)},
        "TRAIN_only_loss_trace":traces,
        "untrained_S95_same_Taiga_internal_development_UD_roles":initial,
        "trained_S96_same_Taiga_internal_development_UD_roles":diag,
        "trained_S96_Taiga_TRAIN_roles":train_score,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (out/MANIFEST).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("NRA3_S96_TRUE_CALIBRI12_IN_WEIGHTS_UD_TRAIN_ONLY",
          json.dumps({
              "TRAIN_edges":train_score["annotated_teacher_event_role_edges"],
              "TRAIN_accuracy":train_score["teacher_anchored_native_BPE_role_top1_accuracy"],
              "internal_same_corpus_dev_edges":diag["annotated_teacher_event_role_edges"],
              "S95_untrained_internal_dev_accuracy":initial["teacher_anchored_native_BPE_role_top1_accuracy"],
              "S96_trained_internal_dev_accuracy":diag["teacher_anchored_native_BPE_role_top1_accuracy"],
              "mapped_source_rows":audit["n_native_BPE_mapping_eligible"],
              "new_in_model_neural_FP16_params":PAID,"gate":0,
          },sort_keys=True),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser()
    for name in ("s95_source","s95_report","s95_child","s94","out"):
        ap.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    args=ap.parse_args()
    train_new_in_calibri_neural_only(
        args.s95_source,args.s95_report,args.s95_child,args.s94,args.out)


if __name__=="__main__":
    main()
