"""NRA3 S102: train ONLY 28,887 embedded Calibri neural parameters on new Taiga TRAIN-d.

S101 true original Calibri CB59 + GreenTunnel ALL12 source on pinned
TRAIN-d was physically computed after original protected 616 J7.
Verify its source SHA before opening ANY S100 separate UD annotations.
Initialize new in-Calibri child from physically SHA-frozen S99 trained
morphology weights (NO S99 internal-development labels or results).
Fixed 26 epochs, morphology and multi-positive all-VERB role loss,
and weak in-weight memory prototype snapshot penalty.

Only offline TRAIN-d first48/64 original source ids are optimized.
Last16/64 SAME TRAIN-d internal development solely scored at the END
(not used to choose checkpoints, hparams, architecture or thresholds).
TRAIN-c remains completely RESERVED for a prospective first-read.
Original CB59/GT/S59/S65 weights unchanged; gate0 research only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file,load_file

from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import NRA3MorphologyInCalibriWeights
from research.calibri_nra3.s99_offline_morphology_native_BPE_TRAIN_loss import offline_native_morphology_supervision_loss
from research.calibri_nra3.s100_offline_allverb_multiargument_nativeBPE_train_loss import (
    OriginalBPEAllVerbsEdges,offline_all_verb_multi_positive_relation_loss,
)
from research.calibri_nra3.s102_real12_TaigaTRAINd_multiverb_gold_AFTER_sourceSHA import (
    load_source_VERIFY_sha_THEN_OPEN_gold,UNKNOWN,
)

STAGE="CALIBRI_NRA3_S102_TAIGA_TRAIND_REAL12_ALLVERB_IN_CALIBRI_28887_TRAIN_GATE0"
SEED=1021119
EPOCHS=26
LR=0.0007
WD=0.01
MORPH_WEIGHT=0.4
MEMORY_WEIGHT=0.02
PAID=28887
PREFIX="model.nra3_neural_weights."
S99_CHILD="NRA3_S99_TAIGA_B_IN_CALIBRI_TRAINED_MORPH_RELATION_28887_FP16.safetensors"
S99_PINNED_SHA="c77a6964ae4e02cb3aac98d69978a25cec2b491a9624cc9668b3b307699bff8a"
S101_CHILD="NRA3_S101_IN_CALIBRI_UNTRAINED_28887_FP16.safetensors"
CHILD_OUT="NRA3_S102_TAIGA_D_IN_CALIBRI_ALLVERB_ROLE_28887_TRAINED_FP16.safetensors"
REPORT_OUT="NRA3_S102_TAIGA_D_ALLVERB_TRAIN_INTERNAL_DIAGNOSTICS.json"
LABELS_OUT="NRA3_S102_TAIGA_D_ORIGINAL_BPE_UD_MULTIVERB_TRAIN_ONLY_LABELS.safetensors"


def require(x,why):
    if not bool(x):
        raise ValueError("NRA3 S102 real original12 multi-VERB neural-only: "+why)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def batches_fixed(source_ids,train_mask):
    cohorts=[]
    for i in range(4):
        chosen=torch.where((source_ids>=64*i)&(source_ids<64*i+48)&train_mask)[0]
        require(len(chosen)>=16,">=16 exact BPE source rows per fixed train stratum")
        cohorts.append(chosen)
    n=min(len(x)//2 for x in cohorts)
    require(n>=8,"at least 8 fixed balanced batches each epoch")
    return [torch.cat([g[2*j:2*j+2] for g in cohorts]) for j in range(n)]


def select_targets(target:OriginalBPEAllVerbsEdges,indices):
    return OriginalBPEAllVerbsEdges(
        native_BPE_valid=target.native_BPE_valid[indices],
        argument_gold_positive=target.argument_gold_positive[indices],
        event_role_supervised=target.event_role_supervised[indices],
        event_verb_positive_only=target.event_verb_positive_only[indices],
    )


@torch.inference_mode()
def teacher_anchored_diagnostic(net,hidden,aligned,subset):
    """Only AFTER forward score annotated argument POSITIVE token; no oracle forward."""
    names=("upos","case","number","voice")
    m={k:{"correct":0,"support":0} for k in names}
    r={k:{"correct":0,"role_slots":0,"positive_edges":0} for k in ROLE_NAMES}
    rows=torch.where(subset)[0].tolist()
    for x in rows:
        g=net(hidden[x:x+1].float().detach(),aligned.role.native_BPE_valid[x:x+1])
        for k in names:
            truth=aligned.morphology[k][x]
            mask=truth!=UNKNOWN
            correct=int((getattr(g,k+"_logits")[0][mask].argmax(-1)==truth[mask]).sum())
            m[k]["correct"]+=correct
            m[k]["support"]+=int(mask.sum())
        for j,name in enumerate(ROLE_NAMES):
            slots=torch.where(aligned.role.event_role_supervised[x,j])[0]
            for p in slots.tolist():
                positive=aligned.role.argument_gold_positive[x,j,p]
                prediction=int(g.role_arc_logits[0,j,p].argmax())
                r[name]["role_slots"]+=1
                r[name]["positive_edges"]+=int(positive.sum())
                r[name]["correct"]+=int(bool(positive[prediction]))
    for d in m.values():
        d["accuracy"]=d["correct"]/d["support"] if d["support"] else None
    for d in r.values():
        d["teacher_anchored_ANY_positive_top1"]=d["correct"]/d["role_slots"] if d["role_slots"] else None
    return {
        "n_source_rows":len(rows),
        "morphology_original_native_BPE":m,
        "multi_verb_role_teacher_anchored":{k:v for k,v in r.items() if v["role_slots"]},
        "not_autonomous_event_detection_or_factuality_QA":True,
        "not_independent_prospective_firstread":True,
    }


def train(source_path,source_manifest,s101_child,older_s99_child,s100_folder,out):
    require(not out.exists(),"one new shadow checkpoint only; no overwrite")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    # S101 true source hashed and protected original 616 first, S100 GOLD only afterwards.
    physical,aligned,audit=load_source_VERIFY_sha_THEN_OPEN_gold(
        source_path,source_manifest,s100_folder)
    origin=json.loads(source_manifest.read_text())
    require(s101_child.name==S101_CHILD
            and sha(s101_child)==origin["IN_CALIBRI_28887_MORPH_FP16_WEIGHTS_SHA256"]
            and origin["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and not origin["candidate_accepted"],
            "S101 genuine original source and untrained attached 28887 original child")
    require(older_s99_child.name==S99_CHILD
            and sha(older_s99_child)==S99_PINNED_SHA,
            "S99 previously trained morphology IN-Calibri child SHA; not old developer labels")
    hidden=physical["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"].detach()
    require(hidden.ndim==4 and hidden.shape[1]==12 and hidden.shape[-1]==320
            and hidden.dtype==torch.float16 and not hidden.requires_grad
            and torch.isfinite(hidden).all()
            and hidden.shape[0]==len(aligned.original_source_row_indices),
            "genuine frozen original12 all-source BPE")
    weights=load_file(str(older_s99_child),device="cpu")
    require(sum(v.numel() for v in weights.values())==PAID
            and all(k.startswith(PREFIX) and v.dtype==torch.float16
                    for k,v in weights.items()),
            "exact old S99 child weights only, no base CB59 weights")
    model=NRA3MorphologyInCalibriWeights()
    model.load_state_dict({k[len(PREFIX):]:v.float().detach().clone()
                           for k,v in weights.items()},strict=True)
    require(model.budget()["new_neural_parameters"]==PAID
            and model.output_gate==0 and not model.candidate_accepted
            and not model.activation_allowed,
            "same new in-Calibri NRA3 28,887 neural weights gated off")
    snapshot=model.event_memory_slots.detach().clone()
    model.eval()
    initial=teacher_anchored_diagnostic(model,hidden,aligned,aligned.internal_dev)
    batches=batches_fixed(aligned.original_source_row_indices,aligned.train)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    trace=[]
    for epoch in range(EPOCHS):
        model.train()
        losses=[]
        for ix in batches:
            opt.zero_grad(set_to_none=True)
            g=model(hidden[ix].float().detach(),aligned.role.native_BPE_valid[ix])
            role,_=offline_all_verb_multi_positive_relation_loss(
                g,select_targets(aligned.role,ix))
            morph,_=offline_native_morphology_supervision_loss(
                g,*[aligned.morphology[k][ix] for k in
                    ("upos","case","number","voice")],
                aligned.role.native_BPE_valid[ix])
            mem=model.knowledge_consolidation_penalty(snapshot)
            loss=role+MORPH_WEIGHT*morph+MEMORY_WEIGHT*mem
            require(torch.isfinite(loss),"source-original UD TRAIN loss finite")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            require(all(p.grad is None or torch.isfinite(p.grad).all()
                        for p in model.parameters()),
                    "only 28,887 new in-Calibri weights get finite gradients")
            opt.step()
            losses.append(float(loss.detach()))
        if epoch in (0,4,12,EPOCHS-1):
            trace.append({"epoch":epoch,"TRAIN_only_mean_total_loss":sum(losses)/len(losses)})
    model.eval()
    model.is_trained=True
    train_result=teacher_anchored_diagnostic(model,hidden,aligned,aligned.train)
    dev_result=teacher_anchored_diagnostic(model,hidden,aligned,aligned.internal_dev)
    require(sha(older_s99_child)==S99_PINNED_SHA
            and sha(s101_child)==origin["IN_CALIBRI_28887_MORPH_FP16_WEIGHTS_SHA256"],
            "source immutable and S99 prior neural snapshot unchanged")
    paid={PREFIX+k:v.half().detach().cpu().contiguous()
          for k,v in model.state_dict().items()}
    require(sum(z.numel() for z in paid.values())==PAID
            and all(torch.isfinite(z).all() for z in paid.values()),
            "only NEW 28,887 original Calibri child FP16 weights")
    out.mkdir(parents=True,exist_ok=False)
    save_file(paid,str(out/CHILD_OUT))
    reread=load_file(str(out/CHILD_OUT))
    require(reread.keys()==paid.keys()
            and all(torch.equal(reread[k],v) for k,v in paid.items()),
            "trained neural child FP16 physical exact readback")
    save_file({
        "UD_original_BPE_UPOS_UNKNOWN_NEG100":aligned.morphology["upos"].contiguous(),
        "UD_original_BPE_Case_UNKNOWN_NEG100":aligned.morphology["case"].contiguous(),
        "UD_original_BPE_Number_UNKNOWN_NEG100":aligned.morphology["number"].contiguous(),
        "UD_original_BPE_Voice_UNKNOWN_NEG100":aligned.morphology["voice"].contiguous(),
        "UD_original_BPE_allVERB_role_multi_positive":
            aligned.role.argument_gold_positive.contiguous(),
        "UD_original_BPE_role_slots_explicit":
            aligned.role.event_role_supervised.contiguous(),
        "UD_original_BPE_VERB_event_mentions_positive_ONLY":
            aligned.role.event_verb_positive_only.contiguous(),
        "source_original_Calibri_BPE_valid":aligned.role.native_BPE_valid.contiguous(),
        "source_original_S100_row_ids":aligned.original_source_row_indices.int().contiguous(),
        "fixed_TRAIN_only_mask":aligned.train.contiguous(),
        "fixed_same_corpus_internal_diag_mask":aligned.internal_dev.contiguous(),
    },str(out/LABELS_OUT))
    report={
        "stage":STAGE,
        "S101_original12_frozen_source_SHA":origin["ALL12_CPU_SAFETENSORS_SHA256"],
        "S101_untrained_inmodel_child_SHA":origin["IN_CALIBRI_28887_MORPH_FP16_WEIGHTS_SHA256"],
        "S99_previous_trained_morph_child_SHA":S99_PINNED_SHA,
        "S102_trained_28887_inmodel_child_SHA":sha(out/CHILD_OUT),
        "S102_offline_original_BPE_allVERB_gold_SHA":sha(out/LABELS_OUT),
        "all_12_original_Calibri_GT_CB59_weights_unchanged":True,
        "original_616_J7_physically_replayed_before_S101":True,
        "new_NRA3_neural_weights_registered_IN_Calibri_state_dict":True,
        "eight_old_NRA2_neural_mini_areas_and_in_weight_event_memory_kept":True,
        "only_new_28887_neural_parameters_trained":True,
        "memory_snapshot_loss_NOT_knowledge_retention_proof":True,
        "S100_gold_OPENED_ONLY_AFTER_S101_original_CPU_SHA":True,
        "S96_S99_old_internal_devs_and_S79_S86_firstreads_NOT_used":True,
        "Taiga_TRAIN_c_reserved_and_never_read_or_used":True,
        "fixed_48_TRAIN_16_internal_per64_no_tuner_or_early_stop":True,
        "internal_TRAIN_d_evaluation_not_prospective_firstread":True,
        "teacher_VERB_event_positions_metric_only_no_event_precision":True,
        "no_external_UD_runtime_parser":True,
        "unknown_event_factuality_roles_NEVER_assumed_negative":True,
        "licensed_CC_BY_SA_4_attribution_commercial_legal_review_needed":True,
        "no_new_616_candidate_or_online_causal_generation_validation":True,
        "no_all_in_bpp_le07_CPU_ge90_CB59_certification":True,
        "train_parameters_fixed":{
            "seed":SEED,"epochs":EPOCHS,"lr":LR,"weight_decay":WD,
            "morphology_weight":MORPH_WEIGHT,"memory_snapshot_weight":MEMORY_WEIGHT,
            "batches_per_epoch":len(batches),
        },
        "S102_original_native_BPE_multiverb_alignment":audit,
        "TRAIN_only_loss_trace":trace,
        "S99_frozen_before_S102_same_Taiga_D_internal_dev":initial,
        "S102_Taiga_D_TRAIN_only":train_result,
        "S102_Taiga_D_same_corpus_internal_dev":dev_result,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (out/REPORT_OUT).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("NRA3_S102_TRUE12_ALLVERB_IN_CALIBRI_NEURAL_SHADOW",
          json.dumps({
              "strict_aligned":audit["counts"]["full_UD_BPE_rows"],
              "verbs":audit["counts"]["known_verb_predicate_BPE_positions"],
              "positive_edges":audit["counts"]["positive_event_argument_edges"],
              "before_same_Taiga_D_internal":initial["multi_verb_role_teacher_anchored"],
              "after_same_Taiga_D_internal":dev_result["multi_verb_role_teacher_anchored"],
              "new_in_Calibri_neural_weights":PAID,"gate":0,
          },ensure_ascii=False,sort_keys=True),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser()
    for key in ("source12","source_report","s101_child","s99_child","s100_folder","out"):
        ap.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    args=ap.parse_args()
    train(args.source12,args.source_report,args.s101_child,
          args.s99_child,args.s100_folder,args.out)


if __name__=="__main__":
    main()
