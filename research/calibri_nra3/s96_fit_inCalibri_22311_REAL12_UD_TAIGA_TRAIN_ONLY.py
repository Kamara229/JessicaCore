"""NRA3 S96: fit ONLY new in-Calibri neural grammar heads on REAL 12layer H8.

S95 original Calibri CB59+GreenTunnel genuine 12layer source features
were physically created AFTER the old protected616 J7, with NO Taiga
grammar GOLD. This module opens the SEPARATE S94 licensed UD TRAIN
annotation ONLY once S95 physical source SHA is frozen. Original old
Calibri + GT weights never loaded into optimizer and remain untouched.

The resulting 22,311 trained parameters have keys under the actual
Calibri model.model.nra3_neural_weights child; this is NOT a separate
production service or a model output activation. Training is on 80%
of original Taiga TRAIN-a; a fixed 20% of SAME upstream TRAIN-a gives
internal diagnostics ONCE (NOT independent language generalization).
S79/S86 48 novel firstreads permanently spent. CC BY-SA review
required before any commercial use. Gate 0, no active 616 candidate
replay, no causal-online or bpp/CPU certification.
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
    NRA3InCalibriNeuralWeights,
)
from research.calibri_nra3.s93_offline_ud_grammatical_role_train_loss_ONLY import (
    graph_train_only_loss,IGNORE,UNKNOWN,NRA3TrainOnlyTargets,
)
from research.calibri_nra3.s96_taiga_UD_to_realBPE_gold_TRAIN_ONLY import (
    build_native_training_targets,serialized_sha,S94_SOURCE_SHA,S94_GOLD_SHA,
    S95_PHYSICAL_SHA,
)

STAGE="CALIBRI_NRA3_S96_REAL12_UD_TAIGA_22311_NEURAL_GRAMMAR_TRAIN_SHADOW"
S95_TENSOR="NRA3_S95_REAL43M_GT_TAIGA_TRAIN_NATIVE_ALL12_H320_NO_GOLD.safetensors"
S95_REPORT="NRA3_S95_REAL43M_GT_TAIGA_SOURCE_ONLY_ALIGNMENT_PROVENANCE.json"
S95_INITIAL_CHILD="NRA3_S95_IN_CALIBRI_UNTRAINED_22311_NEURAL_WEIGHTS_FP16.safetensors"
TRAINED_FP16="NRA3_S96_REAL12_TAIGA_UD_IN_CALIBRI_22311_TRAINED_NEURAL_FP16.safetensors"
REPORT="NRA3_S96_REAL12_UD_TAIGA_GRAMMAR_TRAIN_AND_INTERNAL_DIAGNOSTIC.json"
SEED=961119
EPOCHS=12
LR=0.002
WD=0.02
BATCH=6
N_PARAMS=22311


def require(ok,why):
    if not bool(ok):
        raise ValueError("NRA3 S96 real original12 in-Calibri grammar training: "+why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_targets(src:NRA3TrainOnlyTargets,indexes:torch.Tensor,length:int):
    """Only previously pinned original BPE training labels/indices."""
    return NRA3TrainOnlyTargets(
        mention=src.mention[indexes,:length],
        role_argument_token=src.role_argument_token[indexes,:,:length],
        state=src.state[indexes,:length],
        mini_area=src.mini_area[indexes,:length],
        original_native_valid=src.original_native_valid[indexes,:length],
        evidence=src.evidence,
    )


def neural_forward_only_saved_true12(
        net:NRA3InCalibriNeuralWeights,source_hidden,source_mask,indexes):
    length=int(source_mask[indexes].sum(-1).max().item())
    h=source_hidden[indexes,:,:length,:].float()
    mask=source_mask[indexes,:length]
    require(not h.requires_grad and h.shape[1]==12 and h.shape[-1]==320
            and h.dtype==torch.float32,
            "S95 frozen actual original CB59+GT 12layer BPE values never get gradients")
    return net(h,mask),length


@torch.inference_mode()
def internal_teacher_anchored_metrics(net,source_hidden,valid,targets,indices):
    """One fixed pass, not event-discovery or independent FIRSTREAD F1."""
    net.eval()
    tp_arc=0;total_arc=0
    tp_state=0;total_state=0
    tp_area=0;total_area=0
    event_hit=0;event_total=0
    batches=torch.split(indices,BATCH)
    for group in batches:
        g,t=neural_forward_only_saved_true12(net,source_hidden,valid,group)
        gold=select_targets(targets,group,t)
        ev=gold.mention==1
        event_hit+=int((g.event_mention_logits[ev]>0).sum())
        event_total+=int(ev.sum())
        arcs=gold.role_argument_token
        rm=arcs!=IGNORE
        tp_arc+=int((g.role_arc_logits.argmax(-1)[rm]==arcs[rm]).sum())
        total_arc+=int(rm.sum())
        sm=gold.state!=UNKNOWN
        tp_state+=int(((g.event_state_logits[sm]>=0)==gold.state[sm].bool()).sum())
        total_state+=int(sm.sum())
        am=gold.mini_area!=UNKNOWN
        tp_area+=int(((g.mini_area_logits[am]>=0)==gold.mini_area[am].bool()).sum())
        total_area+=int(am.sum())
    return {
        "n_UD_source_rows":len(indices),
        "n_UD_event_positive_teacher_anchor":event_total,
        "n_teacher_anchored_event_positive_logit":event_hit,
        "n_UD_explicit_event_argument_arc_gold":total_arc,
        "n_UD_arc_argmax_correct_GIVEN_gold_event":tp_arc,
        "n_UD_explicit_passive_negation_state_bits":total_state,
        "n_correct_UD_passive_negation_state_bits":tp_state,
        "n_UD_explicit_two_NRA2_miniheads_bits":total_area,
        "n_correct_explicit_NRA2_minihead_bits":tp_area,
        "arc_accuracy_given_UD_teacher_event":
            tp_arc/total_arc if total_arc else 0.0,
        "passive_negation_state_accuracy":
            tp_state/total_state if total_state else 0.0,
        "NRA2_two_supervised_mini_areas_accuracy":
            tp_area/total_area if total_area else 0.0,
        "all_other_six_NRA2_mini_areas_UNTRAINED":True,
        "not_automatic_unanchored_event_detection_or_factuality":True,
        "not_independent_firstread_generalization_or_knowledge_retention":True,
    }


def main():
    p=argparse.ArgumentParser()
    for k in ("s95","s95_report","s95_child","s94_source","s94_gold","s94_manifest","out"):
        p.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    args=p.parse_args()
    require(not args.out.exists()
            and args.s95.name==S95_TENSOR
            and args.s95_report.name==S95_REPORT
            and args.s95_child.name==S95_INITIAL_CHILD
            and sha(args.s95)==S95_PHYSICAL_SHA,
            "exact first saved old-original12 source physical SHA, no overwrite")
    provenance=json.loads(args.s95_report.read_text(encoding="utf8"))
    source=json.loads(args.s94_source.read_text(encoding="utf8"))
    gold=json.loads(args.s94_gold.read_text(encoding="utf8"))
    prem=json.loads(args.s94_manifest.read_text(encoding="utf8"))
    require(provenance["stage"]==
                "CALIBRI_NRA3_S95_REAL43M_GT_ALL12_UD_TAIGA_TRAIN_SOURCE_ONLY_AFTER616"
            and provenance["ALL12_CPU_SAFETENSORS_SHA256"]==S95_PHYSICAL_SHA
            and provenance["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and provenance["source_only_no_Taiga_UD_gold_loaded_during_original_CPU"]
            and provenance["NRA3_neural_22311_weights_REGISTERED_INSIDE_actual_Calibri_model"]
            and provenance["output_gate"]==0
            and not provenance["candidate_accepted"]
            and prem["source_only_sha256"]==S94_SOURCE_SHA==serialized_sha(source)
            and prem["separately_supervised_only_gold_sha256"]==
                S94_GOLD_SHA==serialized_sha(gold)
            and prem["license"]=="CC BY-SA 4.0"
            and prem["no_production_or_commercial_use_until_license_review"]
            and prem["only_upstream_TRAIN_a_used_NO_upstream_dev_or_test"],
            "S95 genuine 616 baseline/source+locked S94 licensed UD provenance")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    d=load_file(str(args.s95),device="cpu")
    targets,train,diag,label_report=build_native_training_targets(
        source,gold,d)
    native=d["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    valid=d["native_BPE_source_valid_mask"]
    initial=load_file(str(args.s95_child),device="cpu")
    prefix="model.nra3_neural_weights."
    require(len(initial)>10
            and all(k.startswith(prefix) for k in initial)
            and sum(z.numel() for z in initial.values())==N_PARAMS
            and provenance["IN_CALIBRI_22311_NEW_FP16_WEIGHTS_SHA256"]==
                sha(args.s95_child),
            "REAL ORIGINAL Calibri model child state_dict newly registered weights")
    torch.manual_seed(SEED)
    net=NRA3InCalibriNeuralWeights()
    net.load_state_dict(
        {k[len(prefix):]:v.float().clone() for k,v in initial.items()},
        strict=True)
    require(net.budget()["new_neural_parameters"]==N_PARAMS
            and not net.is_trained and net.output_gate==0
            and not net.activation_allowed,
            "only new embedded 22,311 trainable params, never base original CB59")
    train_ids=torch.where(train)[0]
    diag_ids=torch.where(diag)[0]
    with torch.inference_mode():
        base=internal_teacher_anchored_metrics(
            net,native,valid,targets,diag_ids)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WD)
    losses=[]
    for epoch in range(EPOCHS):
        net.train()
        # Deterministic full traversal; NO S94 diagnostic rows in optimizer.
        order=train_ids[torch.randperm(len(train_ids),
                                      generator=torch.Generator().manual_seed(SEED+epoch))]
        scalar=0.0;updates=0
        for batch in torch.split(order,BATCH):
            graph,length=neural_forward_only_saved_true12(net,native,valid,batch)
            loss,info=graph_train_only_loss(
                graph,select_targets(targets,batch,length))
            require(bool(torch.isfinite(loss)) and not loss.isnan().any(),
                    "NRA3 genuine S94 UD word→BPE graph training loss finite")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all()
                        for p in net.parameters()),
                    "NRA3 original12 neural TRAIN gradients finite and old frozen core absent")
            nn.utils.clip_grad_norm_(net.parameters(),3.0)
            opt.step()
            scalar+=float(loss.detach());updates+=1
        losses.append({"epoch":epoch+1,"TRAIN_only_mean_graph_loss":scalar/updates})
    net.eval()
    with torch.inference_mode():
        final_train=internal_teacher_anchored_metrics(
            net,native,valid,targets,train_ids)
        final_diag=internal_teacher_anchored_metrics(
            net,native,valid,targets,diag_ids)
    require(sha(args.s95)==S95_PHYSICAL_SHA
            and sha(args.s95_child)==provenance["IN_CALIBRI_22311_NEW_FP16_WEIGHTS_SHA256"],
            "original all12 frozen Calibri source & S95 child checkpoint unchanged after training")
    exported={
        prefix+k:v.detach().half().cpu().contiguous()
        for k,v in net.state_dict().items()
    }
    require(sum(t.numel() for t in exported.values())==N_PARAMS
            and all(torch.isfinite(t).all() for t in exported.values())
            and net.output_gate==0 and not net.candidate_accepted,
            "new in-Calibri neural trained FP16 child stable gate0")
    args.out.mkdir(parents=True,exist_ok=False)
    destination=args.out/TRAINED_FP16
    save_file(exported,str(destination))
    reread=load_file(str(destination))
    require(set(exported)==set(reread)
            and all(torch.equal(exported[k],reread[k]) for k in reread),
            "actual trained new 22,311 FP16 child weights serialized")
    report={
        "stage":STAGE,
        "S95_true_original12_native_BPE_H320_SHA256":S95_PHYSICAL_SHA,
        "S95_untrained_real_in_Calibri_child_SHA256":sha(args.s95_child),
        "S96_Trained_new_in_Calibri_NRA3_22311_FP16_SHA256":sha(destination),
        "n_new_NRA3_trainable_parameters":N_PARAMS,
        "n_new_nominal_FP16_bytes":N_PARAMS*2,
        "S94_word_to_S95_real_original_native_BPE_UD_TRAIN_labels":label_report,
        "fixed_seed":SEED,"fixed_epochs":EPOCHS,
        "fixed_AdamW_lr":LR,"fixed_AdamW_wd":WD,"batch_size":BATCH,
        "loss_trace_TRAIN_only_not_model_selection":losses,
        "before_training_INTERNAL_same_Taiga_TRAIN_a_20percent":base,
        "after_training_INTERNAL_same_Taiga_TRAIN_a_80percent":final_train,
        "after_training_INTERNAL_same_Taiga_TRAIN_a_20percent":final_diag,
        "eight_separate_NRA2_minihead_architecture_preserved":True,
        "ONLY_negation_and_passive_of_eight_miniareas_supervised":True,
        "new_role_arc_supervision_and_four_ALL12_layer_streams_training":True,
        "original_Calibri_GT_CB59_weights_not_retrained":True,
        "only_learned_NRA3_child_neural_weights_saved_under_Calibri_state_dict_keys":True,
        "old_616_J7_original_preserved_but_NEW_candidate_not_run":True,
        "not_genuine_independent_language_GENERALIZATION":True,
        "true_longterm_event_memory_retention_NOT_certified":True,
        "unanchored_event_false_positive_rate_NOT_measured":True,
        "future_attending_relation_graph_NOT_online_causally_certified":True,
        "all_in_bpp_le07_and_CPU_ge90_NOT_certified":True,
        "S79_S86_SPENT_firstread_datasets_NOT_used":True,
        "Taiga_CC_BY_SA_licensing_blocks_production_pending_legal_review":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (args.out/REPORT).write_text(json.dumps(
        report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("NRA3_S96_REAL_CALIBRI_ALL12_NEW_TRAINED_22311_NEURAL",
          json.dumps({
              "train_rows":len(train_ids),"internal_diag_rows":len(diag_ids),
              "baseline_internal_role_arc_accuracy":base["arc_accuracy_given_UD_teacher_event"],
              "after_training_internal_role_arc_accuracy":final_diag["arc_accuracy_given_UD_teacher_event"],
              "after_training_internal_passive_negation_accuracy":final_diag["passive_negation_state_accuracy"],
              "new_params":N_PARAMS,"output_gate":0,
          },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
