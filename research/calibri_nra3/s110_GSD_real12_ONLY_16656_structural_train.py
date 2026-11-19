"""NRA3 S110 real original Calibri+GT all12 neural structural-role TRAIN shadow.

Only after S109 original 616 J7 physically protected, NEW GSD TRAIN256
original12 source sha frozen and UD GOLD physically absent in CPU forward.
Offline loader maps 433 original UD verbs, 690 annotated argument links
to exact native BPE only AFTER source SHA, fail closed. Old S105 37,087
in-Calibri learned neural parameters copied and kept FP16 BIT-IDENTICAL.
ONLY S107 16,656 NEW structural competition Q/K/bilinear/distance/gates
are trained for FIXED 20 epochs on first48/64 new GSD source rows.
Last16/64 same-GSD diagnostic is scored ONCE AFTER fixed epochs, with
predeclared S105 frozen comparator, no tuning on it afterwards.
Old S96/S99/S102/S105 internal devs, original S79/S86 firstreads and
prospective original Taiga TRAIN-c are never used or even loaded.
No production, no new candidate616/causal/CB59 bpp+CPU certification.
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
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import (
    NRA3RelationalMorphologyInCalibriWeights,
)
from research.calibri_nra3.s107_in_calibri_structural_role_competition_53743 import (
    NRA3StructuralCompetitionInCalibriWeights,NEW_KEYS,
    TOTAL_NEURAL_PARAMETERS,OLD_NEURAL_PARAMETERS,NEW_NEURAL_PARAMETERS,
)
from research.calibri_nra3.s100_offline_allverb_multiargument_nativeBPE_train_loss import (
    OriginalBPEAllVerbsEdges,offline_all_verb_multi_positive_relation_loss,
)
from research.calibri_nra3.s110_original12_GSD_gold_after_S109_SHA import (
    load_source_VERIFY_sha_THEN_OPEN_gold,UNKNOWN,
)

STAGE="CALIBRI_NRA3_S110_ORIGINAL12_NEW_GSD_53743_STRUCTURAL_16656_ONLY_FIXED_TRAIN_GATE0"
SEED=1101119
EPOCHS=20
LR=0.0012
WD=0.01
PREFIX="model.nra3_neural_weights."
S105_CHILD="NRA3_S105_TAIGA_E_STRUCTURAL_AFFINITY_37087_IN_CALIBRI_FP16.safetensors"
S109_CHILD="NRA3_S109_IN_CALIBRI_UNTRAINED_53743_FP16.safetensors"
TRAINED_OUT="NRA3_S110_GSD_NEW_TRAIN_STRUCTURAL_53743_IN_CALIBRI_FP16.safetensors"
REPORT_OUT="NRA3_S110_GSD_STRUCTURAL_FIXED_TRAIN_INTERNAL_REPORT.json"
LABELS_OUT="NRA3_S110_GSD_ORIGINAL_BPE_UD_TRAIN_DIAGNOSTIC_LABELS.safetensors"

def require(c,why):
    if not bool(c):
        raise ValueError("S110 actual Calibri+GT original12 neural structural train: "+why)

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




def train(source12,source_report,s109_child,older_s105_child,s108_gold_folder,out):
    require(not out.exists(),"never overwrite real original source or shadow checkpoint")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    physical,aligned,audit=load_source_VERIFY_sha_THEN_OPEN_gold(
        source12,source_report,s108_gold_folder)
    origin=json.loads(source_report.read_text(encoding="utf8"))
    require(s109_child.name==S109_CHILD
            and sha(s109_child)==origin["IN_CALIBRI_53743_STRUCTURAL_FP16_WEIGHTS_SHA256"]
            and origin["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and origin["output_gate"]==0 and not origin["candidate_accepted"],
            "NEW original GSD true12 physical source, child SHA, old616")
    require(older_s105_child.name==S105_CHILD,
            "prior S105 NEURAL WEIGHTS only, never spent old labels/diagnostics")
    hidden=physical["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"].detach()
    require(hidden.ndim==4 and hidden.shape[1]==12 and hidden.shape[-1]==320
            and hidden.dtype==torch.float16 and not hidden.requires_grad
            and torch.isfinite(hidden).all()
            and hidden.shape[0]==len(aligned.original_source_row_indices),
            "all genuine original CB59 + GT 12-layer source readonly")
    old=load_file(str(older_s105_child),device="cpu")
    initial=load_file(str(s109_child),device="cpu")
    require(len(old)<len(initial) and set(old)<set(initial)
            and all(k.startswith(PREFIX) and v.dtype==torch.float16 for k,v in old.items())
            and all(k.startswith(PREFIX) and v.dtype==torch.float16 for k,v in initial.items())
            and sum(v.numel() for v in old.values())==OLD_NEURAL_PARAMETERS
            and sum(v.numel() for v in initial.values())==TOTAL_NEURAL_PARAMETERS,
            "same in-Calibri child exact OLD37087 and new53743 FP16 architecture")
    new_keys={k[len(PREFIX):] for k in initial if k not in old}
    require(new_keys==NEW_KEYS
            and sum(initial[PREFIX+k].numel() for k in new_keys)==NEW_NEURAL_PARAMETERS
            and torch.count_nonzero(initial[PREFIX+"structural_competition_gate"])==0,
            "ONLY 16,656 new structural neural weights; 8 original gates zero")
    model=NRA3StructuralCompetitionInCalibriWeights()
    model.load_state_dict({k[len(PREFIX):]:
                           (old[k] if k in old else v).float().detach().clone()
                           for k,v in initial.items()},strict=True)
    require(model.budget()["new_neural_parameters"]==TOTAL_NEURAL_PARAMETERS
            and model.output_gate==0 and not model.candidate_accepted
            and not model.activation_allowed and not model.is_trained,
            "registered true in-Calibri structural 53743 child GATE0")
    for name,p in model.named_parameters():
        p.requires_grad_(name in NEW_KEYS)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(trainable==NEW_NEURAL_PARAMETERS,
            "only 16656 structural QK bilinear bias and competition gates trainable")
    # New GSD internal development is physically present in offline GOLD
    # but NEVER indexed/read by any optimizer or before end of fixed epochs.
    batches=batches_fixed(aligned.original_source_row_indices,aligned.train)
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,weight_decay=WD)
    trace=[]
    for epoch in range(EPOCHS):
        model.train()
        losses=[]
        for ix in batches:
            require(bool(aligned.train[ix].all())
                    and not bool(aligned.internal_dev[ix].any()),
                    "old/new internal development may NEVER enter optimizer")
            optimizer.zero_grad(set_to_none=True)
            graph=model(hidden[ix].float().detach(),aligned.role.native_BPE_valid[ix])
            loss,_=offline_all_verb_multi_positive_relation_loss(
                graph,select_targets(aligned.role,ix))
            require(torch.isfinite(loss),"strict all-verb positive-only TRAIN loss finite")
            loss.backward()
            gradients=[p.grad for p in model.parameters() if p.requires_grad]
            require(all(p is not None and bool(torch.isfinite(p).all()) for p in gradients),
                    "all new 16656 structural role Q/K and gates must get finite gradient")
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in (0,4,9,14,EPOCHS-1):
            trace.append({"epoch":epoch,
                          "TRAIN_only_mean_multi_verb_positive_role_loss":
                              sum(losses)/len(losses)})
    model.eval()
    model.is_trained=True
    weights=model.state_dict()
    require(all(torch.equal(weights[k[len(PREFIX):]].half().cpu(),v)
                for k,v in old.items()),
            "all previous 37087 morphology/mini-area/relational/memory FP16 bit-identical")
    require(any(not torch.equal(weights[k].half().cpu(),initial[PREFIX+k])
                for k in NEW_KEYS),
            "actual fixed train must modify at least one of 16656 neural weights")
    require(sha(s109_child)==origin["IN_CALIBRI_53743_STRUCTURAL_FP16_WEIGHTS_SHA256"],
            "parent source/weights immutable after training")
    # Diagnostics happen exactly ONCE after 20 preset epochs; never
    # optimize/select/retrain on the newly exposed internal rows.
    prev=NRA3RelationalMorphologyInCalibriWeights().eval()
    prev.load_state_dict({k[len(PREFIX):]:v.float()
                          for k,v in old.items()},strict=True)
    before=teacher_anchored_diagnostic(prev,hidden,aligned,aligned.internal_dev)
    after=teacher_anchored_diagnostic(model,hidden,aligned,aligned.internal_dev)
    train_result=teacher_anchored_diagnostic(model,hidden,aligned,aligned.train)
    new={PREFIX+k:v.half().detach().cpu().contiguous()
         for k,v in model.state_dict().items()}
    require(sum(z.numel() for z in new.values())==TOTAL_NEURAL_PARAMETERS
            and all(torch.isfinite(z).all() for z in new.values()),
            "only complete 53743 in-Calibri child FP16 saved")
    out.mkdir(parents=True,exist_ok=False)
    save_file(new,str(out/TRAINED_OUT))
    reread=load_file(str(out/TRAINED_OUT),device="cpu")
    require(set(reread)==set(new)
            and all(torch.equal(reread[k],v) for k,v in new.items()),
            "trained neural child FP16 exact bit readback")
    save_file({
        "new_GSD_original_BPE_UPOS_UNKNOWN_NEG100":aligned.morphology["upos"].contiguous(),
        "new_GSD_original_BPE_Case_UNKNOWN_NEG100":aligned.morphology["case"].contiguous(),
        "new_GSD_original_BPE_Number_UNKNOWN_NEG100":aligned.morphology["number"].contiguous(),
        "new_GSD_original_BPE_Voice_UNKNOWN_NEG100":aligned.morphology["voice"].contiguous(),
        "new_GSD_original_BPE_UD_allVERB_arg_positive_ONLY":aligned.role.argument_gold_positive.contiguous(),
        "new_GSD_original_BPE_role_slots_explicits":aligned.role.event_role_supervised.contiguous(),
        "new_GSD_original_BPE_VERB_mentions_positive_only":aligned.role.event_verb_positive_only.contiguous(),
        "original_Calibri_GSD_BPE_valid":aligned.role.native_BPE_valid.contiguous(),
        "new_GSD_orig_source_row_ids":aligned.original_source_row_indices.int().contiguous(),
        "fixed_GSD_TRAIN_only_mask":aligned.train.contiguous(),
        "fixed_GSD_internal_dev_mask_TO_SPEND_ONCE":aligned.internal_dev.contiguous(),
    },str(out/LABELS_OUT))
    manifest={
        "stage":STAGE,
        "S109_ORIGINAL12_new_GSD_source_SHA":origin["ALL12_CPU_SAFETENSORS_SHA256"],
        "S109_untrained_53743_child_SHA":origin["IN_CALIBRI_53743_STRUCTURAL_FP16_WEIGHTS_SHA256"],
        "S105_frozen_pretrained_37087_neural_child_SHA":sha(older_s105_child),
        "S110_new_53743_neural_child_SHA":sha(out/TRAINED_OUT),
        "S110_new_GSD_offline_labels_SHA":sha(out/LABELS_OUT),
        "S108_GSD_original_source_SHA":"4078f408f2355c55c29bca18504355379e2a42ef2144c9113dda9448b086de14",
        "S108_GSD_offline_GOLD_SHA":"e3d33d2f21986e960c22ee23503a473e1a34119ce5d3e4015ccb33eb5b869121",
        "old_Calibri_CB59_GT_and_S59_S65_weights_untouched":True,
        "old_616_J7_physically_replayed_before_UD_gold":True,
        "new_53743_neural_child_inside_actual_Calibri_architecture":True,
        "all12_original_GSD_calibri_and_GT_source_frozen":True,
        "source_SHA_verified_BEFORE_separate_GSD_gold_read":True,
        "only_NEW_16656_structural_competition_neural_weights_trained":True,
        "all_prior_37087_FP16_neural_weights_exact":True,
        "all_original_eight_NRA2_neural_mini_areas_unchanged":True,
        "all_12_learned_router_and_16_memory_slots_retained":True,
        "S96_S99_S102_S105_old_internal_devs_spent_NOT_read":True,
        "S79_S86_older_firstreads_spent_not_reused":True,
        "Taiga_TRAIN_c_new_prospective_ONE_SHOT_reserved_unopened":True,
        "GSD_new_fixed_48TRAIN_16_internal_per64_no_tuner":True,
        "GSD_internal_developer_NOW_SPENT_never_reuse_for_tuning":True,
        "teacher_GOLD_predicate_anchor_only_for_OFFLINE_eval":True,
        "no_factuality_or_autonomous_event_extraction_quality_proven":True,
        "third_party_CC_BY_SA_4_legal_and_attribution_before_any_prod":True,
        "no_all_in_bpp_le07_CPU_ge90_or_causal_generation_certification":True,
        "no_active_new_integrated616_J7":True,
        "locked_training":{"seed":SEED,"epochs":EPOCHS,"lr":LR,
            "weight_decay":WD,"batches_per_epoch":len(batches),
            "prior_neural_params_exact":OLD_NEURAL_PARAMETERS,
            "new_structural_neural_params_trained":NEW_NEURAL_PARAMETERS},
        "strict_native_BPE_alignment":audit,
        "TRAIN_only_loss_trace":trace,
        "frozen_S105_previous_on_NEW_GSD_internal_dev_first_read_at_END":before,
        "S110_trained_structure_NEW_GSD_same_corpus_dev_FIRST_READ":after,
        "S110_new_GSD_TRAIN_only":train_result,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (out/REPORT_OUT).write_text(json.dumps(
        manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("NRA3_S110_REAL_ORIGINAL12_NEW_GSD_STRUCTURE_TRAIN_FINISHED_GATE0",
          json.dumps({"train_rows":audit["counts"]["TRAIN_only_source"],
                      "dev_rows_FIRST_READ_AT_END":audit["counts"]["same_corpus_INTERNAL_dev"],
                      "positive_edges":audit["counts"]["positive_event_argument_edges"],
                      "frozen_S105_before":before["multi_verb_role_teacher_anchored"],
                      "new_S110_after":after["multi_verb_role_teacher_anchored"],
                      "train":train_result["multi_verb_role_teacher_anchored"],
                      "old_neural_untouched":OLD_NEURAL_PARAMETERS,
                      "new_neural_trained":NEW_NEURAL_PARAMETERS,"gate":0
          },sort_keys=True,ensure_ascii=False),flush=True)
    return manifest

def main():
    p=argparse.ArgumentParser()
    for key in ("source12","source_report","s109_child","s105_child","s108_gold_folder","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    v=p.parse_args()
    train(v.source12,v.source_report,v.s109_child,v.s105_child,v.s108_gold_folder,v.out)

if __name__=="__main__":
    main()
