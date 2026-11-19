"""S110: actual original12 NEW Russian-GSD TRAIN-only structural NRA3 learning.

S107 introduced 16,656 neural predicate/argument rank-competitive
parameters INSIDE Calibri 53,743 child; S109 must physically capture
original compressed CB59+GT ALL12 after protected old616 J7, with GSD
offline UD GOLD PHYSICALLY ABSENT. Verify original source12 SHA FIRST,
THEN open separate S108 official GSD TRAIN-only labels. Learn ONLY NEW
16,656 neural weights using predetermined hyperparams, preserving
entire previously trained S105 37,087 neural child exactly bitwise:
8 old NRA2 mini-areas, all12 learned layer router, morphology,
neural role affinities, 16 in-weight memory slots.

Train first 48/64 per four fixed strata, score remaining 16/64
ONCE only after all fixed epochs; this is same-corpus internal dev,
NOT independent first-read or event factuality. Old internal Taiga devs
S96/S99/S102/S105 already spent, old S79/S86 firstreads spent,
official Taiga TRAIN-c still UNOPENED for future prospective test.
No main/LIVE/Render/Supabase/Jessica/APK or production. Output gate0.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import (
    NRA3RelationalMorphologyInCalibriWeights,
)
from research.calibri_nra3.s107_in_calibri_structural_role_competition_53743 import (
    NRA3StructuralCompetitionInCalibriWeights,NEW_KEYS,NEW_NEURAL_PARAMETERS,
    OLD_NEURAL_PARAMETERS,TOTAL_NEURAL_PARAMETERS,
)
from research.calibri_nra3.s100_offline_allverb_multiargument_nativeBPE_train_loss import (
    OriginalBPEAllVerbsEdges, offline_all_verb_multi_positive_relation_loss,
)
from research.calibri_nra3.s110_native_BPE_GSD_gold_AFTER_S109_sourceSHA import (
    load_source_VERIFY_sha_THEN_OPEN_gold,UNKNOWN,
)

STAGE="CALIBRI_NRA3_S110_REAL_ORIGINAL12_GSD_NEW_16656_STRUCTURAL_ONLY_TRAIN_GATE0"
PREFIX="model.nra3_neural_weights."
SEED=1101119
EPOCHS=24
LR=0.001
WEIGHT_DECAY=0.01
OLD_PARAMS=37087
TRAINED_NEW_PARAMS=16656
TOTAL_PARAMS=53743
PRIOR_FILE="NRA3_S105_TAIGA_E_STRUCTURAL_AFFINITY_37087_IN_CALIBRI_FP16.safetensors"
S109_UNTRAINED_FILE="NRA3_S109_IN_CALIBRI_UNTRAINED_53743_FP16.safetensors"
OUT_FILE="NRA3_S110_GSD_ONLY_NEW16656_STRUCTURAL_53743_IN_CALIBRI_FP16.safetensors"
REPORT="NRA3_S110_GSD_TRAIN_ROLE_STRUCTURAL_INTERNAL_DIAGNOSTIC.json"

def require(ok,why):
    if not bool(ok):raise ValueError("S110 genuine original12 GSD neural train: "+why)

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




def train(source12,source_report,s109_untrained,s105_pretrained,
          s105_report,s108_gold_folder,out):
    require(not out.exists(),"never overwrite sealed new neural checkpoint")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    # First verify immutable actual ORIGINAL CB59+GT native12 source SHA,
    # THEN and only then open S108 separate UD training GOLD.
    physical,aligned,audit=load_source_VERIFY_sha_THEN_OPEN_gold(
        source12,source_report,s108_gold_folder)
    provenance=json.loads(source_report.read_text())
    require(s109_untrained.name==S109_UNTRAINED_FILE
            and sha(s109_untrained)==
                provenance["IN_CALIBRI_53743_MORPH_FP16_WEIGHTS_SHA256"]
            and provenance["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and provenance["output_gate"]==0
            and not provenance["candidate_accepted"],
            "original 53,743 in-Calibri S109 neural untrained after 616 true source")
    require(s105_pretrained.name==PRIOR_FILE,
            "only previous S105 FP16 trained child, no dev labels")
    prevmeta=json.loads(s105_report.read_text())
    require(prevmeta["stage"]==
               "CALIBRI_NRA3_S105_TAIGA_E_REAL12_8200_AFFINITY_ONLY_INMODEL_NEURAL_TRAIN_GATE0"
            and prevmeta["S105_trained_37087_in_calibri_child_SHA"]==sha(s105_pretrained)
            and prevmeta["all_prior_28887_neural_weights_remained_FP16_exact"]
            and prevmeta["only_new_8200_neural_affinity_parameters_trained"]
            and prevmeta["output_gate"]==0 and not prevmeta["candidate_accepted"],
            "physically frozen old S105 trained 37,087 neural weights SHA only")
    old=load_file(str(s105_pretrained),device="cpu")
    untrained=load_file(str(s109_untrained),device="cpu")
    require(OLD_NEURAL_PARAMETERS==OLD_PARAMS
            and NEW_NEURAL_PARAMETERS==TRAINED_NEW_PARAMS
            and TOTAL_NEURAL_PARAMETERS==TOTAL_PARAMS
            and set(old)<set(untrained)
            and sum(v.numel() for v in old.values())==OLD_PARAMS
            and sum(v.numel() for v in untrained.values())==TOTAL_PARAMS
            and all(k.startswith(PREFIX) for k in untrained),
            "old37087 and full53743 exact neural child state_dict keys")
    added={k[len(PREFIX):] for k in untrained if k not in old}
    require(added==NEW_KEYS and all(v.dtype==torch.float16 for v in untrained.values()),
            "S107 adds exact five structural parameter tensors, no core mutation")
    require(bool(torch.equal(
        untrained[PREFIX+"structural_competition_gate"],
        torch.zeros_like(untrained[PREFIX+"structural_competition_gate"]))),
        "pretrain structural gates exactly 0, previous baseline identity")
    h=physical["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    require(h.dtype==torch.float16 and h.ndim==4
            and h.shape[1]==12 and h.shape[-1]==320
            and h.shape[0]==len(aligned.original_source_row_indices)
            and not h.requires_grad and torch.isfinite(h).all(),
            "native original12 Calibri hidden source-only physically frozen")
    n=NRA3StructuralCompetitionInCalibriWeights()
    n.load_state_dict({
      k[len(PREFIX):]:(old[k] if k in old else v).float().clone()
      for k,v in untrained.items()},strict=True)
    require(n.budget()["new_neural_parameters"]==TOTAL_PARAMS
            and n.output_gate==0 and not n.candidate_accepted and not n.activation_allowed,
            "new in-model neural 53743 child, not external classifier")
    for name,p in n.named_parameters():p.requires_grad_(name in NEW_KEYS)
    require(sum(p.numel() for p in n.parameters() if p.requires_grad)==TRAINED_NEW_PARAMS
            and sum(p.numel() for p in n.parameters() if not p.requires_grad)==OLD_PARAMS,
            "optimizer strictly only appended 16,656 structural neural weights")
    batches=batches_fixed(aligned.original_source_row_indices,aligned.train)
    opt=torch.optim.AdamW(
        [p for p in n.parameters() if p.requires_grad],
        lr=LR,weight_decay=WEIGHT_DECAY)
    trace=[]
    for epoch in range(EPOCHS):
        n.train()
        losses=[]
        for ix in batches:
            require(bool(aligned.train[ix].all()) and
                    not bool(aligned.internal_dev[ix].any()),
                    "TRAIN-e/GSD fixed first48/64 ONLY, internal developer not fitted")
            opt.zero_grad(set_to_none=True)
            g=n(h[ix].float().detach(),aligned.role.native_BPE_valid[ix])
            loss,_=offline_all_verb_multi_positive_relation_loss(
                g,select_targets(aligned.role,ix))
            require(torch.isfinite(loss),"finite real BPE original-GSD UD multi-positive role loss")
            loss.backward()
            grads=[p.grad for p in n.parameters() if p.requires_grad]
            require(all(v is not None and torch.isfinite(v).all() for v in grads),
                    "train only new 16,656 neural role competition weights")
            torch.nn.utils.clip_grad_norm_(
                [p for p in n.parameters() if p.requires_grad],1.)
            opt.step()
            losses.append(float(loss.detach()))
        if epoch in (0,5,11,17,EPOCHS-1):
            trace.append({"epoch":epoch,
                          "TRAIN_only_mean_role_loss":sum(losses)/len(losses)})
    n.eval()
    n.is_trained=True
    # Before any DEV, prior FP16 neural key values must remain exactly.
    current=n.state_dict()
    require(all(torch.equal(current[k[len(PREFIX):]].half().cpu(),v)
                for k,v in old.items()),
            "every prior trained 37,087 Calibri/NRA2+memory+GT-router neural FP16 weight EXACT")
    require(any(not torch.equal(current[k].half().cpu(),untrained[PREFIX+k])
                for k in NEW_KEYS),
            "new structural gradients must physically change at least one FP16 weight")
    require(sha(s105_pretrained)==prevmeta["S105_trained_37087_in_calibri_child_SHA"]
            and sha(s109_untrained)==provenance["IN_CALIBRI_53743_MORPH_FP16_WEIGHTS_SHA256"],
            "both reference checkpoints immutable SHA through all training")
    # No access to GSD internal-dev labels or scores occurred during
    # training or checkpoint selection; exactly one posthoc score.
    base=NRA3RelationalMorphologyInCalibriWeights().eval()
    base.load_state_dict({
      k[len(PREFIX):]:v.float().clone() for k,v in old.items()
    },strict=True)
    baseline=teacher_anchored_diagnostic(base,h,aligned,aligned.internal_dev)
    after=teacher_anchored_diagnostic(n,h,aligned,aligned.internal_dev)
    train_result=teacher_anchored_diagnostic(n,h,aligned,aligned.train)
    full={PREFIX+k:v.half().detach().cpu().contiguous()
          for k,v in current.items()}
    require(sum(v.numel() for v in full.values())==TOTAL_PARAMS
            and all(torch.isfinite(v).all() for v in full.values()),
            "only new in-Calibri 53,743 child weights saved, no original 43M")
    out.mkdir(parents=True,exist_ok=False)
    save_file(full,str(out/OUT_FILE))
    check=load_file(str(out/OUT_FILE),device="cpu")
    require(set(check)==set(full) and
            all(torch.equal(check[k],v) for k,v in full.items()),
            "S110 trained FP16 neural child exact readback")
    summary={
        "stage":STAGE,
        "S109_REAL_original12_GSD_SOURCE_SHA":provenance["ALL12_CPU_SAFETENSORS_SHA256"],
        "S109_UNTRAINED_53743_child_SHA":provenance["IN_CALIBRI_53743_MORPH_FP16_WEIGHTS_SHA256"],
        "S105_prior_37087_trained_child_SHA":prevmeta["S105_trained_37087_in_calibri_child_SHA"],
        "S110_trained_53743_neural_child_SHA":sha(out/OUT_FILE),
        "S108_GSD_source_only_SHA":"4078f408f2355c55c29bca18504355379e2a42ef2144c9113dda9448b086de14",
        "S108_offline_GSD_gold_SHA":"407ee8a5ca720ebad96379859e647623aa154e32277376f24e29be6681a1fb0b",
        "original_616_J7_replayed_before_S109_source":True,
        "original_Calibri_CB59_GT_43M_weights_frozen_unchanged":True,
        "all12_original_H320_native_BPE_no_GOLD_during_model_forward":True,
        "GSD_UD_gold_opened_ONLY_AFTER_S109_original12_SHA_verified":True,
        "only_NEW_16656_structural_predicate_arg_neural_params_trained":True,
        "all_prior_37087_FP16_NRA2_mini_areas_router_morph_memory_preserved":True,
        "fixed_48_TRAIN_16_internal_PER64_NO_earlystop_or_tuner":True,
        "old_S96_S99_S102_S105_devs_SPENT_NEVER_reused":True,
        "original_S79_S86_firstreads_SPENT_NEVER_reused":True,
        "entire_official_Taiga_TRAIN_c_prospective_RESERVED_UNOPENED":True,
        "GSD_same_corpus_INTERNAL_developer_now_SPENT_NO_retraining_tuning":True,
        "teacher_VERB_positions_provided_metric_ONLY_not_autonomous_event_detection":True,
        "no_independent_grammar_transfer_or_factuality_QA_proven":True,
        "third_party_CC_BY_SA_attribution_sharealike_legal_review_required":True,
        "all_in_bpp_le07_CPU_ge90_and_online_causal_UNCERTIFIED":True,
        "no_active_candidate616_merged_or_deployed":True,
        "train_parameters_FIXED_BEFORE_diagnostic":{
            "seed":SEED,"epochs":EPOCHS,"lr":LR,
            "weight_decay":WEIGHT_DECAY,
            "batches_per_epoch":len(batches),
            "frozen_prior_neural_params":OLD_PARAMS,
            "new_trainable_structural_params":TRAINED_NEW_PARAMS,
        },
        "original_GSD_BPE_alignment_provenance":audit,
        "TRAIN_only_loss_trace":trace,
        "S105_frozen_baseline_on_new_GSD_internal_diag":baseline,
        "S110_after_structural_train_new_GSD_internal_diag":after,
        "S110_original_GSD_TRAIN_only":train_result,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (out/REPORT).write_text(json.dumps(summary,ensure_ascii=False,
       sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("NRA3_S110_NEW_GSD_TRAIN_ONLY_STRUCTURAL_NEURAL_REAL12_SHADOW",
          json.dumps({"native_BPE_rows":audit["counts"]["full_UD_BPE_rows"],
                      "true_positive_ARCS":audit["counts"]["positive_event_argument_edges"],
                      "prior_on_new_GSD_internal":baseline["multi_verb_role_teacher_anchored"],
                      "after_on_new_GSD_internal":after["multi_verb_role_teacher_anchored"],
                      "TRAIN_only":train_result["multi_verb_role_teacher_anchored"],
                      "new_trained_params":TRAINED_NEW_PARAMS,
                      "full_neural_params":TOTAL_PARAMS,
                      "gate":0},ensure_ascii=False,sort_keys=True),flush=True)
    return summary

def main():
    a=argparse.ArgumentParser()
    for k in ("source12","source_report","s109_untrained",
              "s105_pretrained","s105_report","s108_gold_folder","out"):
        a.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    args=a.parse_args()
    train(args.source12,args.source_report,args.s109_untrained,
          args.s105_pretrained,args.s105_report,
          args.s108_gold_folder,args.out)

if __name__=="__main__":
    main()
