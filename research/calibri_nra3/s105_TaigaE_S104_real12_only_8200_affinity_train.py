"""NRA3 S105: strictly TRAIN-only neural morphosyntactic role affinity (8,200 weights).

After S104 genuine original Calibri CB59+GreenTunnel ALL12 source-only,
original 616 J7 physically protected, PIN S104 native BPE/source hash
BEFORE opening separately precommitted S103 Taiga TRAIN-e official UD GOLD.

S105 loads frozen S102 trained 28,887-parameter Calibri child (SHA-fixed)
into the genuine S104 37,087-parameter in-Calibri child, preserving all old
NRA2 8 mini-areas, morphology, 12-layer routing and in-weight memory.
ONLY newly added 8,200 morphology-conditioned pairwise affinity + 8 gates
receive gradients, never old CB59, GT or old 28,887 Calibri child params.
S105 tests the architectural *inductive bias*, not unconstrained retraining.

TRAIN-e 0..47 of every source stratum for 24 fixed epochs; dev 48..63
used ONCE after training for teacher-VERB anchored diagnostic, NEVER
for model selection. No independent firstread/causal/live accuracy claims.
Old S96/S99/S102 internal developer and S79/S86 firstreads are SPENT.
Entire Taiga TRAIN-c remains unopened for one-shot prospective assessment.
Research-only output gate 0, candidate rejected, no merge/deploy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import (
    NRA3RelationalMorphologyInCalibriWeights,
)
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyInCalibriWeights,
)
from research.calibri_nra3.s100_offline_allverb_multiargument_nativeBPE_train_loss import (
    OriginalBPEAllVerbsEdges,offline_all_verb_multi_positive_relation_loss,
)
from research.calibri_nra3.s105_original12_TaigaE_UD_gold_after_S104_SHA import (
    load_source_VERIFY_sha_THEN_OPEN_gold, UNKNOWN,
)

STAGE="CALIBRI_NRA3_S105_TAIGA_E_REAL12_8200_AFFINITY_ONLY_INMODEL_NEURAL_TRAIN_GATE0"
SEED=1051119
EPOCHS=24
LR=0.002
WD=0.01
OLD_PARAMS=28887
EXTRA_PARAMS=8200
TOTAL_PARAMS=37087
PREFIX="model.nra3_neural_weights."
NEW_ONLY=frozenset({"role_morph_affinity","role_morph_affinity_gate"})
S102_FILE="NRA3_S102_TAIGA_D_IN_CALIBRI_ALLVERB_ROLE_28887_TRAINED_FP16.safetensors"
S102_SHA="f40dddbc03282a6c45c4f807746275d76a4ba776aa613247e73fbdf7f43329df"
S104_FILE="NRA3_S104_IN_CALIBRI_UNTRAINED_37087_FP16.safetensors"
OUT_FILE="NRA3_S105_TAIGA_E_STRUCTURAL_AFFINITY_37087_IN_CALIBRI_FP16.safetensors"
REPORT="NRA3_S105_TAIGA_E_STRUCTURAL_TRAIN_SHADOW_DIAGNOSTICS.json"

def require(x,why):
    if not bool(x):
        raise ValueError("S105 strict original12 Taiga-E neural research: "+why)

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



def train(source12,source_report,s104_child,s102_child,s103_gold_folder,out):
    require(not out.exists(),"shadow model output must not overwrite any checkpoint")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)

    # Mandatory real old-616 protected original source SHA before GOLD opens.
    source,labels,audit=load_source_VERIFY_sha_THEN_OPEN_gold(
        source12,source_report,s103_gold_folder)
    origin=json.loads(source_report.read_text(encoding="utf8"))
    require(s104_child.name==S104_FILE
            and sha(s104_child)==origin["IN_CALIBRI_37087_MORPH_FP16_WEIGHTS_SHA256"]
            and origin["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and not origin["candidate_accepted"]
            and origin["output_gate"]==0,
            "untrained 37087 S104 child physically pinned after original616")
    require(s102_child.name==S102_FILE and sha(s102_child)==S102_SHA,
            "only previously fixed TRAIN-d S102 neural checkpoint; NO internal diagnostic inputs")

    hidden=source["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    require(hidden.ndim==4 and hidden.shape[1]==12
            and hidden.shape[-1]==320 and hidden.dtype==torch.float16
            and torch.isfinite(hidden).all() and not hidden.requires_grad
            and hidden.shape[0]==len(labels.original_source_row_indices),
            "original 12-layer Calibri GT raw source frozen and intact")
    old=load_file(str(s102_child),device="cpu")
    new=load_file(str(s104_child),device="cpu")
    require(len(old)<len(new) and set(old)<set(new)
            and all(k.startswith(PREFIX) for k in old)
            and all(k.startswith(PREFIX) for k in new)
            and sum(v.numel() for v in old.values())==OLD_PARAMS
            and sum(v.numel() for v in new.values())==TOTAL_PARAMS
            and all(z.dtype==torch.float16 and bool(torch.isfinite(z).all())
                    for z in [*old.values(),*new.values()]),
            "actual in-Calibri 28887→37087 full neural state_dict compatibility")
    old_keys={k[len(PREFIX):] for k in old}
    new_keys={k[len(PREFIX):] for k in new if k not in old}
    require(new_keys==NEW_ONLY
            and sum(new[PREFIX+k].numel() for k in new_keys)==EXTRA_PARAMS,
            "ONLY 8200 new gradient-bearing structural weights/gates")
    require(all(torch.count_nonzero(new[PREFIX+"role_morph_affinity_gate"])==0
                for _ in (0,)),
            "S104 structural gates must start at exact zero")

    net=NRA3RelationalMorphologyInCalibriWeights()
    net.load_state_dict({
        k[len(PREFIX):]:(old[k] if k in old else v).float().clone()
        for k,v in new.items()},strict=True)
    require(net.budget()["new_neural_parameters"]==TOTAL_PARAMS
            and net.output_gate==0 and not net.candidate_accepted
            and not net.activation_allowed and not net.is_trained,
            "new 37087 neural child belongs to Calibri research architecture gate0")
    for name,p in net.named_parameters():
        p.requires_grad_(name in NEW_ONLY)
    paid=sum(p.numel() for p in net.parameters() if p.requires_grad)
    require(paid==EXTRA_PARAMS,"only 8192 affinity entries and 8 role gates trainable")

    # No same-corpus INTERNAL_DEV is evaluated or read by training iterations.
    batches=batches_fixed(labels.original_source_row_indices,labels.train)
    opt=torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],lr=LR,weight_decay=WD)
    trace=[]
    net.train()
    for ep in range(EPOCHS):
        losses=[]
        for ix in batches:
            require(bool(labels.train[ix].all()) and not bool(labels.internal_dev[ix].any()),
                    "S105 optimizer absolutely excludes new TRAIN-e internal developer")
            opt.zero_grad(set_to_none=True)
            g=net(hidden[ix].float().detach(),labels.role.native_BPE_valid[ix])
            loss,_=offline_all_verb_multi_positive_relation_loss(
                g,select_targets(labels.role,ix))
            require(torch.isfinite(loss),"finite multi-positive all-VERB TRAIN-only loss")
            loss.backward()
            gradients=[p.grad for p in net.parameters() if p.requires_grad]
            require(all(z is not None and torch.isfinite(z).all() for z in gradients),
                    "role affinity+gates must receive gradients, no frozen head gradients")
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad],1.0)
            opt.step()
            losses.append(float(loss.detach()))
        if ep in (0,5,11,17,EPOCHS-1):
            trace.append({"epoch":ep,
                          "TRAIN_ONLY_multi_positive_role_loss":sum(losses)/len(losses)})

    net.eval()
    net.is_trained=True
    current=net.state_dict()
    require(all(torch.equal(current[k].half().cpu(),v)
                for k,v in ((k[len(PREFIX):],v) for k,v in old.items())),
            "previous 28887 trained morphology+miniarea+memory weights unchanged at FP16 bit-level")
    require(any(not torch.equal(current[k].half().cpu(),new[PREFIX+k])
                for k in NEW_ONLY),
            "new structural 8200 gradients must have physically changed at least one FP16 weight")
    require(sha(s102_child)==S102_SHA
            and sha(s104_child)==origin["IN_CALIBRI_37087_MORPH_FP16_WEIGHTS_SHA256"],
            "original S102 and S104 immutable artifacts still same SHA")

    # DIAGNOSTIC SCORED ONLY AFTER ALL fixed epochs. No model choice.
    frozen=NRA3MorphologyInCalibriWeights().eval()
    frozen.load_state_dict({k[len(PREFIX):]:v.float().clone() for k,v in old.items()},strict=True)
    base=teacher_anchored_diagnostic(frozen,hidden,labels,labels.internal_dev)
    result=teacher_anchored_diagnostic(net,hidden,labels,labels.internal_dev)
    train_score=teacher_anchored_diagnostic(net,hidden,labels,labels.train)

    paid_state={PREFIX+k:v.half().cpu().contiguous()
                for k,v in current.items()}
    require(sum(v.numel() for v in paid_state.values())==TOTAL_PARAMS
            and all(torch.isfinite(z).all() for z in paid_state.values()),
            "save complete original in-model neural child only, no GT or CB59")
    out.mkdir(parents=True,exist_ok=False)
    save_file(paid_state,str(out/OUT_FILE))
    back=load_file(str(out/OUT_FILE),device="cpu")
    require(set(back)==set(paid_state)
            and all(torch.equal(back[k],v) for k,v in paid_state.items()),
            "real FP16 trained Calibri child exact round-trip")
    manifest={
        "stage":STAGE,
        "S104_original12_source_SHA":origin["ALL12_CPU_SAFETENSORS_SHA256"],
        "S104_untrained_37087_child_SHA":origin["IN_CALIBRI_37087_MORPH_FP16_WEIGHTS_SHA256"],
        "S102_prior_trained_28887_child_SHA":S102_SHA,
        "S105_trained_37087_in_calibri_child_SHA":sha(out/OUT_FILE),
        "source_only_PINNED_Taiga_E_SHA":"5fa1c234cf2bd53aee62eb8c4faba9a8ec116c0e70f7b8c5e2b13a11f4d574fc",
        "offline_Taiga_E_GOLD_SHA":"6f0c05aa988c500693d088a4147159ad106b10601e388ee317690fab7570a2bb",
        "original_Calibri_GT_CB59_weights_untouched":True,
        "original_616_J7_replayed_before_S104":True,
        "all_12_original_Calibri_GT_hidden_source_frozen":True,
        "S103_offline_gold_opened_ONLY_after_physical_S104_source_SHA_verified":True,
        "all_prior_28887_neural_weights_remained_FP16_exact":True,
        "eight_NRA2_mini_areas_12layer_router_neural_memory_preserved":True,
        "only_new_8200_neural_affinity_parameters_trained":True,
        "old_S96_S99_S102_internal_development_SPENT_not_reused":True,
        "original_S79_S86_firstreads_SPENT_not_reused":True,
        "prospective_Taiga_TRAIN_c_RESERVED_UNOPENED":True,
        "fixed_train48_internal16_per64_no_dev_tuning":True,
        "teacher_known_predicate_anchor_DIAGNOSTIC_only_not_autonomous":True,
        "reported_speech_factuality_coreference_NOT_proven":True,
        "CC_BY_SA_4_attribution_and_commercial_legal_review_required":True,
        "not_new_active_616J7_or_online_causal_generation_validation":True,
        "not_all_in_bpp_le07_CPU_ge90_CB59_certified":True,
        "train_parameters_locked_before_all_dev_scoring":{
            "seed":SEED,"epochs":EPOCHS,"lr":LR,"weight_decay":WD,
            "frozen_old_params":OLD_PARAMS,"trained_new_params":EXTRA_PARAMS,
            "batches_per_epoch":len(batches)},
        "strict_original_BPE_TaigaE_ALIGNMENT":audit,
        "TRAIN_only_loss_trace":trace,
        "S102_pretrained_frozen_on_NEW_Taiga_E_same_corpus_internal_diag":base,
        "S105_new_8200_Taiga_E_same_corpus_internal_diag":result,
        "S105_Taiga_E_TRAIN_only":train_score,
        "never_use_S105_internal_diag_for_any_hparam_tuning":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (out/REPORT).write_text(json.dumps(
        manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("NRA3_S105_TRUE12_TAIGAE_ONLY_NEW_8200_NEURAL_AFFINITY_GATE0",
          json.dumps({
            "source_rows":audit["counts"]["full_UD_BPE_rows"],
            "positive_role_edges":audit["counts"]["positive_event_argument_edges"],
            "pretrained_frozen_S102_INTERNAL":base["multi_verb_role_teacher_anchored"],
            "new_S105_INTERNAL":result["multi_verb_role_teacher_anchored"],
            "TRAIN":train_score["multi_verb_role_teacher_anchored"],
            "new_trainable_params":paid,"inmodel_total":TOTAL_PARAMS,"gate":0,
          },sort_keys=True,ensure_ascii=False),flush=True)
    return manifest

def main():
    ap=argparse.ArgumentParser()
    for key in ("source12","source_report","s104_child","s102_child","s103_gold_folder","out"):
        ap.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=ap.parse_args()
    train(a.source12,a.source_report,a.s104_child,a.s102_child,a.s103_gold_folder,a.out)

if __name__=="__main__":
    main()
