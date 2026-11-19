"""S110 fixed new Russian UD-GSD TRAIN-only Calibri12 structured neural learning.
Uses 616 J7 protected S109 original 12x320 CB59+GT native BPE source.
Load offline GOLD only after source SHA. Freeze existing 37,087 in-Calibri
neural parameters including original eight NRA2 minis, router, memory,
morphology. Train only 16,656 S107 Q/K, role bilinear, distance, gates.
All prior developer and Taiga-C firstread are SPENT. Gate0, no LIVE.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file,save_file
from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import NRA3RelationalMorphologyInCalibriWeights
from research.calibri_nra3.s107_in_calibri_structural_role_competition_53743 import (
    NRA3StructuralCompetitionInCalibriWeights, NEW_KEYS, NEW_NEURAL_PARAMETERS, OLD_NEURAL_PARAMETERS, TOTAL_NEURAL_PARAMETERS,
)
from research.calibri_nra3.s100_offline_allverb_multiargument_nativeBPE_train_loss import OriginalBPEAllVerbsEdges,offline_all_verb_multi_positive_relation_loss
from research.calibri_nra3.s110_GSD_real12_gold_AFTER_S109_sourceSHA import load_source_VERIFY_sha_THEN_OPEN_gold,UNKNOWN

STAGE="CALIBRI_NRA3_S110_ORIGINAL12_GSD_NEW16656_STRUCTURAL_IN_CALIBRI_TRAINONLY_GATE0"
PREFIX="model.nra3_neural_weights."
SEED=1101120
EPOCHS=24
LR=0.0015
WD=0.01
CHILD_SOURCE="NRA3_S109_IN_CALIBRI_UNTRAINED_53743_FP16.safetensors"
TRAINED="NRA3_S110_GSD_ORIGINAL12_NEW16656_STRUCTURAL_53743_IN_CALIBRI_FP16.safetensors"
REPORT="NRA3_S110_GSD_ORIGINAL12_STRUCTURAL_TRAIN_ONLY_DIAGNOSTICS.json"

def require(c,why):
    if not bool(c):
        raise ValueError("S110 true-original12 GSD structural train-only: "+why)

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




def train(source12,source_report,s109_child,s108_gold_folder,out):
    require(not out.exists(),"never overwrite one-shot new GSD TRAIN shadow")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    # S110 must first check physically protected original616 source SHA BEFORE GOLD opens.
    physical,aligned,audit=load_source_VERIFY_sha_THEN_OPEN_gold(
        source12,source_report,s108_gold_folder)
    provenance=json.loads(source_report.read_text(encoding="utf8"))
    require(s109_child.name==CHILD_SOURCE
        and sha(s109_child)==provenance["IN_CALIBRI_53743_MORPH_FP16_WEIGHTS_SHA256"]
        and provenance["S107_pretrained_53743_in_model_child_FP16_SHA256"]==
            "46ca9d5c946168332b4d6d1108d442d38c9628fe7bd4afb338693f1d56e27cee"
        and provenance["prior_37087_and_new_16656_state_exactly_restored_from_S107"]
        and provenance["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
        and provenance["output_gate"]==0 and not provenance["candidate_accepted"],
        "S109 real Calibri neural 53743 child and original J7 physically pinned")
    h=physical["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    require(h.ndim==4 and h.shape[1]==12 and h.shape[-1]==320
            and h.dtype==torch.float16 and torch.isfinite(h).all()
            and not h.requires_grad and h.shape[0]==len(aligned.original_source_row_indices),
            "frozen original genuine CB59-GT original-native-BPE layer vectors")
    source_weights=load_file(str(s109_child),device="cpu")
    require(sum(v.numel() for v in source_weights.values())==53743==TOTAL_NEURAL_PARAMETERS
            and OLD_NEURAL_PARAMETERS==37087 and NEW_NEURAL_PARAMETERS==16656
            and all(k.startswith(PREFIX) and v.dtype==torch.float16
                    and bool(torch.isfinite(v).all()) for k,v in source_weights.items()),
            "no original core weights in source only 53,743 child")
    model=NRA3StructuralCompetitionInCalibriWeights()
    model.load_state_dict({k[len(PREFIX):]:v.float().clone()
                           for k,v in source_weights.items()},strict=True)
    require(model.budget()["new_neural_parameters"]==53743
            and torch.count_nonzero(model.structural_competition_gate)==0
            and model.output_gate==0 and not model.activation_allowed
            and not model.candidate_accepted and not model.is_trained,
            "new S107 competition starts exactly gate-zero on genuine S109 source")
    for k,p in model.named_parameters():
        p.requires_grad_(k in NEW_KEYS)
    trainable=[p for p in model.parameters() if p.requires_grad]
    require(sum(p.numel() for p in trainable)==16656
            and sum(p.numel() for p in model.parameters() if not p.requires_grad)==37087,
            "train only 16,656 new in-Calibri neural graph weights")
    batches=batches_fixed(aligned.original_source_row_indices,aligned.train)
    opt=torch.optim.AdamW(trainable,lr=LR,weight_decay=WD)
    trace=[]
    for epoch in range(EPOCHS):
        model.train()
        losses=[]
        for ix in batches:
            require(bool(aligned.train[ix].all()) and not bool(aligned.internal_dev[ix].any()),
                    "no same-corpus dev used for TRAIN or selection")
            opt.zero_grad(set_to_none=True)
            graph=model(h[ix].float().detach(),aligned.role.native_BPE_valid[ix])
            loss,_=offline_all_verb_multi_positive_relation_loss(
                graph,select_targets(aligned.role,ix))
            require(torch.isfinite(loss),"positive-only original UD predicate-argument loss")
            loss.backward()
            require(model.structural_competition_gate.grad is not None
                    and bool(torch.isfinite(model.structural_competition_gate.grad).all()),
                    "differentiable learned structural competition gate")
            nn.utils.clip_grad_norm_(trainable,1.0)
            opt.step()
            losses.append(float(loss.detach()))
        if epoch in (0,5,11,17,EPOCHS-1):
            trace.append({"epoch":epoch,"TRAIN_ONLY_loss":sum(losses)/len(losses)})
    model.eval()
    model.is_trained=True
    paid={PREFIX+k:v.detach().half().cpu().contiguous()
          for k,v in model.state_dict().items()}
    require(sum(v.numel() for v in paid.values())==53743
            and all(torch.isfinite(v).all() for v in paid.values())
            and all(torch.equal(paid[k],v) for k,v in source_weights.items()
                    if k[len(PREFIX):] not in NEW_KEYS),
            "old original 37,087 NRA2/memory/12router/morphology weights bit identical")
    require(any(not torch.equal(paid[PREFIX+k],source_weights[PREFIX+k]) for k in NEW_KEYS),
            "new structural neural gradient updates physically changed FP16 weights")
    # First INTERNAL diagnostic AFTER fixed optimizer/training, never tune on it.
    frozen=NRA3RelationalMorphologyInCalibriWeights().eval()
    frozen.load_state_dict({k[len(PREFIX):]:v.float().clone()
                            for k,v in source_weights.items()
                            if k[len(PREFIX):] not in NEW_KEYS},strict=True)
    baseline=teacher_anchored_diagnostic(frozen,h,aligned,aligned.internal_dev)
    final=teacher_anchored_diagnostic(model,h,aligned,aligned.internal_dev)
    training=teacher_anchored_diagnostic(model,h,aligned,aligned.train)
    require(sha(s109_child)==provenance["IN_CALIBRI_53743_MORPH_FP16_WEIGHTS_SHA256"],
            "physical S109 source-and-child remain immutable after training")
    out.mkdir(parents=True,exist_ok=False)
    save_file(paid,str(out/TRAINED))
    reread=load_file(str(out/TRAINED))
    require(set(reread)==set(paid)
            and all(torch.equal(paid[k],v) for k,v in reread.items()),
            "physical FP16 53,743 learned neural child roundtrip")
    manifest={
        "stage":STAGE,
        "S109_original_all12_source_SHA":provenance["ALL12_CPU_SAFETENSORS_SHA256"],
        "S109_original_registered_in_Calibri_child_SHA":sha(s109_child),
        "S110_trained_in_Calibri_neural_53743_FP16_SHA":sha(out/TRAINED),
        "original_CB59_GT_616_J7_unchanged":True,
        "precommitted_NEW_GSD_source_SHA":"4078f408f2355c55c29bca18504355379e2a42ef2144c9113dda9448b086de14",
        "precommitted_offline_GSD_GOLD_SHA":"e3d33d2f21986e960c22ee23503a473e1a34119ce5d3e4015ccb33eb5b869121",
        "GOLD_opened_ONLY_AFTER_original_S109_source_SHA":True,
        "only_new_16656_structural_neural_params_trained":True,
        "old_37087_in_Calibri_params_FP16_identical":True,
        "NRA2_8_mini_areas_original12_router_16_weight_memory_preserved":True,
        "S96_S99_S102_S105_internal_devs_SPENT_not_used":True,
        "S79_S86_old_firstreads_SPENT_not_used":True,
        "TaigaC_one_shot_firstread_BURNED_NOT_USED":True,
        "GSD_internal_diag_scored_ONCE_after_fixed_epochs_no_tuning":True,
        "no_autonomous_event_recognition_factuality_or_independent_firstread":True,
        "CC_BY_SA_4_license_legal_review_before_commercial":True,
        "no_new_live_or_main_or_merged_or_CPU_BPP_or_causal_certificate":True,
        "fixed_training":{"seed":SEED,"epochs":EPOCHS,"lr":LR,
                          "weight_decay":WD,"new_grad_params":16656,
                          "old_frozen":37087,"batches":len(batches)},
        "original_native_BPE_GSD_alignment":audit,
        "TRAIN_loss_trace":trace,
        "previous_frozen_S105_on_new_GSD_internal":baseline,
        "new_S110_on_new_GSD_internal":final,
        "new_S110_GSD_TRAIN_ONLY":training,
        "internal_GSD_dev_SPENT_no_reuse_after_this":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (out/REPORT).write_text(json.dumps(
        manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("NRA3_S110_NEW_RUSSIAN_GSD_STRUCTURAL_TRAINONLY_NEURAL",
        json.dumps({"n_rows":audit["counts"]["full_UD_BPE_rows"],
                    "n_edges":audit["counts"]["positive_event_argument_edges"],
                    "baseline":baseline["multi_verb_role_teacher_anchored"],
                    "after":final["multi_verb_role_teacher_anchored"],
                    "TRAIN":training["multi_verb_role_teacher_anchored"],
                    "gate":0,"new_grad_params":16656},
                   ensure_ascii=False,sort_keys=True),flush=True)
    return manifest

def main():
    p=argparse.ArgumentParser()
    for k in ("source12","source_report","s109_child","s108_gold_folder","out"):
        p.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    train(a.source12,a.source_report,a.s109_child,a.s108_gold_folder,a.out)

if __name__=="__main__":
    main()
