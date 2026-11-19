"""S111 fixed GSD-TRAIN-source-only, GOLD-BLIND diagnostics of frozen S110.

Purpose: quantify whether structural candidate changed original BPE logits,
native-token argmaxes, entropy, role gates and margins, not correctness.
No training, no UD labels or S110 INTERNAL developer report read. No
previous heldout / spent TaigaC, source only first16 of each 64 GSD
TRAIN stratum. All 616 original J7 protected in physical S109.
S110's SAME-GSD internal diagnostic was regressive and is permanently SPENT.
This cannot support a decision to tune or activate model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import math

import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import (
    NRA3RelationalMorphologyInCalibriWeights,
)
from research.calibri_nra3.s107_in_calibri_structural_role_competition_53743 import (
    NRA3StructuralCompetitionInCalibriWeights,NEW_KEYS,TOTAL_NEURAL_PARAMETERS,
)

STAGE="CALIBRI_NRA3_S111_REAL12_GSD_GOLDBLIND_TRAIN_SOURCE_ONLY_STRUCTURAL_GATE_RANK_AUDIT"
PHYSICAL_SOURCE_SHA="5880ea08df06587a8a75a628dd4c55d94c89da9f1f2fa3aa6ab10c6ebd40f6b4"
S109_NAME="NRA3_S109_IN_CALIBRI_UNTRAINED_53743_FP16.safetensors"
S110_NAME="NRA3_S110_GSD_ORIGINAL12_NEW16656_STRUCTURAL_53743_IN_CALIBRI_FP16.safetensors"
SOURCE="NRA3_S109_GSD_TRAIN256_ALL12_ORIGINAL_H320_NO_GOLD.safetensors"
MANIFEST="NRA3_S109_REAL43MGT_GSD_TRAIN256_ALLVERBS_SOURCE_PROVENANCE.json"
OUT="NRA3_S111_GSD_SOURCE_ONLY_GOLDBLIND_STRUCTURAL_RANK_GATE_AUDIT.json"
PREFIX="model.nra3_neural_weights."

def require(ok,reason):
    if not bool(ok):
        raise ValueError("NRA3 S111 source-only GOLD-BLIND original12 audit: "+reason)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.inference_mode()
def audit(source_path,manifest_path,pre_s109,post_s110,out):
    require(not out.exists(),"one physical post-hoc audit report, no overwrite")
    require(source_path.name==SOURCE and manifest_path.name==MANIFEST
            and pre_s109.name==S109_NAME and post_s110.name==S110_NAME,
            "strict original S109 source and frozen S110 neural only; no labels")
    source_meta=json.loads(manifest_path.read_text(encoding="utf8"))
    require(source_meta["stage"]==
            "CALIBRI_NRA3_S109_REAL43MGT_ALL12_GSD_TRAIN256_ALLVERBS_SOURCE_ONLY_AFTER616"
            and sha(source_path)==PHYSICAL_SOURCE_SHA==
                source_meta["ALL12_CPU_SAFETENSORS_SHA256"]
            and source_meta["S108_GSD_TRAIN_PINNED_SOURCE256_SHA256"]==
                "4078f408f2355c55c29bca18504355379e2a42ef2144c9113dda9448b086de14"
            and source_meta["S107_pretrained_53743_in_model_child_FP16_SHA256"]==
                "46ca9d5c946168332b4d6d1108d442d38c9628fe7bd4afb338693f1d56e27cee"
            and source_meta["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and source_meta["source_ONLY_no_S108_GSD_UD_GOLD_loaded_in_original_CPU"]
            and source_meta["output_gate"]==0
            and not source_meta["candidate_accepted"]
            and not source_meta["activation_allowed"],
            "verified true original12 source-only Calibri after protected J7")
    require(sha(pre_s109)==source_meta["IN_CALIBRI_53743_MORPH_FP16_WEIGHTS_SHA256"],
            "previous full in-Calibri 53,743 neural child physical SHA")
    a=load_file(str(pre_s109),device="cpu")
    b=load_file(str(post_s110),device="cpu")
    require(set(a)==set(b)
            and sum(x.numel() for x in a.values())==TOTAL_NEURAL_PARAMETERS==53743
            and sum(x.numel() for x in b.values())==53743
            and all(x.dtype==torch.float16 for x in a.values())
            and all(x.dtype==torch.float16 for x in b.values()),
            "only pinned actual S109 and S110 FP16 neural checkpoints, not external parser")
    old={k:v for k,v in a.items() if k[len(PREFIX):] not in NEW_KEYS}
    require(sum(x.numel() for x in old.values())==37087
            and all(torch.equal(v,b[k]) for k,v in old.items())
            and any(not torch.equal(a[PREFIX+k],b[PREFIX+k]) for k in NEW_KEYS),
            "old 37,087 including NRA2 mini-areas+router+memory 100% frozen; new 16,656 trained")
    oldnet=NRA3RelationalMorphologyInCalibriWeights().eval()
    oldnet.load_state_dict({k[len(PREFIX):]:v.float() for k,v in old.items()},strict=True)
    trained=NRA3StructuralCompetitionInCalibriWeights().eval()
    trained.load_state_dict({k[len(PREFIX):]:v.float() for k,v in b.items()},strict=True)
    require(trained.budget()["new_neural_parameters"]==53743
            and trained.output_gate==0 and not trained.candidate_accepted
            and not trained.activation_allowed,
            "trained 53,743 in-Calibri graph must remain physically gate0")
    gate=torch.tanh(trained.structural_competition_gate).tolist()
    require(len(gate)==len(ROLE_NAMES)==8 and all(math.isfinite(v) for v in gate),
            "8 learned structural gates can be analyzed but output stays OFF")
    original=load_file(str(source_path),device="cpu")
    h=original["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    valid=original["native_BPE_source_valid_mask"]
    src_idx=original["source_row_indices_in_PINNED_GSD_TRAIN256"].long()
    require(h.ndim==4 and h.shape[1]==12 and h.shape[-1]==320
            and h.shape[0]==len(src_idx) and h.dtype==torch.float16
            and valid.shape==(h.shape[0],h.shape[2])
            and src_idx.tolist()==sorted(set(src_idx.tolist())),
            "true original Calibri 12 layer BPE source-only geometry")
    # Predeclared source-only selection (no knowledge of predicate, role or gold).
    chosen=torch.where((src_idx%64)<16)[0].tolist()
    for stratum in range(4):
        n=sum(64*stratum<=int(src_idx[x])<64*(stratum+1) for x in chosen)
        require(n>=8,"at least eight source-only rows per predeclared GSD TRAIN stratum")
    require(len(chosen)>=32 and all(int(src_idx[x])%64<16 for x in chosen),
            "only first16 original TRAIN-source per64, never same-GSD internal developer")
    stats={name:{
        "fixed_unlabeled_predicate_positions":0,
        "top1_native_argument_predictions_changed":0,
        "sum_absolute_original_rank_top1_minus_top2":0.,
        "sum_mean_absolute_neural_logit_delta":0.,
        "sum_JS_divergence_native_argument_distribution":0.,
        "new_arg_before_predicate":0,
        "new_arg_after_predicate":0,
        "new_arg_same_token":0,
    } for name in ROLE_NAMES}
    zero_gate_checked=False
    for x in chosen:
        orig_hidden=h[x:x+1].float().detach()
        mask=valid[x:x+1]
        baseline=oldnet(orig_hidden,mask)
        new=trained(orig_hidden,mask)
        # Confirm S110 frozen inherited morphology remains byte-identical.
        require(all(torch.equal(getattr(baseline,z),getattr(new,z)) for z in
                    ("event_mention_logits","event_state_logits","mini_area_logits",
                     "layer_router_weights","memory_slot_attention",
                     "upos_logits","case_logits","number_logits","voice_logits")),
                "only structural 16,656 change arcs, not eight NRA2 mini-areas or morphology")
        if not zero_gate_checked:
            backup=trained.structural_competition_gate.detach().clone()
            trained.structural_competition_gate.zero_()
            gated=trained(orig_hidden,mask)
            require(torch.equal(gated.role_arc_logits,baseline.role_arc_logits),
                    "returning competition gate to zero restores exact old logits")
            trained.structural_competition_gate.copy_(backup)
            zero_gate_checked=True
        positions=torch.where(mask[0])[0]
        require(len(positions)>=2,"native source BPE must contain at least 2 pieces")
        # At most 12 deterministic equally spaced unknown-predicate tokens.
        stride=max(1,len(positions)//12)
        sample=positions[::stride][:12].tolist()
        for role_id,name in enumerate(ROLE_NAMES):
            st=stats[name]
            for p in sample:
                old_scores=baseline.role_arc_logits[0,role_id,p,positions].float()
                new_scores=new.role_arc_logits[0,role_id,p,positions].float()
                require(bool(torch.isfinite(old_scores).all())
                        and bool(torch.isfinite(new_scores).all()),"no nonfinite source scores")
                old_log=F.log_softmax(old_scores,dim=-1)
                new_log=F.log_softmax(new_scores,dim=-1)
                distribution_old=old_log.exp()
                distribution_new=new_log.exp()
                mix=(distribution_old+distribution_new)*0.5
                js=(0.5*torch.sum(distribution_old*
                    (old_log-torch.log(mix.clamp_min(1.e-30))))
                   +0.5*torch.sum(distribution_new*
                    (new_log-torch.log(mix.clamp_min(1.e-30)))))
                idx_old=int(old_scores.argmax())
                idx_new=int(new_scores.argmax())
                target=int(positions[idx_new])
                tops=torch.topk(old_scores,2).values
                st["fixed_unlabeled_predicate_positions"]+=1
                st["top1_native_argument_predictions_changed"]+=int(idx_old!=idx_new)
                st["sum_absolute_original_rank_top1_minus_top2"]+=float(tops[0]-tops[1])
                st["sum_mean_absolute_neural_logit_delta"]+=float((new_scores-old_scores).abs().mean())
                st["sum_JS_divergence_native_argument_distribution"]+=float(js)
                st["new_arg_before_predicate"]+=int(target<p)
                st["new_arg_after_predicate"]+=int(target>p)
                st["new_arg_same_token"]+=int(target==p)
    for st in stats.values():
        n=st["fixed_unlabeled_predicate_positions"]
        require(n>0,"fixed source-only tokens for every neural role")
        st["changed_fraction_no_accuracy_implication"]=st["top1_native_argument_predictions_changed"]/n
        st["mean_baseline_top1_margin"]=st.pop("sum_absolute_original_rank_top1_minus_top2")/n
        st["mean_abs_logit_delta"]=st.pop("sum_mean_absolute_neural_logit_delta")/n
        st["mean_Jensen_Shannon_divergence"]=st.pop("sum_JS_divergence_native_argument_distribution")/n
        require(all(math.isfinite(st[k]) for k in (
            "mean_baseline_top1_margin","mean_abs_logit_delta",
            "mean_Jensen_Shannon_divergence")),
            "only finite no-GOLD source metrics")
    report={
        "stage":STAGE,
        "genuine_original12_S109_CB59_GT_physical_source_SHA256":PHYSICAL_SOURCE_SHA,
        "S109_registered_53743_FP16_child_SHA256":sha(pre_s109),
        "S110_train_only_53743_FP16_child_SHA256":sha(post_s110),
        "only_first16of64_GSD_TRAIN_source_rows_examined":True,
        "n_fixed_source_only_rows":len(chosen),
        "n_supervision_UD_gold_or_teacher_labels_loaded":0,
        "no_S110_spent_INTERNAL_developer_report_or_labels_downloaded_for_use":True,
        "no_TaigaC_burned_firstread_or_old_dev_used":True,
        "zero_structural_gate_exactly_recovers_previous_inmodel_role_logits":zero_gate_checked,
        "old_37087_NRA2_router_memory_morphology_FP16_exact":True,
        "new_16656_structural_params_physically_learned":True,
        "learned_tanh_structural_role_gate":dict(zip(ROLE_NAMES,gate)),
        "only_unlabeled_predictions_no_accuracy_or_gain_proof":True,
        "source_ONLY_unlabeled_rank_geometry_per_role":stats,
        "original616_J7_physically_protected":True,
        "no_model_training_or_weight_change_in_S111":True,
        "CC_BY_SA_4_LICENSE_legal_review_before_commercial":True,
        "no_main_live_or_candidate_certification":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/OUT).write_text(json.dumps(report,sort_keys=True,
                         ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("NRA3_S111_GSD_TRUE12_GOLDBLIND_STRUCTURAL_LOGIT_AUDIT",
          json.dumps({"rows":len(chosen),"gates":report["learned_tanh_structural_role_gate"],
                      "unlabeled":stats,"gate":0},sort_keys=True),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    for k in ("source_path","manifest_path","pre_s109","post_s110","out"):
        p.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    args=p.parse_args()
    audit(args.source_path,args.manifest_path,args.pre_s109,args.post_s110,args.out)

if __name__=="__main__":
    main()
