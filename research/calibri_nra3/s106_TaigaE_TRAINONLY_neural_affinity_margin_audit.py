"""S106: frozen TRAIN-only neural rank-margin audit of S105 (NO training or DEV reads).

Scientific question: did the 8,200 S105 learned in-Calibri relational weights
change native BPE predicate→argument scores but fail to flip rank-1, or
were their score corrections numerically negligible? Reuse S105 *TRAIN*
rows solely; never inspect S105 internal-dev labels, old spent dev, or
reserved prospective Taiga TRAIN-c. Do not adjust S105 or tune on the
already-spent same-corpus diagnostic. No re-training or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
from safetensors.torch import load_file
from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import NRA3MorphologyInCalibriWeights
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import NRA3RelationalMorphologyInCalibriWeights
from research.calibri_nra3.s105_original12_TaigaE_UD_gold_after_S104_SHA import load_source_VERIFY_sha_THEN_OPEN_gold

STAGE="CALIBRI_NRA3_S106_TRUE12_TAIGA_E_TRAIN_ONLY_MORPH_AFFINITY_RANK_MARGIN_FORENSICS"
BASE_SHA="f40dddbc03282a6c45c4f807746275d76a4ba776aa613247e73fbdf7f43329df"
PREFIX="model.nra3_neural_weights."
REPORT="NRA3_S106_TAIGA_E_TRAIN_ONLY_AFFINITY_SCORE_MARGIN_AUDIT.json"

def require(ok,why):
    if not bool(ok):
        raise ValueError("S106 frozen TRAIN-only margin audit: "+why)

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@torch.inference_mode()
def audit(source12,source_report,previous_s102,learned_s105,s105_report,offline_s103,out):
    require(not out.exists(),"one frozen audit report only; never overwrite")
    torch.set_num_threads(2)
    # This loader MUST independently recheck S104 actual original12
    # source HASH before opening NEW Taiga E UD gold.
    source,truth,provenance=load_source_VERIFY_sha_THEN_OPEN_gold(
        source12,source_report,offline_s103)
    origin=json.loads(source_report.read_text(encoding="utf8"))
    stage105=json.loads(s105_report.read_text(encoding="utf8"))
    require(stage105["stage"]==
             "CALIBRI_NRA3_S105_TAIGA_E_REAL12_8200_AFFINITY_ONLY_INMODEL_NEURAL_TRAIN_GATE0"
            and stage105["S105_trained_37087_in_calibri_child_SHA"]==file_sha(learned_s105)
            and stage105["S102_prior_trained_28887_child_SHA"]==BASE_SHA
            and stage105["S104_original12_source_SHA"]==origin["ALL12_CPU_SAFETENSORS_SHA256"]
            and stage105["all_prior_28887_neural_weights_remained_FP16_exact"]
            and stage105["only_new_8200_neural_affinity_parameters_trained"]
            and stage105["prospective_Taiga_TRAIN_c_RESERVED_UNOPENED"]
            and stage105["output_gate"]==0
            and not stage105["candidate_accepted"]
            and not stage105["activation_allowed"],
            "only S105 frozen research checkpoint/source allowed, no prior DEV results used")
    require(previous_s102.name=="NRA3_S102_TAIGA_D_IN_CALIBRI_ALLVERB_ROLE_28887_TRAINED_FP16.safetensors"
            and file_sha(previous_s102)==BASE_SHA,
            "previous 28887 FP16 base must be exact frozen S102 weights")
    original=load_file(str(previous_s102),device="cpu")
    trained=load_file(str(learned_s105),device="cpu")
    require(set(original)<set(trained)
            and sum(z.numel() for z in original.values())==28887
            and sum(z.numel() for z in trained.values())==37087
            and all(torch.equal(trained[k],v) for k,v in original.items()),
            "old eight NRA2 mini areas, memory, router, morphology and arcs exactly frozen")
    b=NRA3MorphologyInCalibriWeights().eval()
    b.load_state_dict({k[len(PREFIX):]:v.float() for k,v in original.items()},strict=True)
    n=NRA3RelationalMorphologyInCalibriWeights().eval()
    n.load_state_dict({k[len(PREFIX):]:v.float() for k,v in trained.items()},strict=True)
    gates=torch.tanh(n.role_morph_affinity_gate).tolist()
    require(len(gates)==8
            and bool(torch.isfinite(n.role_morph_affinity).all())
            and n.output_gate==0 and not n.candidate_accepted and not n.activation_allowed,
            "8 gated per-role IN CALIBRI morph relational weights, output still gate0")
    hidden=source["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    train_rows=torch.where(truth.train)[0].tolist()
    require(len(train_rows)>=96 and not any(not bool(truth.train[x]) for x in train_rows)
            and not any(bool(truth.internal_dev[x]) for x in train_rows),
            "strict previous TRAIN-e only; no old or internal dev labels evaluated")
    roles=[{"name":name,"TRAIN_known_role_slots":0,
            "frozen_S102_top1_correct":0,"learned_S105_top1_correct":0,
            "TRAIN_top1_predictions_changed":0,"sum_abs_logit_delta_at_gold_predicate":0.,
            "max_abs_logit_delta_at_gold_predicate":0.,
            "base_margin_top1_minus_top2":[],"delta_to_margin_ratio":[]}
           for name in ROLE_NAMES]
    for x in train_rows:
        z=hidden[x:x+1].float().detach()
        valid=truth.role.native_BPE_valid[x:x+1]
        old=b(z,valid)
        new=n(z,valid)
        require(torch.equal(old.upos_logits,new.upos_logits)
                and torch.equal(old.case_logits,new.case_logits)
                and torch.equal(old.number_logits,new.number_logits)
                and torch.equal(old.voice_logits,new.voice_logits)
                and torch.equal(old.mini_area_logits,new.mini_area_logits)
                and torch.equal(old.memory_slot_attention,new.memory_slot_attention),
                "only previously frozen 28887 remain identical before/after")
        candidate=torch.where(valid[0])[0]
        require(len(candidate)>1,"native BPE original source has multiple valid tokens")
        for j,name in enumerate(ROLE_NAMES):
            for pred in torch.where(truth.role.event_role_supervised[x,j])[0].tolist():
                reference=old.role_arc_logits[0,j,pred,candidate].float()
                adapted=new.role_arc_logits[0,j,pred,candidate].float()
                positives=truth.role.argument_gold_positive[x,j,pred]
                op=int(candidate[int(reference.argmax())])
                np=int(candidate[int(adapted.argmax())])
                gap=torch.topk(reference,2).values
                margin=float((gap[0]-gap[1]).abs())
                maxdelta=float((adapted-reference).abs().max())
                report=roles[j]
                report["TRAIN_known_role_slots"]+=1
                report["frozen_S102_top1_correct"]+=int(bool(positives[op]))
                report["learned_S105_top1_correct"]+=int(bool(positives[np]))
                report["TRAIN_top1_predictions_changed"]+=int(op!=np)
                report["sum_abs_logit_delta_at_gold_predicate"]+=float((adapted-reference).abs().mean())
                report["max_abs_logit_delta_at_gold_predicate"]=max(
                    report["max_abs_logit_delta_at_gold_predicate"],maxdelta)
                report["base_margin_top1_minus_top2"].append(margin)
                report["delta_to_margin_ratio"].append(maxdelta/(margin+1.e-9))
    for r in roles:
        nslots=r["TRAIN_known_role_slots"]
        if nslots:
            r["mean_abs_logit_delta"]=r.pop("sum_abs_logit_delta_at_gold_predicate")/nslots
            r["median_frozen_rank_margin"]=statistics.median(r.pop("base_margin_top1_minus_top2"))
            r["median_max_delta_over_rank_margin"]=statistics.median(r.pop("delta_to_margin_ratio"))
        else:
            r.pop("sum_abs_logit_delta_at_gold_predicate")
            r.pop("base_margin_top1_minus_top2")
            r.pop("delta_to_margin_ratio")
            r["mean_abs_logit_delta"]=None
            r["median_frozen_rank_margin"]=None
            r["median_max_delta_over_rank_margin"]=None
        require(r["TRAIN_known_role_slots"]>=0
                and r["TRAIN_top1_predictions_changed"]>=0
                and (r["mean_abs_logit_delta"] is None or r["mean_abs_logit_delta"]>=0),
                "original known-word TRAIN-only margin report valid; unknown roles NOT negative")
    summary={
        "stage":STAGE,
        "original12_source_SHA":origin["ALL12_CPU_SAFETENSORS_SHA256"],
        "frozen_S102_neural_child_SHA":BASE_SHA,
        "S105_frozen_neural_child_SHA":file_sha(learned_s105),
        "precommitted_original_Taiga_E_NEW_source_only_SHA":origin["S103_TAIGA_TRAIN_E_PINNED_SOURCE256_SHA256"],
        "source_SHAs_verified_before_offline_GOLD":True,
        "n_STRICT_TRAIN_only_rows":len(train_rows),
        "n_strict_source_native_BPE_aligned":provenance["counts"]["full_UD_BPE_rows"],
        "only_training_data_role_positions_used":True,
        "S105_internal_dev_scores_and_labels_NEVER_accessed_in_audit":True,
        "S96_S99_S102_old_devs_S79_S86_firstreads_NOT_accessed":True,
        "original_Taiga_TRAIN_c_prospective_UNOPENED":True,
        "affinity_tanh_role_gates":dict(zip(ROLE_NAMES,gates)),
        "TRAIN_ONLY_per_role_S102_vs_S105_unchanged_base_params":roles,
        "trained_8200_weights_frozen_for_forensic_no_training":True,
        "no_independent_generalization_or_quality_claim":True,
        "original_616_J7_protected_no_model_activation":True,
        "no_all_in_bpp_or_CPU_certification":True,
        "licensed_CC_BY_SA_4_no_commercial_use_without_legal_review":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/REPORT).write_text(json.dumps(
        summary,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("NRA3_S106_TRAIN_ONLY_AFFINITY_MARGIN_NO_DEV",
          json.dumps({"rows":len(train_rows),
                      "role_gates":summary["affinity_tanh_role_gates"],
                      "per_role":roles,
                      "output_gate":0},ensure_ascii=False,sort_keys=True),flush=True)
    return summary

def main():
    a=argparse.ArgumentParser()
    for v in ("source12","source_report","previous_s102","learned_s105",
              "s105_report","offline_s103","out"):
        a.add_argument("--"+v.replace("_","-"),type=Path,required=True)
    q=a.parse_args()
    audit(q.source12,q.source_report,q.previous_s102,q.learned_s105,
          q.s105_report,q.offline_s103,q.out)

if __name__=="__main__":
    main()
