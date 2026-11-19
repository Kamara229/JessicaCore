"""S118 one-time disjoint SynTagRus-B ORIGINAL CALIBRI12 neural structural trust first-read.

S116 preregistered 256 completely unseen licensed NONCOMMERCIAL SynTagRus
TRAIN-b sentences source/GOLD SHA, S115 eight bounded trust gates trained
ONLY on different TRAIN-a, frozen 53,743 old in-Calibri neural weights.
S117 protected original 616 J7 CB59+GreenTunnel true12 source GOLD-BLIND.
This is ONE benchmark, never for tuning. Check S117 true12 SHA and old
S112/S115 FP16 neural child SHAs BEFORE opening B GOLD.
Compare teacher-anchored all-verb predicate->argument positive top1 on
EXACT SAME source positions, with paired sentence-cluster bootstrap.
No autonomous event detection or real-world factual QA inference.
B becomes PERMANENTLY SPENT after this first read regardless of outcome.
8 NRA2 mini areas, neural morphology, relational memory, all12 router
and original 616 weights unchanged; research output gate0.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random
from pathlib import Path

import torch
from safetensors.torch import load_file
from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES
from research.calibri_nra3.s112_in_calibri_bounded_structural_trust_53751 import (
    NRA3BoundedStructuralTrustInCalibriWeights,
)
from research.calibri_nra3.s118_SynTagRusB_NC_GOLD_AFTER_S117_AND_weightsSHA import (
    load_source_VERIFY_sha_THEN_OPEN_gold,
)
STAGE="CALIBRI_NRA3_S118_DISJOINT_SYNTAGRUS_B_FIRSTREAD_ONCE_PRECOMMITTED_BASE_VS_EIGHT_TRUST_NEURAL_GATE0"
REPORT="NRA3_S118_B_FIRSTREAD_ONCE_UD_RELATIONS_REJECT_OR_CONTINUE_NOT_ACTIVATE.json"
PREFIX="model.nra3_neural_weights."
S117_SOURCE="NRA3_S117_SYNTAGRUS_TRAINB256_ALL12_ORIGINAL_H320_NO_GOLD.safetensors"
S117_MANIFEST="NRA3_S117_REAL43MGT_SYNTAGRUS_TRAINB256_ALLVERBS_SOURCE_PROVENANCE.json"
S112_BASE="NRA3_S112_ORIGINAL12_BOUNDED_53751_IN_CALIBRI_ZERO_TRUST_FP16.safetensors"
S115_TRAINED="NRA3_S117_IN_CALIBRI_TRAINED_TRUST_53751_FP16.safetensors"
S112_MANIFEST="NRA3_S112_BOUNDED_ORIGINAL12_GSD_SOURCEONLY_NEURAL_ABI.json"
# Immutable physical source and trained FP16 child verified by S117 original CPU
# GitHub Actions #44625178799 SUCCESS 76/76, NO B teacher GOLD opened.
PRECOMMITTED_S117_TRUE12_SOURCE_SHA="74c43db51b7bf357fa09f1af6b318bfd018f7df2b66f227ffb9de5ae0ef216ac"
PRECOMMITTED_S115_TRAINED_53751_CHILD_SHA="42e0bb4e81cb4ea199e4ee8b5fccd939ab4550b7c6269af44ede6a77462d795c"

SEED=11820261120
BOOT=4000

def require(ok,reason):
    if not bool(ok):
        raise ValueError("S118 prospective B first-read sealed / immutable: "+reason)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def bootstrap_paired_rows(per_row,*,seed=SEED,replicates=BOOT):
    """Pairs never separated: sample ORIGINAL sentences, not role edges."""
    require(len(per_row)>=50 and replicates==4000,
            "at least 50 independent sentence clusters; fixed predeclared bootstrap")
    total=sum(x["role_slots"] for x in per_row)
    old=sum(x["base_correct"] for x in per_row)
    new=sum(x["trained_correct"] for x in per_row)
    require(total>0 and old<=total and new<=total,
            "every score must be on the same supervised word-origin slots")
    rng=random.Random(seed)
    n=len(per_row)
    differences=[]
    for _ in range(replicates):
        num=den=0
        for __ in range(n):
            row=per_row[rng.randrange(n)]
            num+=row["trained_correct"]-row["base_correct"]
            den+=row["role_slots"]
        if den:
            differences.append(num/den)
    require(len(differences)>=replicates-5,"stable finite sentence-cluster bootstrap")
    differences.sort()
    def q(p):
        return differences[min(len(differences)-1,int(p*(len(differences)-1)))]
    return {
        "n_original_source_sentence_clusters":n,
        "n_teacher_anchored_supervised_predicate_role_slots":total,
        "S112_zero_trust_base_correct":old,
        "S115_eight_trust_trained_correct":new,
        "base_top1_positive_accuracy":old/total,
        "trained_top1_positive_accuracy":new/total,
        "paired_delta_percentage_points":100*(new-old)/total,
        "paired_sentence_cluster_bootstrap_95pct_delta_pp":[100*q(.025),100*q(.975)],
        "bootstrap_fixed_seed":seed,"bootstrap_precommitted_iterations":replicates,
        "not_autonomous_predicate_detection_or_factual_QA":True,
    }

@torch.inference_mode()
def firstread(source12,source_manifest,s112_child,s112_report,
              s115_child,separate_b_gold,out,expected_source_sha):
    require(not out.exists(),"NEVER rerun or overwrite once-only new B gold firstread")
    require(source12.name==S117_SOURCE and source_manifest.name==S117_MANIFEST
            and s112_child.name==S112_BASE
            and s112_report.name==S112_MANIFEST and s115_child.name==S115_TRAINED,
            "ONLY precommitted ORIGINAL Calibri12 source + S112 and S115 FROZEN FP16 weights")
    require(expected_source_sha==PRECOMMITTED_S117_TRUE12_SOURCE_SHA,
            "MUST use GitHub-verified LITERAL S117 true12 SHA before B GOLD, not dynamic caller-selected SHA")
    source_info=json.loads(source_manifest.read_text(encoding="utf8"))
    baseline_info=json.loads(s112_report.read_text(encoding="utf8"))
    require(source_info["stage"]==
            "CALIBRI_NRA3_S117_REAL43MGT_ALL12_SYNTAGRUS_TRAINB256_ALLVERBS_SOURCE_ONLY_AFTER616"
            and source_info["ALL12_CPU_SAFETENSORS_SHA256"]==
                expected_source_sha==sha(source12)
            and source_info["S116_SYNTAGRUS_TRAIN_PINNED_SOURCE256_SHA256"]==
                "f687b1f677a3795ba4c4dbf3b5246e1105cf19f236f023e4d9c4d18b5418f9bf"
            and source_info["S116_SYNTAGRUS_TRAIN_original_git_blob"]==
                "416fcb93e223869e2718be30629dad9fadd4c290"
            and source_info["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and source_info["source_ONLY_no_S116_SYNTAGRUS_UD_GOLD_loaded_in_original_CPU"]
            and source_info["new_SynTagRus_TRAINB_firstread_RAW_ONLY_NO_PERFORMANCE_SCORED"]
            and source_info["S115_trainA_INTERNAL_dev_remains_UNSCORED"]
            and source_info["noncommercial_research_only_CC_BY_NC_SA_4_license"]
            and source_info["output_gate"]==0 and not source_info["candidate_accepted"],
            "physical genuine original 12layer B source, original616 and GOLD excluded")
    require(baseline_info["stage"]==
            "CALIBRI_NRA3_S112_TRUE_ORIGINAL12_SOURCEONLY_BOUNDED_53751_NEURAL_SHADOW_GATE0"
            and baseline_info["new_S112_53751_8trust_in_calibri_FP16_SHA256"]==sha(s112_child)
            and baseline_info["old_53743_learnt_neural_weights_FP16_exact"]
            and baseline_info["original_37k_NRA2_mini_areas_memory_morphology_router_unchanged"]
            and baseline_info["zero_trust_restores_original_old_all_neural_role_logits"]
            and baseline_info["new_neural_8_role_trust_params_UNTRAINED"]
            and baseline_info["output_gate"]==0 and not baseline_info["candidate_accepted"],
            "the S112 baseline MUST be physically committed BEFORE prospective B firstread")
    require(sha(s115_child)==PRECOMMITTED_S115_TRAINED_53751_CHILD_SHA
            and sha(s115_child)==source_info["S115_trained_53751_in_model_child_FP16_SHA256"]
            and sha(s115_child)==source_info["IN_CALIBRI_53751_MORPH_FP16_WEIGHTS_SHA256"],
            "S115 trained child immutable, physically attached in S117 original Calibri before B GOLD")
    old=load_file(str(s112_child),device="cpu")
    new=load_file(str(s115_child),device="cpu")
    require(set(old)==set(new) and sum(v.numel() for v in old.values())==53751
            and all(v.dtype==torch.float16 for v in old.values())
            and all(v.dtype==torch.float16 for v in new.values())
            and all(torch.equal(v,new[k]) for k,v in old.items()
                    if k!=PREFIX+"structural_trust_gate")
            and torch.count_nonzero(old[PREFIX+"structural_trust_gate"])==0
            and torch.count_nonzero(new[PREFIX+"structural_trust_gate"])>0,
            "only precommitted eight neural trust weights differ; 53743 bit-identical")
    torch.set_num_threads(2)
    base=NRA3BoundedStructuralTrustInCalibriWeights().eval()
    base.load_state_dict({k[len(PREFIX):]:v.float() for k,v in old.items()},strict=True)
    trained=NRA3BoundedStructuralTrustInCalibriWeights().eval()
    trained.load_state_dict({k[len(PREFIX):]:v.float() for k,v in new.items()},strict=True)
    require(base.output_gate==trained.output_gate==0
            and not base.candidate_accepted and not trained.candidate_accepted
            and not base.activation_allowed and not trained.activation_allowed,
            "both models RESEARCH shadow gate0, zero output integration")
    # ONLY NOW AFTER all source/child SHA and frozen base/trained checks
    # may OFFLINE B teacher GOLD be opened exactly once. Never train on it.
    original,aligned,audit=load_source_VERIFY_sha_THEN_OPEN_gold(
        source12,source_manifest,separate_b_gold)
    h=original["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    require(h.ndim==4 and h.shape[1]==12 and h.shape[-1]==320
            and h.dtype==torch.float16 and h.shape[0]==len(aligned.original_source_row_indices)
            and torch.isfinite(h).all(),
            "S117 original Calibri CB59 GT source12 after protected J7")
    per_row=[]
    per_role={name:{"role_slots":0,"base_correct":0,"trained_correct":0,
                    "both_correct":0,"base_only_correct":0,"trained_only_correct":0,
                    "both_wrong":0,"positive_edges":0,"argmax_changes":0}
              for name in ROLE_NAMES}
    unchanged_morph_count=0
    max_abs_residual=0.
    for i in range(h.shape[0]):
        source=h[i:i+1].float().detach()
        valid=aligned.role.native_BPE_valid[i:i+1]
        old_graph=base(source,valid)
        new_graph=trained(source,valid)
        require(all(torch.equal(getattr(old_graph,k),getattr(new_graph,k))
                    for k in ("event_mention_logits","event_state_logits",
                              "mini_area_logits","layer_router_weights",
                              "memory_slot_attention","upos_logits","case_logits",
                              "number_logits","voice_logits")),
                "all frozen 53743 old neural attributes preserved on genuine original12 B")
        unchanged_morph_count+=1
        pair=valid[:,None,:,None]&valid[:,None,None,:]
        delta=(new_graph.role_arc_logits-old_graph.role_arc_logits).abs()
        max_abs_residual=max(max_abs_residual,float(delta[pair.expand_as(delta)].max()))
        require(not bool((delta[pair.expand_as(delta)]>1.00002).any()),
                "bounded new trust FP16 original B source every candidate <=1 logit")
        row={"original_B_source_row_id":int(aligned.original_source_row_indices[i]),
             "role_slots":0,"base_correct":0,"trained_correct":0}
        for role_id,name in enumerate(ROLE_NAMES):
            for p in torch.where(aligned.role.event_role_supervised[i,role_id])[0].tolist():
                allowed=aligned.role.argument_gold_positive[i,role_id,p]
                prior_idx=int(old_graph.role_arc_logits[0,role_id,p].argmax())
                newer_idx=int(new_graph.role_arc_logits[0,role_id,p].argmax())
                correct_old=int(bool(allowed[prior_idx]))
                correct_new=int(bool(allowed[newer_idx]))
                row["role_slots"]+=1
                row["base_correct"]+=correct_old
                row["trained_correct"]+=correct_new
                bucket=per_role[name]
                bucket["role_slots"]+=1
                bucket["base_correct"]+=correct_old
                bucket["trained_correct"]+=correct_new
                bucket["positive_edges"]+=int(allowed.sum())
                bucket["both_correct"]+=int(correct_old and correct_new)
                bucket["base_only_correct"]+=int(correct_old and not correct_new)
                bucket["trained_only_correct"]+=int(correct_new and not correct_old)
                bucket["both_wrong"]+=int(not correct_old and not correct_new)
                bucket["argmax_changes"]+=int(prior_idx!=newer_idx)
        if row["role_slots"]:
            per_row.append(row)
    require(len(per_row)>=50 and sum(x["role_slots"] for x in per_row)>=100,
            "external one-shot relation evaluation sufficient sample support")
    summary=bootstrap_paired_rows(per_row)
    for name,b in per_role.items():
        if b["role_slots"]:
            b["S112_base_anchored_top1"]=b["base_correct"]/b["role_slots"]
            b["S115_trained_anchored_top1"]=b["trained_correct"]/b["role_slots"]
            b["delta_percentage_points"]=100*(b["trained_correct"]-b["base_correct"])/b["role_slots"]
            require(b["both_correct"]+b["base_only_correct"]+b["trained_only_correct"]+b["both_wrong"]==b["role_slots"],
                    "all paired role-slot outcomes accounted for")
        else:
            b["no_supervision_UNKNOWN"]=True
    report={
        "stage":STAGE,
        "first_and_ONLY_prospective_SynTagRus_B_firstread_SPENT_AFTER_THIS":True,
        "source_and_checkpoint_shas_verified_BEFORE_B_GOLD":True,
        "actual_S117_original12_SOURCE_SHA256":sha(source12),
        "firstread_B_SOURCE_immutable_SHA":"f687b1f677a3795ba4c4dbf3b5246e1105cf19f236f023e4d9c4d18b5418f9bf",
        "firstread_B_GOLD_immutable_SHA":"fca74d9f7b83cc93374dab774d30dc0d881b62c9a5dcef00a9b113598b4c21d3",
        "S112_frozen_zero_trust_53751_FP16_SHA256":sha(s112_child),
        "S115_frozen_trained_trust_53751_FP16_SHA256":sha(s115_child),
        "ALL_PREVIOUS_53743_in_Calibri_FP16_weights_BIT_IDENTICAL":True,
        "only_8_new_bounded_neural_role_trust_weights_differ":True,
        "original_616_J7_core_GT_frozen_untouched":True,
        "all_other_original12_neural_attributes_equal":unchanged_morph_count==h.shape[0],
        "maximum_abs_native_BPE_role_logit_correction":max_abs_residual,
        "paired_independent_sentence_cluster_bootstrap":summary,
        "per_neural_event_role_firstread_known_predicate":per_role,
        "new_B_ALL_original_source_rows_original_BPE_alignment":audit,
        "NC_LICENSE":"CC BY-NC-SA 4.0",
        "NONCOMMERCIAL_only_NEVER_deploy_without_rightsholder_license":True,
        "NO_dev_or_B_gold_training_or_model_selection_or_repeats":True,
        "S115_same_A_internal_developer_NEVER_SCORED":True,
        "old_S79_S86_S96_S99_S102_S105_S110_TaigaC_spent_NEVER_REUSED":True,
        "teacher_known_predicate_NOT_autonomous_event_recognition":True,
        "not_factuality_QA_or_active_Calibri_616_certification":True,
        "no_allin_bpp_under0_7_or_CPU_over90_certificate":True,
        "candidate_accepted":False,"activation_allowed":False,"output_gate":0,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/REPORT).write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("NRA3_S118_FINAL_BURNED_INDEPENDENT_SYNTAGRUS_B_ONESHOT",
          json.dumps({"paired":summary,"roles":per_role,
                      "old_frozen":53743,"new_trust":8,
                      "max_bounded_abs_logit":max_abs_residual,
                      "NC_only":True,"B_SPENT":True,"gate":0},
                     ensure_ascii=False,sort_keys=True),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    for key in ("source12","source_manifest","s112_child","s112_report",
                "s115_child","separate_b_gold","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    p.add_argument("--expected-source-sha",required=True)
    args=p.parse_args()
    firstread(args.source12,args.source_manifest,args.s112_child,args.s112_report,
              args.s115_child,args.separate_b_gold,args.out,args.expected_source_sha)

if __name__=="__main__":
    main()
