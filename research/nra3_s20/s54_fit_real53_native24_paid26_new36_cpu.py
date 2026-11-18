"""S54 fit 36-FP16 residual ONLY on previously scored S53 TRAIN-development24.

Physically archived S53 native 43M+GT real CPU layer8 + original
prior S26 paid graph, no original model weights or Green Tunnel.
S54 HELD24 plaintext AND HELD24 human gold NOT IMPORTED, scored or
examined here. Train gold only from S53 source already scored in S53.

All S26 paid 12,384 parameters frozen / bitwise identical, and original
native339 unchanged. New readout 36 params paid physically via FP16.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s26_predicted_event_inversion_rescue import EventGroundedInversionGraph
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    S52QuoteAwareFrozenPaidGraphBridge,tensor_sha,
)
from research.nra3_s20.s54_trainonly_real43m_recipient_residual import (
    FEATURE_NAMES,PAID_READOUT,
    pair_features_gold_blind,train_on_previously_scored_S53_only,
)

STAGE="NRA3_S54_TRAIN_ONLY_S53_PREVIOUSLY_SCORED24_GENUINE43M_GT_PAID36"
SOURCE_RUN=44274389010
PAID_RUN=44196720655
SOURCE_FILE="NRA3_S53_GENUINE43M_GT_NEGATION24_WORD339_AND_UNSAFE.safetensors"
SOURCE_REPORT="NRA3_S53_GENUINE43M_GT_NEGATION24_OFFSET_PROVENANCE.json"
PAID_FILE="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
PAID_REPORT="NRA3_S26_REAL43M_GT_TRAIN256_GROUPED40_MATCHED_DIAGNOSTICS.json"


def require(ok,why):
    if not bool(ok):raise ValueError("S54 S53-only actual43M train: "+why)

def sha(file:Path):
    return hashlib.sha256(file.read_bytes()).hexdigest()

def word_id(toks,needle):
    matches=[i for i,t in enumerate(toks) if t.form==needle]
    require(len(matches)==1,"S54 previously scored TRAIN gold raw word alignment ambiguous")
    return matches[0]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--source-provenance",type=Path,required=True)
    p.add_argument("--s26",type=Path,required=True)
    p.add_argument("--s26-provenance",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"S54 train-only paid output overwrites prior")
    require(all(f.is_file() for f in (a.source,a.source_provenance,a.s26,a.s26_provenance))
            and a.source.name==SOURCE_FILE
            and a.source_provenance.name==SOURCE_REPORT
            and a.s26.name==PAID_FILE
            and a.s26_provenance.name==PAID_REPORT,
            "S53 real43M source-only native or S26 original paid training physical absent")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_info=json.loads(a.source_provenance.read_text())
    model_info=json.loads(a.s26_provenance.read_text())
    require(source_info["stage"]=="NRA3_S53_GENUINE43M_GT_NEGATION24_BPE_SOURCE_ONLY"
            and source_info["derived_REAL43M_WORD339_SHA256"]==sha(a.source)
            and source_info["original_616_J7_replayed_before_read_only_quotes"]
            and source_info["output_gate"]==0
            and not source_info["candidate_accepted"]
            and model_info["stage"]=="NRA3_S26_REAL43M_GT_NEW256_SYNTHETIC_TRAIN_ONLY_GROUPED40_GRAPH"
            and model_info["physical_FP16_candidate_SHA256"]==sha(a.s26)
            and model_info["full_physical_FP16_parameters"]==12384
            and model_info["output_gate"]==0
            and not model_info["candidate_accepted"],
            "S53 originally physical REAL GT or original paid S26 source SHA/ABI drift")
    data=load_file(str(a.source),device="cpu")
    require(set(data)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S53_NEW24"},
        "S54 train S53 original native file cannot have gold fields")
    native=data["original_real43m_gt_word339_READONLY"].detach().contiguous()
    valid=data["original_native_word_valid_mask"]
    unsafe=data["native_BPE_causally_unsafe_word_mask"]
    rows=data["source_row_indices_in_frozen_S53_NEW24"]
    require(native.shape[0]==24 and native.shape[-1]==339
            and rows.tolist()==list(range(24))
            and valid.shape==unsafe.shape==native.shape[:2]
            and int(unsafe.sum())==0,
            "S53 fixed native 24 feature shape / unsafe changed")
    base_sha=tensor_sha(native,valid,unsafe)
    old_bytes=a.s26.read_bytes()
    old_model=EventGroundedInversionGraph.from_physical(old_bytes)
    require(old_model.output_gate==old_model.parent.output_gate==0
            and all(not z.requires_grad for z in old_model.parameters()),
            "original paid S26 graph cannot train")
    paid_parameters_sha=tensor_sha(*(z.detach() for z in old_model.parameters()))
    with torch.inference_mode():
        old_result=old_model(native[...,:338],valid)
        observed=S52QuoteAwareFrozenPaidGraphBridge(old_model).observe(native,valid,unsafe)
        require(all(torch.equal(getattr(old_result,key),
                                getattr(observed.original_paid_graph_logits,key))
                    for key in ("event_status","role","argument_polarity",
                                "coreference","event_relation")),
                "S26 original paid five-head graph was altered by native quote/BPE safety mask")
        features=pair_features_gold_blind(native,observed)
    require(features.shape[:3]==observed.eligible_directed_pair_mask.shape
            and features.shape[-1]==len(FEATURE_NAMES),
            "S54 genuine original GT/S26 feature ABI changed")
    # S53 has ALREADY been scored (S53 first run 44274389010);
    # it is now explicitly *TRAIN-only* and cannot support independent EVAL.
    from research.nra3_s20.s53_precommitted_negation_quote24_plain import (
        SOURCE_ROWS,text_only_inputs,text_sha256,
    )
    from research.nra3_s20.s53_separate_manual_gold_negation24 import (
        gold24,gold_sha256,
    )
    gold=gold24()
    require(len(SOURCE_ROWS)==len(text_only_inputs())==len(gold)==24
            and len(text_sha256())==len(gold_sha256())==64,
            "S53 previously scored manual TRAIN source no longer frozen")
    targets=torch.zeros_like(observed.eligible_directed_pair_mask)
    rejected=torch.zeros_like(targets)
    for i,(row,g) in enumerate(zip(SOURCE_ROWS,gold)):
        require(row[0]==g.row_id,"S53 TRAIN source/gold row order drift")
        toks=raw_surface_grammar(row[2])
        require(len(toks)==int(valid[i].sum()),"genuine original GT raw word count drift")
        if g.recipient is None:continue
        source_word=word_id(toks,g.predicate)
        destination_word=word_id(toks,g.recipient)
        require(bool(observed.eligible_directed_pair_mask[i,source_word,destination_word]),
                "S54 S53 TRAIN human true link was excluded by raw quote/BPE safety gate")
        targets[i,source_word,destination_word]=True
        if g.recipient_polarity=="REJECTED":
            rejected[i,source_word,destination_word]=True
    require(int(targets.sum())==18 and int(rejected.sum())==12,
            "S54 previously scored S53 actual original GT gold counts differ")
    model,physical,diagnostics=train_on_previously_scored_S53_only(
        features,observed.eligible_directed_pair_mask,targets,rejected)
    require(physical.shape==(2,len(FEATURE_NAMES)) and physical.dtype==torch.float16
            and physical.numel()==PAID_READOUT==36
            and bool(torch.isfinite(physical).all())
            and not model.candidate_accepted
            and not model.activation_allowed
            and model.output_gate==0
            and tensor_sha(native,valid,unsafe)==base_sha
            and tensor_sha(*(z.detach() for z in old_model.parameters()))==paid_parameters_sha,
            "original43M GT source or original S26 weights mutated by NEW S54 readout training")
    a.out.mkdir(parents=True,exist_ok=False)
    fp=a.out/"NRA3_S54_S53_TRAINONLY_36FP16_DIRECTED_RECIPIENT_REJECTED.safetensors"
    save_file({"S54_new36_FP16_readout":physical},str(fp))
    require(torch.equal(load_file(str(fp),device="cpu")["S54_new36_FP16_readout"],physical),
            "S54 physically paid new 36 FP16 train-only weights failed reload")
    report={
      "stage":STAGE,
      "source_real_original43M_GT_S53_616J7_run":SOURCE_RUN,
      "old_previously_trained_S26_12384_FP16_run":PAID_RUN,
      "source_real_original43M_GT_S53_native339_SHA256":sha(a.source),
      "old_S26_original_paid_FP16_SHA256":sha(a.s26),
      "S53_previously_scored_TRAIN_only_raw_source_sha256":text_sha256(),
      "S53_previously_scored_TRAIN_only_manual_gold_sha256":gold_sha256(),
      "fixed_36_FP16_FEATURE_NAMES":FEATURE_NAMES,
      "newly_paid_trainable_FP16_params":PAID_READOUT,
      "physically_saved_36_FP16_readout_file":fp.name,
      "physically_saved_36_FP16_readout_SHA256":sha(fp),
      "S53_TRAIN_fit_diagnostics_NOT_independent_EVAL":diagnostics,
      "old_original_S26_five_head_logits_bitwise_unchanged":True,
      "old_original_S26_paid_param_tensors_SHA256_same_before_after":True,
      "original_12layer_Calibri43M_GT_source_native_SHA256_same_before_after":True,
      "no_S54_HELD24_text_or_gold_imported_in_this_process":True,
      "S54_readout_only_training_never_original43M_GT_or_S26":True,
      "new_candidate_ACTIVE_616J7_bpp_CB59_NOT_certified":True,
      "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    p=a.out/"NRA3_S54_S53_TRAINONLY_NEW36_FP16_FIRST_TRAIN_REPORT.json"
    p.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                 encoding="utf-8")
    print("S54_FITTED_ONLY_S53_REAL43M_GT_TRAIN24",json.dumps({
       "new_paid":PAID_READOUT,"train_expected_recipient":int(targets.sum()),
       "train_expected_rejected":int(rejected.sum()),
       "train_pred_link":diagnostics["resubstitution_pred_RECIPIENT"],
       "train_pred_rejected":diagnostics["resubstitution_pred_REJECTED"],
       "saved_FP16_SHA256":sha(fp),"output_gate":0,
    },sort_keys=True),flush=True)
    print("S54_NO_S54_HELD_GOLD_OR_TEXT_SEEN_DURING_TRAIN_ONLY",flush=True)


if __name__=="__main__":
    main()
