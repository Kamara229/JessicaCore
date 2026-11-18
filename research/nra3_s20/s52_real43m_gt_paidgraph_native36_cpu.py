"""S52 physical CPU paid graph interface with genuine original Calibri43M+GT words.

No new original 43M inference: inputs are S51 output of genuine 12-layer
CPU original43M+GT + 616-J7-protected replay. The model here is the
original S26 physically paid FP16 12,384-parameter *shadow graph* head,
trained previously only on separate S22 original43M synthetic TRAIN256.
No training, no fine-tune, no J7 replay, no evaluation gold.

Source is S51 only: 36 S48 TRAIN-only quote plaintexts, never S48 HELD26
nor S47 manual36, S38 protected32 or independent EVAL. Conservative
event/argument masks do not modify original logits or model weights;
they merely support offline abstention diagnostics for an actual
genuine43M representation and an existing previously trained graph.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from research.nra3_s20.s26_predicted_event_inversion_rescue import (
    EventGroundedInversionGraph,
)
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    S52QuoteAwareFrozenPaidGraphBridge,tensor_sha,prediction_totals,
)

STAGE="NRA3_S52_S26_PAID12384_NATIVE_TRUE43M_GT_S51_QUOTE_BPE_ABSTAIN_FIRST_CPU"
S51_RUN=44267779820
S26_RUN=44196720655
SOURCE_REPORT="NRA3_S51_ORIGINAL_GT_QUOTE36_BPE_FRACTIONAL_CHAR_OVERLAP_AUDIT.json"
NATIVE_FILE="NRA3_S51_DERIVED_REAL43M_GT_WORD339_QUOTE_AND_UNSAFE.safetensors"
S26_REPORT="NRA3_S26_REAL43M_GT_TRAIN256_GROUPED40_MATCHED_DIAGNOSTICS.json"
S26_CHECKPOINT="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"


def require(ok,why):
    if not bool(ok):
        raise ValueError("S52 true original43M+GT paid graph safe mask: "+why)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s51-features",type=Path,required=True)
    p.add_argument("--s51-provenance",type=Path,required=True)
    p.add_argument("--s26-graph",type=Path,required=True)
    p.add_argument("--s26-provenance",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"research result overwrite denied")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(all(f.is_file() for f in (
        a.s51_features,a.s51_provenance,a.s26_graph,a.s26_provenance)),
        "physically downloaded S51 original GT and original S26 paid FP16 missing")
    require(a.s51_features.name==NATIVE_FILE
            and a.s51_provenance.name==SOURCE_REPORT
            and a.s26_graph.name==S26_CHECKPOINT
            and a.s26_provenance.name==S26_REPORT,
            "original source model/feature filename changed")
    s51=json.loads(a.s51_provenance.read_text(encoding="utf-8"))
    require(s51.get("stage")=="NRA3_S51_REAL43M_GT_BPE_MANY_TO_MANY_RAW_OFFSET_AUDIT"
            and s51.get("actual_original43M_12layer_GT_post616J7") is True
            and s51.get("original_616_J7_replayed_before_read_only_quotes") is True
            and s51.get("new_BPE_fractional_aligned_count")==36
            and s51.get("previously_strict_S50_aligned_count")==29
            and s51.get("previously_strict_S50_unaligned_row_ids")==[8,20,21,25,32,34,35]
            and s51.get("new_BPE_fractional_newly_recovered_row_ids")==[8,20,21,25,32,34,35]
            and s51.get("new_BPE_fractional_remaining_failed")==[]
            and s51.get("new_BPE_fractional_unsafe_raw_word_count")==24
            and s51.get("original_CPU_GT_S48_quote_word_count")==72
            and s51.get("derived_REAL43M_WORD339_filename")==a.s51_features.name
            and s51.get("derived_REAL43M_WORD339_SHA256")==sha(a.s51_features)
            and s51.get("gold_targets_not_in_input_or_pooling") is True
            and s51.get("S38_S47_S48_HELD26_EVAL32_EVAL128_not_loaded_or_used") is True
            and s51.get("pooled_overlaps_NOT_causally_safe_for_tokenwise_training") is True
            and s51.get("output_gate")==0
            and not s51.get("candidate_accepted")
            and not s51.get("activation_allowed"),
            "genuine S51 original GT CPU 36-source SHA/unsafe provenance drift")
    src=load_file(str(a.s51_features),device="cpu")
    require(set(src)=={
            "original_real43m_gt_word339_READONLY",
            "original_native_word_valid_mask",
            "native_BPE_causally_unsafe_word_mask",
            "source_row_indices_in_frozen_S48_TRAIN36"},
            "S51 original43M feature archive contains extra gold/unknown values")
    x=src["original_real43m_gt_word339_READONLY"].detach()
    valid=src["original_native_word_valid_mask"].detach()
    unsafe=src["native_BPE_causally_unsafe_word_mask"].detach()
    ids=src["source_row_indices_in_frozen_S48_TRAIN36"]
    require(x.ndim==3 and x.shape[0]==36 and x.shape[-1]==339
            and x.dtype==torch.float32
            and valid.shape==unsafe.shape==x.shape[:2]
            and valid.dtype==unsafe.dtype==torch.bool
            and ids.tolist()==list(range(36))
            and int(unsafe.sum())==24
            and int(x[...,338].sum())==72,
            "S51 actual original GT 36 original text-sidecar native ABI drift")

    s26=json.loads(a.s26_provenance.read_text(encoding="utf-8"))
    require(s26.get("stage")=="NRA3_S26_REAL43M_GT_NEW256_SYNTHETIC_TRAIN_ONLY_GROUPED40_GRAPH"
            and s26.get("physical_FP16_filename")==a.s26_graph.name
            and s26.get("physical_FP16_candidate_SHA256")==sha(a.s26_graph)
            and s26.get("full_physical_FP16_parameters")==12384
            and s26.get("frozen_parent_FP16_parameters")==9332
            and s26.get("trainable_graph_FP16_parameters")==3052
            and s26.get("true_frozen_original43M_GT_from_source_S22b") is True
            and s26.get("no_active_candidate_616_J7_certification") is True
            and s26.get("output_gate")==0
            and not s26.get("candidate_accepted")
            and not s26.get("activation_allowed"),
            "original S26 genuine43M trained paid FP16 model physical SHA/ABI drift")
    paid=a.s26_graph.read_bytes()
    graph=EventGroundedInversionGraph.from_physical(paid)
    require(graph.output_gate==graph.parent.output_gate==0
            and not graph.candidate_accepted and not graph.activation_allowed
            and not graph.parent.candidate_accepted
            and not graph.parent.activation_allowed
            and all(not p.requires_grad for p in graph.parameters()),
            "old frozen S26 model became trainable or active")
    sha_paid_before=hashlib.sha256(graph.physical_bytes()).hexdigest()
    require(sha_paid_before==sha(a.s26_graph),
            "S26 paid original FP16 physical roundtrip changed")
    old_x_sha=tensor_sha(x,valid,unsafe)
    original_old_logits=None
    with torch.inference_mode():
        original_old_logits=graph(x[...,:338],valid)
    bridge=S52QuoteAwareFrozenPaidGraphBridge(graph)
    observed=bridge.observe(x,valid,unsafe)
    heads=("event_status","role","argument_polarity","coreference","event_relation")
    require(all(torch.equal(getattr(original_old_logits,name),
                            getattr(observed.original_paid_graph_logits,name))
                for name in heads),
            "S52 new quote/BPE masks modified original five S26 paid logits")
    require(tensor_sha(x,valid,unsafe)==old_x_sha
            and hashlib.sha256(graph.physical_bytes()).hexdigest()==sha_paid_before
            and bridge.output_gate==0 and bridge.model.output_gate==0
            and bridge.model.parent.output_gate==0,
            "S52 changed original GT features or trained/tampered old S26 graph")
    num_words=int(valid.sum())
    quote_words=int((observed.quote_word_mask&valid).sum())
    unsafe_words=int((unsafe&valid).sum())
    punctuation_words=int((observed.punctuation_mask&valid).sum())
    eligible=int(observed.eligible_word_mask.sum())
    eligible_pairs=int(observed.eligible_directed_pair_mask.sum())
    raw_pairs=int((valid.sum(-1)*(valid.sum(-1)-1)).sum())
    require(num_words>300
            and quote_words==72
            and unsafe_words==24
            and 0<eligible<num_words
            and eligible_pairs>0
            and eligible_pairs<raw_pairs
            and not observed.original_paid_graph_logits.event_status.requires_grad,
            "S52 zero-gate masking did not restrict real original 43M graph")
    outputs=prediction_totals(observed)
    count_by_sample=[]
    for i in range(36):
        v=valid[i]
        q=observed.quote_word_mask[i]
        bad=unsafe[i]
        safe=observed.eligible_word_mask[i]
        pair=observed.eligible_directed_pair_mask[i]
        count_by_sample.append({
            "S48_original_TRAIN36_index":i,
            "original_true43M_word_count":int(v.sum()),
            "raw_quote_content_count":int((q&v).sum()),
            "native_BPE_causally_unsafe_word_count":int((bad&v).sum()),
            "nonpunct_nonquote_nonshared_eligible_word_count":int(safe.sum()),
            "eligible_directed_pair_count":int(pair.sum()),
            "no_gold_or_answer_in_model_or_bridge_forward":True,
        })
    # No training, scoring against human gold, model selection or acceptance.
    result={
        "stage":STAGE,
        "genuine_original_Calibri43M_GT_S51_source_CPU_run":S51_RUN,
        "original_paid_S26_true43M_GT_train256_source_CPU_run":S26_RUN,
        "physical_real_GT_S51_word339_SHA256":sha(a.s51_features),
        "S51_provenance_SHA256":sha(a.s51_provenance),
        "original_S26_paid12384_FP16_SHA256":sha(a.s26_graph),
        "original_S26_provenance_SHA256":sha(a.s26_provenance),
        "physical_true_original_GreenTunnel_layer8_inputs_36":36,
        "original_NATIVE_GT_total_words":num_words,
        "previous_S51_quote_words":quote_words,
        "causal_unsafe_native_BPE_words_excluded":unsafe_words,
        "punctuation_native_words":punctuation_words,
        "eligible_word_candidates_no_gold":eligible,
        "eligible_directed_pair_candidates_no_gold":eligible_pairs,
        "raw_all_directed_nonself_pair_count":raw_pairs,
        "original_trained_S26_five_head_prediction_counts_NOT_accuracy":outputs,
        "S52_36_case_offline_description":count_by_sample,
        "S26_old_original_five_heads_bitwise_equal_masked_observer":True,
        "S26_old_12384_physical_FP16_bytes_identical_pre_post":True,
        "S51_true_original43M_GT_native_hidden_and_masks_identical_pre_post":True,
        "S48_quote_signals_not_passed_into_original_S26_incompatible338D_model":True,
        "original_S26_PARENT_context_uses_both_word_directions_NO_CAUSAL_CERTIFICATION":True,
        "original43M_S93_GT_full_prefill_causal_safety_NOT_proven":True,
        "S52_candidate_masks_are_conservative_abstention_NOT_ONLINE_DECODER":True,
        "S52_masked_edges_NOT_semantically_validated_NO_F1":True,
        "no_S51_36_source_training_hyperparameter_tuning_or_selection":True,
        "S26_original_model_trained_in_prior_RUN_only_not_retrained":True,
        "S38_S47_S48_HELD26_independent_EVAL_not_imported":True,
        "original_616_J7_retention_replayed_IN_PRIOR_S51_not_here":True,
        "active_candidate_616J7_bpp_CB59_NOT_CERTIFIED":True,
        "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    a.out.mkdir(parents=True,exist_ok=False)
    dest=a.out/"NRA3_S52_GENUINE43M_GT_S26_PAIDGRAPH_QUOTE_BPE_ABSTAIN_FIRST_CPU.json"
    dest.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                    encoding="utf-8")
    print("S52_REAL43M_GT_FROZEN_S26_PAID_5HEADS",json.dumps({
        "true_original_GT_words":num_words,
        "quoted":quote_words,"native_unsafe":unsafe_words,
        "punctuation":punctuation_words,"eligible_nodes":eligible,
        "all_nonself_pairs":raw_pairs,"eligible_pairs":eligible_pairs,
        "original_s26_unchanged":True,
        "S51_source_only_not_independent":True,
        "gate":0},sort_keys=True),flush=True)
    print("S52_PREVIOUSLY_TRAINED_S26_ARGMAX_NO_QUALITY_SCORE",
          json.dumps(outputs,sort_keys=True),flush=True)
    print("S52_NO_REAL_MODEL_IMPROVEMENT_OR_ACTIVE_INTEGRATION_CERTIFIED",flush=True)


if __name__=="__main__":
    main()
