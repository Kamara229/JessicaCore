"""S56 FIRST and ONLY structurally diverse NEW HELD24 true43M+GT evaluation.

Original Calibri43M/Green Tunnel 12-layer CPU gold-blind replays
protected 616 J7 and captures first HELD24 native features separately.
Previously paid original S26, prior 36-FP16 S54, newly trained 108-FP16
S56 are ALL FROZEN BEFORE any S56 HELD text/model evaluation.

All 3 semantic model predictions are completed before importing the
separately precommitted S56 manual HELD gold. No training/threshold
selection here. Author-generated 24-case pilot, not general Russian
evaluation. S38/S47/S48 HELD26, independent EVAL all untouched.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors.torch import load_file
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s19.s19f_real43m_train9_graph_targets import STATUS,ROLE
from research.nra3_s20.s26_predicted_event_inversion_rescue import EventGroundedInversionGraph
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    S52QuoteAwareFrozenPaidGraphBridge,tensor_sha,
)
from research.nra3_s20.s54_trainonly_real43m_recipient_residual import (
    S54DirectedRecipientResidual,pair_features_gold_blind,
)
from research.nra3_s20.s56_grammar_clause_scope_semantic_readout import (
    S56HierarchicalClauseRecipient,real43m_semantic_structure_features,FEATURE_NAMES,
)
from research.nra3_s20.s57_frozen_new_held24_plain import (
    SOURCE_ROWS,FAMILIES,text_only_inputs,text_sha256,
)
from research.nra3_s20.s57_event_first_native_semantic_decoder import (
    S57EventFirstDecoder,create_event_first_features,TRAINABLE_PARAMS,
)


STAGE="NRA3_S57_FIRST_EVENT_FIRST_HELD24_GENUINE43M_GT_S26_S54_S56_S57"
NATIVE="NRA3_S57_GENUINE43M_GT_FIRST_HELD24_WORD339_AND_UNSAFE.safetensors"
PROVENANCE="NRA3_S57_GENUINE43M_GT_FIRST_HELD24_OFFSET_PROVENANCE.json"
CAPTURE="NRA3_S57_GENUINE43M_GT_FIRST_HELD24_NATIVE_J7_REPORT.json"
OLD_S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
OLD_S54="NRA3_S54_S53_TRAINONLY_36FP16_DIRECTED_RECIPIENT_REJECTED.safetensors"
NEW_S56="NRA3_S56_SPENT72_ONLY_108FP16_SCOPE_RECIPIENT_REJECTED.safetensors"
NEW_S57="NRA3_S57_SPENT96_TRAINONLY_EVENT_FIRST_FP16.safetensors"
TRAIN_S57="NRA3_S57_SPENT96_EVENT_FIRST_TRAIN_ONLY_REPORT.json"
TRAIN_S56="NRA3_S56_SPENT72_SCOPE_RECIPIENT_108FP16_TRAIN_REPORT.json"

def require(ok,why):
    if not bool(ok):
        raise ValueError("S56 NEW HELD24 original43M scope first read: "+why)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def count(pred,truth):
    tp=int((pred&truth).sum())
    fp=int((pred&~truth).sum())
    fn=int((~pred&truth).sum())
    return {"TP":tp,"FP":fp,"FN":fn,
            "precision":tp/(tp+fp) if tp+fp else 0.,
            "recall":tp/(tp+fn) if tp+fn else 0.,
            "F1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}

def word_id(words,term):
    found=[i for i,w in enumerate(words) if w.form==term]
    require(len(found)==1,"frozen S56 manual predicate/recipient word mismatch")
    return found[0]

def main():
    p=argparse.ArgumentParser()
    for field in ("native","native-provenance","native-capture","s26",
                  "s54","s54-manifest","s56","s56-manifest","s57","s57-manifest","out"):
        p.add_argument("--"+field,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"S56 held audit cannot overwrite already seen results")
    require(all(t.is_file() for t in (
        a.native,a.native_provenance,a.native_capture,a.s26,
        a.s54,a.s54_manifest,a.s56,a.s56_manifest,a.s57,a.s57_manifest)),
        "real Calibri43M GT new held or trained S54/S56 FP16 source not physically present")
    require((a.native.name,a.native_provenance.name,a.native_capture.name,
             a.s26.name,a.s54.name,a.s56.name,a.s56_manifest.name,a.s57.name,a.s57_manifest.name)
            ==(NATIVE,PROVENANCE,CAPTURE,OLD_S26,OLD_S54,NEW_S56,TRAIN_S56,NEW_S57,TRAIN_S57),
            "only pin exact physical frozen old and new real43M semantic archives")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    prov=json.loads(a.native_provenance.read_text(encoding="utf-8"))
    cap=json.loads(a.native_capture.read_text(encoding="utf-8"))
    fit=json.loads(a.s56_manifest.read_text(encoding="utf-8"))
    previous_fit=json.loads(a.s54_manifest.read_text(encoding="utf-8"))
    latest_fit=json.loads(a.s57_manifest.read_text(encoding="utf-8"))
    require(latest_fit["stage"]=="NRA3_S57_EVENT_FIRST_OLD_SPENT96_GENUINE43M_GT_TRAIN_ONLY"
            and latest_fit["S57_new_FP16_file_SHA256"]==sha(a.s57)
            and latest_fit["S57_old96_only_TRAIN_not_independent"]["old_spent96_gold_source_events"]==60
            and latest_fit["S57_old96_only_TRAIN_not_independent"]["old_spent96_gold_REJECTED_edges"]==36
            and latest_fit["S57_NEW_HELD24_source_and_gold_NOT_IMPORTED_in_training"]
            and latest_fit["output_gate"]==0
            and not latest_fit["candidate_accepted"],
            "S57 event-first paid readout must be frozen before HELD24 first read")

    require(prov["stage"]=="NRA3_S57_GENUINE43M_GT_FIRST_HELD24_BPE_SOURCE_ONLY"
            and prov["frozen_S57_HELD24_raw_source_sha256"]==text_sha256()
            and prov["derived_REAL43M_WORD339_SHA256"]==sha(a.native)
            and prov["new_BPE_fractional_aligned_count"]==24
            and prov["original_616_J7_replayed_before_read_only_quotes"]
            and prov["S57_HUMAN_GOLD_NOT_IMPORTED_DURING_ANY_CAPTURE_STEP"]
            and prov["output_gate"]==0
            and not prov["candidate_accepted"],
            "true original GT postlayer8 24 raw HELD native provenance wrong")
    require(cap["stage"]=="NRA3_S57_GENUINE43M_GT_FIRST_HELD24_NATIVE_CPU_FIRST_PASS"
            and cap["all_616_native_CPU_generated_token_digests_unchanged"]
            and cap["original_456_full_EOS"]==456
            and cap["original_new128_full_EOS"]==128
            and cap["original_new32_full_EOS"]==0
            and cap["actual_genuine_43M_CPU_new_HELD24_prefills"]==24
            and cap["S57_FROZEN_HELD24_PLAINTEXT_SHA256"]==text_sha256()
            and cap["S57_HELD24_SOURCE_COMMIT"]=="2ef0bc0287c1d4fa650c50f793356da46611bcb7"
            and cap["S57_HELD24_SOURCE_BLOB"]=="6245761c58be31f6a11f99aec89a214b87be1ed0"
            and cap["S57_HELD_manual_gold_NOT_IMPORTED_into_model_capture"]
            and cap["NRA3_output_gate"]==0
            and not cap["NRA3_candidate_accepted"],
            "S56 gold-blind original 43M/GT CPU 616J7 source drift")
    require(fit["stage"]=="NRA3_S56_SPENT_S53_S54_S55_TRAIN72_TRUE43M_GT_SCOPE_108FP16"
            and fit["new_108_FP16_standalone_weights_SHA256"]==sha(a.s56)
            and fit["sum_training_gold_RECIPIENT"]==48
            and fit["sum_training_gold_REJECTED"]==30
            and fit["fixed_optimizer_and_training_CONSUMED_SPENT72_ONLY"]["S56_HELD24_never_loaded_in_train"]
            and fit["S56_fresh_new_scope_HELD24_text_and_gold_NOT_IMPORTED"]
            and fit["output_gate"]==0 and not fit["candidate_accepted"],
            "fresh HELD24 could have entered new model training — prohibited")
    require(previous_fit["stage"]=="NRA3_S54_TRAIN_ONLY_S53_PREVIOUSLY_SCORED24_GENUINE43M_GT_PAID36"
            and previous_fit["physically_saved_36_FP16_readout_SHA256"]==sha(a.s54)
            and previous_fit["output_gate"]==0
            and not previous_fit["candidate_accepted"],
            "S54 old old FP16 weights changed, cannot compare")
    native=load_file(str(a.native),device="cpu")
    require(set(native)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S57_HELD24"},
        "original GT raw native HELD24 contains unknown or labeled fields")
    x=native["original_real43m_gt_word339_READONLY"].detach().clone()
    valid=native["original_native_word_valid_mask"].detach().clone()
    unsafe=native["native_BPE_causally_unsafe_word_mask"].detach().clone()
    require(x.ndim==3 and x.shape[0]==24 and x.shape[-1]==339
            and valid.shape==unsafe.shape==x.shape[:2]
            and native["source_row_indices_in_frozen_S57_HELD24"].tolist()==list(range(24))
            and valid.dtype==unsafe.dtype==torch.bool,
            "original 43M GT S56 HELD24 source offset/mask contract broken")
    before=tensor_sha(x,valid,unsafe)
    old=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    old_before=tensor_sha(*(p.detach() for p in old.parameters()))
    old36=S54DirectedRecipientResidual.from_physical_fp16(
        load_file(str(a.s54),device="cpu")["S54_new36_FP16_readout"])
    new108=S56HierarchicalClauseRecipient.from_physical_fp16(
        load_file(str(a.s56),device="cpu")["S56_SCOPE_PAIR108_FP16"])
    new57=S57EventFirstDecoder.from_physical_fp16(
        load_file(str(a.s57),device="cpu")["S57_EVENT_FIRST_FP16"])
    require(all(not p.requires_grad for model in (old,old36,new108,new57)
                for p in model.parameters())
            and old.output_gate==old.parent.output_gate==0
            and old36.output_gate==new108.output_gate==new57.output_gate==0,
            "first S56 HELD candidate not frozen or original gate activated")
    before36=tensor_sha(*(p.detach() for p in old36.parameters()))
    before108=tensor_sha(*(p.detach() for p in new108.parameters()))
    before57=tensor_sha(*(p.detach() for p in new57.parameters()))
    with torch.inference_mode():
        full_old=old(x[...,:338],valid)
        observed=S52QuoteAwareFrozenPaidGraphBridge(old).observe(x,valid,unsafe)
        require(all(torch.equal(getattr(full_old,k),
                                getattr(observed.original_paid_graph_logits,k))
                    for k in ("event_status","role","argument_polarity",
                              "coreference","event_relation")),
                "S56 original S26 five-head model changed after structural mask")
        mask=observed.eligible_directed_pair_mask
        features36=pair_features_gold_blind(x,observed)
        features108=real43m_semantic_structure_features(
            x,observed,tuple(text_only_inputs()))
        p36,rej36=old36.predict(features36,mask)
        p108,rej108=new108.predict(features108,mask)
        node57,pair57,mask57=create_event_first_features(
            x,observed,tuple(text_only_inputs()),new108)
        p57,rej57,predicted_src57=new57.predict(
            node57,pair57,observed.eligible_word_mask,mask57)
        require(int(p57.sum())<=24 and int(rej57.sum())<=int(p57.sum())
                and not bool((p57&~mask57).any()),
                "S57 event-first hard one-edge constraint violated")

        old_role=full_old.role.argmax(-1)==ROLE.index("RECIPIENT")
        old_pol=full_old.argument_polarity.argmax(-1)==2
        all_pairs=valid.unsqueeze(1)&valid.unsqueeze(2)
        all_pairs &= ~torch.eye(valid.shape[1],dtype=torch.bool).unsqueeze(0)
        r26=(old_role&all_pairs).clone()
        neg26=(old_role&old_pol&all_pairs).clone()
        p26_masked=(r26&mask).clone()
        neg26_masked=(neg26&mask).clone()
        neg_event=(full_old.event_status.argmax(-1)==STATUS.index("NEGATED"))&valid
    # All model forward evaluations completed. Only NOW import manually
    # authored new S56 HELD labels. No optimizer or forwards after this.
    frozen={
      "original_native_43M_SHA":tensor_sha(x,valid,unsafe)==before,
      "original_paid_S26_SHA":tensor_sha(*(p.detach() for p in old.parameters()))==old_before,
      "previous_S54_36_FP16_SHA":tensor_sha(*(p.detach() for p in old36.parameters()))==before36,
      "S56_new108_FP16_SHA":tensor_sha(*(p.detach() for p in new108.parameters()))==before108,
      "S57_new_event_first_FP16_SHA":tensor_sha(*(p.detach() for p in new57.parameters()))==before57,
      "gates_zero":old.output_gate==old36.output_gate==new108.output_gate==new57.output_gate==0,
    }
    require(all(frozen.values()),"S56 changed original GT / pretrained heads / new candidate")
    from research.nra3_s20.s57_frozen_new_held24_gold import (
        held_gold24,gold_sha256,
    )
    gold=held_gold24()
    require(len(gold)==len(SOURCE_ROWS)==24
            and len(text_only_inputs())==24
            and len(gold_sha256())==64,
            "S56 precommitted manual HELD24 gold missing")
    recipient_gold=torch.zeros_like(r26)
    rejected_gold=torch.zeros_like(r26)
    neg_event_gold=torch.zeros_like(valid)
    per_row=[]
    for i,(row,g) in enumerate(zip(SOURCE_ROWS,gold)):
        require(row[0]==g.row_id,"S56 source/gold row order no longer fixed")
        words=raw_surface_grammar(row[2])
        require(len(words)==int(valid[i].sum()),
                "original43M GT held words mismatch raw Unicode source")
        src=word_id(words,g.main_predicate)
        if g.outer_factuality=="NEGATED": neg_event_gold[i,src]=True
        target=word_id(words,g.outer_recipient) if g.outer_recipient is not None else None
        if target is not None:
            recipient_gold[i,src,target]=True
            if g.directed_recipient_polarity=="REJECTED":
                rejected_gold[i,src,target]=True
        per_row.append({
          "case":g.row_id,"family":row[1],
          "text_original_SEALED":row[2],
          "main_predicate_gold":g.main_predicate,
          "recipient_gold":g.outer_recipient,
          "directed_polarity_gold":g.directed_recipient_polarity,
          "directed_recipient_S26":bool(r26[i,src,target]) if target is not None else None,
          "directed_recipient_S54":bool(p36[i,src,target]) if target is not None else None,
          "directed_recipient_S56":bool(p108[i,src,target]) if target is not None else None,
          "directed_recipient_S57":bool(p57[i,src,target]) if target is not None else None,
          "directed_REJECTED_S54":bool(rej36[i,src,target]) if target is not None else None,
          "directed_REJECTED_S56":bool(rej108[i,src,target]) if target is not None else None,
          "directed_REJECTED_S57":bool(rej57[i,src,target]) if target is not None else None,
          "S57_any_transfer_event_predicted":bool(predicted_src57[i].any()),
          "native_pair_eligible":bool(mask[i,src,target]) if target is not None else None,
          "human_gold_loaded_only_after_all_model_forward":True,
        })
    require(int(recipient_gold.sum())==12 and int(rejected_gold.sum())==6
            and int(neg_event_gold.sum())==12,
            "S56 gold totals not matching originally committed 12/6/12 source")
    totals={
      "S26_RECIPIENT_unmasked":count(r26,recipient_gold),
      "S26_RECIPIENT_safety":count(p26_masked,recipient_gold),
      "S54_RECIPIENT_safety":count(p36,recipient_gold),
      "S56_RECIPIENT_safety":count(p108,recipient_gold),
      "S57_EVENT_FIRST_RECIPIENT_safety":count(p57,recipient_gold),
      "S26_REJECTED_unmasked":count(neg26,rejected_gold),
      "S26_REJECTED_safety":count(neg26_masked,rejected_gold),
      "S54_REJECTED_safety":count(rej36,rejected_gold),
      "S56_REJECTED_safety":count(rej108,rejected_gold),
      "S57_EVENT_FIRST_REJECTED_safety":count(rej57,rejected_gold),
      "original_S26_NEGATED_node":count(neg_event,neg_event_gold),
    }
    family={}
    for tag in FAMILIES:
        indices=torch.tensor([i for i,t in enumerate(SOURCE_ROWS) if t[1]==tag])
        require(len(indices)==6,"S56 each originally frozen challenge family must have 6 items")
        rg=recipient_gold.index_select(0,indices)
        ng=rejected_gold.index_select(0,indices)
        family[tag]={
            "gold_recipient":int(rg.sum()),
            "gold_rejected":int(ng.sum()),
            "S54_recipient":count(p36.index_select(0,indices),rg),
            "S56_recipient":count(p108.index_select(0,indices),rg),
            "S57_recipient":count(p57.index_select(0,indices),rg),
            "S54_rejected":count(rej36.index_select(0,indices),ng),
            "S56_rejected":count(rej108.index_select(0,indices),ng),
            "S57_rejected":count(rej57.index_select(0,indices),ng),
            "old_S26_rejected":count(neg26_masked.index_select(0,indices),ng),
            "BPE_unsafe_words":int(unsafe.index_select(0,indices).sum()),
        }
    report={
      "stage":STAGE,
      "new_HELD24_precommitted_RAW_commit":"2ef0bc0287c1d4fa650c50f793356da46611bcb7",
      "new_HELD24_precommitted_RAW_blob":"6245761c58be31f6a11f99aec89a214b87be1ed0",
      "new_HELD24_precommitted_GOLD_commit":"2d132fea53c3594ca4a03d59d1b57c15510d81db",
      "new_HELD24_precommitted_GOLD_blob":"ebf3f66cf22d795e293e3e618106179936f4436f",
      "new_HELD24_raw_text_sha256":text_sha256(),
      "new_HELD24_manual_gold_sha256":gold_sha256(),
      "real_original_Calibri43M_GT_new_HELD24_native_SHA256":sha(a.native),
      "new_108_FP16_fitted_S53_S54_S55_only_SHA256":sha(a.s56),
      "new_S57_EVENT_FIRST_FP16_fitted_S53_S54_S55_S56_only_SHA256":sha(a.s57),
      "original_old_S54_36_FP16_SHA256":sha(a.s54),
      "original_S26_12384_FP16_SHA256":sha(a.s26),
      "all_frozen_model_and_original_true_GT_source_SHA_same_before_after":frozen,
      "original_616J7_protected_replayed_at_CPU_capture_before_HELD_texts":True,
      "new_HELD24_author_written_STRUCTURAL_shift_not_random_realworld":True,
      "new_HELD24_human_gold_loaded_ONLY_AFTER_three_model_predictions":True,
      "new_HELD24_human_gold_loaded_ONLY_AFTER_FOUR_model_predictions":True,
      "S57_max_one_directed_transfer_recipient_per_utterance":int(p57.sum())<=24,
      "S57_0_new_43M_12layer_GT_S26_S54_S56_weight_updates":True,
      "new_HELD24_never_used_to_fit_or_select_hyperparameter_threshold":True,
      "genuine_original_GreenTunnel_43M_12layer_full_context_not_streaming_causal":True,
      "new_108FP16_is_offline_research_sidecar_only_not_active_model":True,
      "new_S57_event_first_is_offline_noncausal_not_active_model":True,
      "first_new_HELD24_gold_RECIPIENT_total":int(recipient_gold.sum()),
      "first_new_HELD24_gold_REJECTED_total":int(rejected_gold.sum()),
      "first_new_HELD24_gold_NEGATED_total":int(neg_event_gold.sum()),
      "first_HELD24_model_counts_not_population_unbiased_EVAL":totals,
      "first_HELD24_by_predeclared_family":family,
      "first_HELD24_all24_manual_gold_and_frozen_predictions":per_row,
      "no_S38_S47_S48_HELD26_independent_EVAL32_128_loaded":True,
      "active_candidate_616J7_bpp_CB59_NOT_verified":True,
      "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
      "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    a.out.mkdir(parents=True,exist_ok=False)
    dst=a.out/"NRA3_S57_TRUE43M_GT_EVENT_FIRST_NEW_HELD24_MANUAL_REPORT.json"
    dst.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                   encoding="utf-8")
    print("S57_TRUE43M_GT_NEW_HELD24_FROZEN_S26_S54_S56_EVENT_FIRST",
          json.dumps(totals,sort_keys=True),flush=True)
    print("S57_TRUE43M_GT_NEW_HELD24_EVENT_FIRST_FAMILY_BREAKDOWN",
          json.dumps(family,sort_keys=True),flush=True)
    print("S57_NO_FIRST_HELD_LEAK_NO_LIVE_ACTIVATION_OR_CANDIDATE_APPROVAL",flush=True)


if __name__=="__main__":
    main()
