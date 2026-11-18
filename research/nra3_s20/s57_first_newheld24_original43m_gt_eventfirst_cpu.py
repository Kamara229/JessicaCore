"""S57 first held24 original Calibri43M+GT event-first model comparison.

All original model and S26/S54/S56/S57 frozen forward completed
BEFORE first import of separately precommitted manual S57 HELD gold.
S57 event→transfer→one-recipient→rejection structure caps 1 directed
edge per source text. Do NOT select threshold, retrain or activate
based on this synthetic first-read HELD24. It is NOT general EVAL.
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
    S56HierarchicalClauseRecipient,real43m_semantic_structure_features,
)
from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import (
    S57EventFirstRecipient,gold_blind_eventfirst_features,N_PAID,
)
from research.nra3_s20.s57_frozen_eventfirst_held24_plain import (
    SOURCE_ROWS,FAMILIES,text_only_inputs,text_sha256,
)

STAGE="NRA3_S57_FIRST_TRUE43M_GT_HELD24_EVENTFIRST_SPENT96_231FP16"
NATIVE="NRA3_S57_GENUINE43M_GT_EVENTFIRST24_WORD339_AND_UNSAFE.safetensors"
PROV="NRA3_S57_GENUINE43M_GT_EVENTFIRST24_OFFSET_PROVENANCE.json"
CAP="NRA3_S57_GENUINE43M_GT_EVENTFIRST24_NATIVE_J7_REPORT.json"
OLD26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
OLD54="NRA3_S54_S53_TRAINONLY_36FP16_DIRECTED_RECIPIENT_REJECTED.safetensors"
OLD56="NRA3_S56_SPENT72_ONLY_108FP16_SCOPE_RECIPIENT_REJECTED.safetensors"
NEW57="NRA3_S57_EVENTFIRST_SPENT96_ONLY_231FP16_UNACCEPTED.safetensors"
TRAIN57="NRA3_S57_EVENTFIRST_SPENT96_231FP16_TRAIN_REPORT.json"

def require(ok,why):
    if not bool(ok):raise ValueError("S57 original true43M+GT NEW24 first held: "+why)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def counts(pred,gold):
    tp=int((pred&gold).sum())
    fp=int((pred&~gold).sum())
    fn=int((~pred&gold).sum())
    d=2*tp+fp+fn
    return {"TP":tp,"FP":fp,"FN":fn,
            "precision":tp/(tp+fp) if tp+fp else 0.,
            "recall":tp/(tp+fn) if tp+fn else 0.,
            "F1":2*tp/d if d else 0.}

def only_word(words,token):
    match=[i for i,w in enumerate(words) if w.form==token]
    require(len(match)==1,"manual gold token must match exactly one original Unicode word")
    return match[0]


def main():
    parser=argparse.ArgumentParser()
    for name in ("native","native-provenance","native-capture","s26","s54",
                 "s56","s57","s57-manifest","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args()
    require(not a.out.exists(),"first S57 HELD24 report must never be overwritten")
    require(all(z.is_file() for z in (
        a.native,a.native_provenance,a.native_capture,a.s26,a.s54,a.s56,
        a.s57,a.s57_manifest)),
        "physical genuine original43M+GT native24 or old/new paid models missing")
    require((a.native.name,a.native_provenance.name,a.native_capture.name,
             a.s26.name,a.s54.name,a.s56.name,a.s57.name,a.s57_manifest.name)
            ==(NATIVE,PROV,CAP,OLD26,OLD54,OLD56,NEW57,TRAIN57),
            "S57 original frozen model archive filenames or source changed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    provenance=json.loads(a.native_provenance.read_text())
    cap=json.loads(a.native_capture.read_text())
    fit=json.loads(a.s57_manifest.read_text())
    require(provenance["stage"]=="NRA3_S57_GENUINE43M_GT_EVENTFIRST24_BPE_SOURCE_ONLY"
            and provenance["frozen_S57_HELD24_raw_source_sha256"]==text_sha256()
            and provenance["derived_REAL43M_WORD339_SHA256"]==sha(a.native)
            and provenance["new_BPE_fractional_aligned_count"]==24
            and provenance["original_616_J7_replayed_before_read_only_quotes"]
            and provenance["S57_HUMAN_GOLD_NOT_IMPORTED_DURING_ANY_CAPTURE_STEP"]
            and provenance["output_gate"]==0 and not provenance["candidate_accepted"],
            "S57 original true43M+GT first HELD24 gold-blind features not provenance-locked")
    require(cap["stage"]=="NRA3_S57_GENUINE43M_GT_EVENTFIRST24_NATIVE_CPU_FIRST_PASS"
            and cap["all_616_native_CPU_generated_token_digests_unchanged"]
            and cap["original_456_full_EOS"]==456
            and cap["original_new128_full_EOS"]==128
            and cap["original_new32_full_EOS"]==0
            and cap["actual_genuine_43M_CPU_new_HELD24_prefills"]==24
            and cap["S57_FROZEN_HELD24_PLAINTEXT_SHA256"]==text_sha256()
            and cap["S57_HELD24_SOURCE_COMMIT"]=="0c8bf5a93e954350b2cc35f718d2d7c185019f10"
            and cap["S57_HELD24_SOURCE_BLOB"]=="daf7dd4af8f66ab54ff44972bf7da144bbd4aafe"
            and cap["S57_HELD_manual_gold_NOT_IMPORTED_into_model_capture"]
            and cap["NRA3_output_gate"]==0 and not cap["NRA3_candidate_accepted"],
            "S57 original real CPU 616 J7 replay or raw-source lock changed")
    require(fit["stage"]=="NRA3_S57_OLD_SPENT_S53_S54_S55_S56_TRAIN96_REAL43M_GT_EVENTFIRST_231FP16"
            and fit["new_event_first_FP16_weights_SHA256"]==sha(a.s57)
            and fit["new_event_first_FP16_model_parameters"]==N_PAID==231
            and fit["old_spent_TRAIN96_fixed_optimization_only"]["old_true_directed_recipient_edges"]==60
            and fit["old_spent_TRAIN96_fixed_optimization_only"]["old_true_directed_rejected_edges"]==36
            and fit["new_S57_first_HELD24_raw_and_manual_gold_NEVER_IMPORTED"]
            and fit["output_gate"]==0 and not fit["candidate_accepted"],
            "S57 new 231 paid weights are not frozen old96-only BEFORE fresh HELD")
    native=load_file(str(a.native),device="cpu")
    require(set(native)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S57_HELD24"},
        "S57 real original GT native archive includes unknown source labels")
    x=native["original_real43m_gt_word339_READONLY"].detach().clone()
    valid=native["original_native_word_valid_mask"].detach().clone()
    unsafe=native["native_BPE_causally_unsafe_word_mask"].detach().clone()
    require(x.ndim==3 and x.shape[0]==24 and x.shape[-1]==339
            and valid.shape==unsafe.shape==x.shape[:2]
            and valid.dtype==unsafe.dtype==torch.bool
            and native["source_row_indices_in_frozen_S57_HELD24"].tolist()==list(range(24)),
            "true frozen original 43M+GT HELD24 native339/unsafe ABI wrong")
    before=tensor_sha(x,valid,unsafe)
    old26=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    old54=S54DirectedRecipientResidual.from_physical_fp16(
        load_file(str(a.s54),device="cpu")["S54_new36_FP16_readout"])
    old56=S56HierarchicalClauseRecipient.from_physical_fp16(
        load_file(str(a.s56),device="cpu")["S56_SCOPE_PAIR108_FP16"])
    new57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(a.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    models=(old26,old54,old56,new57)
    model_shas=[tensor_sha(*(p.detach() for p in m.parameters())) for m in models]
    require(all(not p.requires_grad for m in models for p in m.parameters())
            and old26.parent.output_gate==old26.output_gate==0
            and all(m.output_gate==0 and not m.candidate_accepted for m in models),
            "S57 frozen S26/S54/S56/S57 models unexpectedly trainable/active")
    with torch.inference_mode():
        raw_old=old26(x[...,:338],valid)
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
        require(all(torch.equal(getattr(raw_old,key),
                                getattr(obs.original_paid_graph_logits,key))
                    for key in ("event_status","role","argument_polarity",
                                "coreference","event_relation")),
                "S57 old S26 original logits changed under safety mask")
        pair_mask=obs.eligible_directed_pair_mask
        x54=pair_features_gold_blind(x,obs)
        x56=real43m_semantic_structure_features(x,obs,text_only_inputs())
        x57=gold_blind_eventfirst_features(x,obs,text_only_inputs())
        old54_rec,old54_rej=old54.predict(x54,pair_mask)
        old56_rec,old56_rej=old56.predict(x56,pair_mask)
        result57=new57.decode(x57)
        new_rec=result57["directed_recipient"]
        new_rej=result57["directed_rejected"]
        chosen_event=result57["chosen_outer_event_index"]
        old26_rec=(raw_old.role.argmax(-1)==ROLE.index("RECIPIENT"))&pair_mask
        old26_rej=old26_rec&(raw_old.argument_polarity.argmax(-1)==2)
    frozen={
      "original_true43M_GT_native339_unchanged":tensor_sha(x,valid,unsafe)==before,
      "original_old_S26_S54_S56_new_S57_paid_FP16_parameters_unchanged":all(
         tensor_sha(*(p.detach() for p in m.parameters()))==s
         for m,s in zip(models,model_shas)),
      "all_research_gates_zero":all(m.output_gate==0 for m in models),
    }
    require(all(frozen.values()),"S57 original 43M GT, old paid, or new 231 model mutated")
    require(int(new_rec.sum())<=24 and int(new_rej.sum())<=int(new_rec.sum())
            and not bool((new_rec&~pair_mask).any()),
            "S57 staged event-first failed at-most-one directed link safety invariant")
    # ONLY NOW, after 43M/S26/S54/S56/S57 prediction complete, import
    # new manual HELD24 gold. Never use for any model training/selection.
    from research.nra3_s20.s57_frozen_eventfirst_held24_manual_gold import (
        held_gold24,gold_sha256,
    )
    gold=held_gold24()
    require(len(gold)==len(SOURCE_ROWS)==len(text_only_inputs())==24,
            "S57 NEW manual gold source row order changed")
    edge_truth=torch.zeros_like(new_rec)
    reject_truth=torch.zeros_like(new_rej)
    event_truth=torch.zeros_like(chosen_event)
    manual_neg_event=torch.zeros_like(valid)
    row_out=[]
    for i,(r,g) in enumerate(zip(SOURCE_ROWS,gold)):
        require(r[0]==g.row_id,"S57 new HELD manual labels reordered")
        toks=raw_surface_grammar(r[2])
        require(len(toks)==int(valid[i].sum()),
                "S57 genuine original GT word alignment and gold lexical text mismatch")
        src=only_word(toks,g.main_predicate)
        event_truth[i]=src
        if g.event_factuality=="NEGATED":manual_neg_event[i,src]=True
        dst=only_word(toks,g.transferred_recipient) if g.transferred_recipient is not None else None
        if dst is not None:
            edge_truth[i,src,dst]=True
            if g.directed_transfer_polarity=="REJECTED":
                reject_truth[i,src,dst]=True
        row_out.append({
          "id":g.row_id,"family":r[1],
          "frozen_source_text":r[2],
          "gold_main_predicate":g.main_predicate,
          "gold_directed_recipient":g.transferred_recipient,
          "gold_directed_rejection":g.directed_transfer_polarity,
          "new_S57_predicted_event_token":toks[int(chosen_event[i])].form,
          "new_S57_main_predicate_correct":int(chosen_event[i])==src,
          "new_S57_proposed_directed_links":int(new_rec[i].sum()),
          "new_S57_proposed_REJECTED":int(new_rej[i].sum()),
          "gold_edge_safe":bool(pair_mask[i,src,dst]) if dst is not None else None,
          "human_S57_HELD_gold_loaded_after_all_model_forward":True,
        })
    require(int(edge_truth.sum())==12 and int(reject_truth.sum())==6
            and int(manual_neg_event.sum())==12,
            "S57 precommitted HELD24 original 12/6/12 manual gold totals drift")
    totals={
        "S26_RECIPIENT":counts(old26_rec,edge_truth),
        "S54_RECIPIENT":counts(old54_rec,edge_truth),
        "S56_RECIPIENT":counts(old56_rec,edge_truth),
        "S57_EVENTFIRST_RECIPIENT":counts(new_rec,edge_truth),
        "S26_REJECTED":counts(old26_rej,reject_truth),
        "S54_REJECTED":counts(old54_rej,reject_truth),
        "S56_REJECTED":counts(old56_rej,reject_truth),
        "S57_EVENTFIRST_REJECTED":counts(new_rej,reject_truth),
        "S26_NEGATED_event":counts(
           (raw_old.event_status.argmax(-1)==STATUS.index("NEGATED"))&valid,
           manual_neg_event),
        "S57_MAIN_PREDICATE": {
           "correct":int((chosen_event==event_truth).sum()),
           "total":24,
           "accuracy":float((chosen_event==event_truth).float().mean()),
        },
    }
    by_family={}
    for kind in FAMILIES:
        ids=torch.tensor([i for i,x in enumerate(SOURCE_ROWS) if x[1]==kind])
        require(len(ids)==6,"S57 HELD family 6-case precommitted structure changed")
        by_family[kind]={
          "gold_directed_recipient":int(edge_truth.index_select(0,ids).sum()),
          "gold_directed_REJECTED":int(reject_truth.index_select(0,ids).sum()),
          "old_S54_recipient":counts(old54_rec.index_select(0,ids),edge_truth.index_select(0,ids)),
          "old_S56_recipient":counts(old56_rec.index_select(0,ids),edge_truth.index_select(0,ids)),
          "new_S57_recipient":counts(new_rec.index_select(0,ids),edge_truth.index_select(0,ids)),
          "new_S57_REJECTED":counts(new_rej.index_select(0,ids),reject_truth.index_select(0,ids)),
          "S57_main_predicate_correct":int((chosen_event.index_select(0,ids)==event_truth.index_select(0,ids)).sum()),
          "old_BPE_unsafe_words":int(unsafe.index_select(0,ids).sum()),
        }
    report={
      "stage":STAGE,
      "first_HELD24_RAW_precommitted_commit":"0c8bf5a93e954350b2cc35f718d2d7c185019f10",
      "first_HELD24_RAW_precommitted_blob":"daf7dd4af8f66ab54ff44972bf7da144bbd4aafe",
      "first_HELD24_GOLD_precommitted_commit":"8f64040e066a9bdfc79d01108be2232e4535b443",
      "first_HELD24_GOLD_precommitted_blob":"20a1ebe556772fddc452fd4c55bbf17d47357b76",
      "new_HELD24_RAW_source_sha256":text_sha256(),
      "new_HELD24_human_GOLD_source_sha256":gold_sha256(),
      "first_HELD_true_original43M_GT_native339_SHA256":sha(a.native),
      "original_paid_S26_source_archive_SHA256":sha(a.s26),
      "original_old_S54_36FP16_SHA256":sha(a.s54),
      "original_old_S56_108FP16_SHA256":sha(a.s56),
      "new_S57_231FP16_SPENT96_only_SHA256":sha(a.s57),
      "all_original_GreenTunnel_native_and_four_paid_model_FROZEN":frozen,
      "original_616J7_protected_reproduced_before_new_HELD24_CPU_capture":True,
      "event_first_at_most_one_directed_recipient_per_input":True,
      "total_S57_proposed_recipient_pairs":int(new_rec.sum()),
      "total_S57_proposed_rejected_pairs":int(new_rej.sum()),
      "total_S57_no_safe_target_abstentions":result57["unavailable_native_pair_abstentions"],
      "first_HELD24_manual_gold_imported_ONLY_AFTER_all_models_forward":True,
      "first_HELD24_manual_gold_never_trained_tuned_or_selected":True,
      "first_HELD24_is_authored_narrow_synthetic_not_independent_realworld_EVAL":True,
      "first_HELD24_gold_recipient_count":int(edge_truth.sum()),
      "first_HELD24_gold_rejected_count":int(reject_truth.sum()),
      "first_HELD24_gold_main_negated_events":int(manual_neg_event.sum()),
      "first_HELD24_S26_S54_S56_S57_comparative_counts":totals,
      "first_HELD24_four_families":by_family,
      "first_HELD24_all24_manual_gold_diagnostics":row_out,
      "not_merged_or_activated_in_original_Calibri43M_GT_model":True,
      "no_streaming_causal_certification_bidirectional_context":True,
      "active_candidate_616J7_bpp_CB59_NOT_certified":True,
      "S38_S47_S48_HELD26_independent_EVAL32_128_NOT_loaded":True,
      "LIVE_Render_Supabase_Jessica_APK_not_modified":True,
      "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    a.out.mkdir(parents=True,exist_ok=False)
    dest=a.out/"NRA3_S57_TRUE43M_GT_EVENTFIRST_NEW24_FIRST_HELD_MANUAL_REPORT.json"
    dest.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                    encoding="utf-8")
    print("S57_ORIGINAL43M_GT_NEW24_MAIN_EVENT_FIRST_HELD",
          json.dumps(totals,sort_keys=True),flush=True)
    print("S57_EVENTFIRST_FOUR_STRUCTURAL_FAMILIES",
          json.dumps(by_family,sort_keys=True),flush=True)
    print("S57_ZERO_GATE_S57_HELD_ONCE_NO_MODEL_SELECTION",flush=True)


if __name__=="__main__":
    main()
