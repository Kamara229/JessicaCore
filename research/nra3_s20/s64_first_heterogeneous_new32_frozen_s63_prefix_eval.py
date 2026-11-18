"""S64 TRUE original43M+GT NEW32 first-read: frozen S63 distribution shift + BPE.

NO new trainable weights. All older S57/S59/S60/S61/S62/S63 readouts
unchanged and SHA frozen; source-only S64 NEW32 raw (researcher-authored
heterogeneous challenges) physically captured by original CPU after
616 protected original J7. NEW32 GOLD imported ONLY AFTER all predictions.
Score BOTH same conservative prefix-risk/BPE gate S62 and S63 to avoid
unfair comparison to old S62 on mixed unsafe rows. Also show old S63
strict-gate and prefix-risk guarded as research diagnostic. Never claim
original model, GT full context or prefix-risk guard streaming certified.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors.torch import load_file
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s26_predicted_event_inversion_rescue import EventGroundedInversionGraph
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    S52QuoteAwareFrozenPaidGraphBridge,tensor_sha,
)
from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import (
    S57EventFirstRecipient,gold_blind_eventfirst_features,
)
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    S59NativeEventSemantic,semantic_features,
)
from research.nra3_s20.s60_native43m_local_triclass_model import (
    S60Joint3,source_local_scope,
)
from research.nra3_s20.s61_native_pair_null_polarity_2752 import S61NativePairNull
from research.nra3_s20.s62_joint_event_target_beam2166 import (
    S62JointEventRecipient,source_only_frozen_beam,
)
from research.nra3_s20.s63_quote_outer_strictbpe_residual471 import (
    S63QuoteOuterSafe,quote_overlay,
)
from research.nra3_s20.s64_new32_raw_only_preregistered import (
    SOURCE_ROWS,FAMILIES,raw_only_inputs,raw_sha256,
)
STAGE="NRA3_S64_TRUE_ORIGINAL43M_GT_NEW32_HELD_GENERALIZATION_AND_PREFIX_CAUSAL_AUDIT"
NATIVE="NRA3_S64_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
PROV="NRA3_S64_GENUINE43M_GT_EVENTFIRST32_OFFSET_PROVENANCE.json"
CAPTURE="NRA3_S64_GENUINE43M_GT_EVENTFIRST32_NATIVE_J7_REPORT.json"
PREFIX="NRA3_S64_ORIGINAL_NATIVE_BPE_PREFIX_STABILITY_AUDIT.json"
S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
S57="NRA3_S57_EVENTFIRST_SPENT96_ONLY_231FP16_UNACCEPTED.safetensors"
S59="NRA3_S59_REAL43M_GT_OLD152_NATIVE320_SEMANTIC2046_FP16.safetensors"
S60="NRA3_S60_ORIGINAL43M_GT_OLD184_TRICLASS2776_FP16.safetensors"
S61="NRA3_S61_REAL43M_GT_SPENT216_PAID2752_PAIR_NONE.safetensors"
S62="NRA3_S62_REAL43M_GT_SPENT248_JOINT_EVENT_TARGET2166_FP16.safetensors"
S63="NRA3_S63_REAL43M_GT_SPENT280_QUOTE_OUTER_BPE471_FP16.safetensors"
FIT63="NRA3_S63_SPENT280_QUOTE471_TRAIN_MANIFEST.json"
REPORT="NRA3_S64_REAL43M_GT_NEW32_FIRSTREAD_GENERALIZATION_PREFIX_AUDIT.json"

def require(ok,why):
    if not bool(ok):raise ValueError("S64 genuine original43M GT NEW32 first-read: "+why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(pred,ref):
    tp=int((pred&ref).sum());fp=int((pred&~ref).sum());fn=int((~pred&ref).sum())
    den=2*tp+fp+fn
    return {"TP":tp,"FP":fp,"FN":fn,
            "precision":tp/(tp+fp) if tp+fp else 0.,
            "recall":tp/(tp+fn) if tp+fn else 0.,
            "F1":(2*tp/den) if den else 0.}

def masked_out(pred,invalid):
    out=dict(pred)
    out["directed_recipient"]=pred["directed_recipient"]&~invalid[:,None,None]
    out["directed_rejected"]=pred["directed_rejected"]&~invalid[:,None,None]
    require(not bool(out["directed_recipient"][invalid].any())
            and not bool(out["directed_rejected"][invalid].any()),
            "s64 risk/strict row gate emitted unsafe edge")
    return out

def main():
    p=argparse.ArgumentParser()
    for z in ("native","prov","capture","prefix","s26","s57","s59","s60",
              "s61","s62","s63","fit63","out"):
        p.add_argument("--"+z,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"never rescore first S64 NEW32 once completed")
    paths=(a.native,a.prov,a.capture,a.prefix,a.s26,a.s57,a.s59,
           a.s60,a.s61,a.s62,a.s63,a.fit63)
    names=(NATIVE,PROV,CAPTURE,PREFIX,S26,S57,S59,S60,S61,S62,S63,FIT63)
    require(all(z.is_file() and z.name==n for z,n in zip(paths,names)),
            "strict exact true original model and previous archived heads")
    prov=json.loads(a.prov.read_text())
    cpu=json.loads(a.capture.read_text())
    prefix=json.loads(a.prefix.read_text())
    fit=json.loads(a.fit63.read_text())
    require(
        cpu["stage"]=="NRA3_S64_GENUINE43M_GT_EVENTFIRST32_NATIVE_CPU_FIRST_PASS"
        and cpu["S64_FROZEN_HELD32_PLAINTEXT_SHA256"]==raw_sha256()
        and cpu["S64_HELD32_SOURCE_COMMIT"]==
             "a3e40feec6f321176d719498e054714fb615a779"
        and cpu["S64_HELD32_SOURCE_BLOB"]==
             "828bc717360407403764c39b9d659f4ea754685f"
        and cpu["all_616_native_CPU_generated_token_digests_unchanged"]
        and cpu["original_456_full_EOS"]==456 and cpu["original_new128_full_EOS"]==128
        and cpu["actual_43M_CPU_12_transformer_layers"]
        and cpu["GT_residuals_before_readport_in_same_active_S93_scope"]
        and cpu["actual_genuine_43M_CPU_new_HELD32_prefills"]==32
        and cpu["S64_HELD_manual_gold_NOT_IMPORTED_into_model_capture"]
        and cpu["NRA3_output_gate"]==0 and not cpu["NRA3_candidate_accepted"]
        and prov["stage"]=="NRA3_S64_GENUINE43M_GT_EVENTFIRST32_BPE_SOURCE_ONLY"
        and prov["frozen_S64_HELD32_raw_source_sha256"]==raw_sha256()
        and prov["derived_REAL43M_WORD339_SHA256"]==sha(a.native)
        and prov["original_616_J7_replayed_before_read_only_quotes"]
        and prov["new_BPE_fractional_aligned_count"]==32
        and prov["S64_HUMAN_GOLD_NOT_IMPORTED_DURING_ANY_CAPTURE_STEP"],
        "true ORIGINAL43M GT protected616 J7 CPU or new32 source-only SHA")
    require(
        prefix["stage"]=="NRA3_S64_TRUE_ORIGINAL43M_GT_NEW32_GOLDBLIND_BPE_PREFIX_AUDIT"
        and len(prefix["rows"])==32
        and prefix["only_unmodified_original_model_tokenizer_used"]
        and prefix["same_full_original_model_prompt_BPE_IDs_verified"]
        and prefix["all_616_original_J7_replayed_BEFORE_audit"]
        and prefix["does_NOT_certify_full_context_original_GT_online_causality"]
        and prefix["source_ONLY_never_imported_S64_gold_or_EVAL"]
        and prov["S64_source_only_native_prefix_audit_SHA256"]==sha(a.prefix)
        and cpu["S64_original_BPE_prefix_audit_SHA256"]==sha(a.prefix),
        "S64 native original tokenizer word prefix stability report altered")
    require(
        fit["stage"]=="NRA3_S63_GENUINE43M_GT_SPENT280_QUOTE_OUTER_STRICT_BPE471FP16"
        and fit["new_S63_PAID471_SHA256"]==sha(a.s63)
        and fit["original_S26_PAID12384_SHA256"]==sha(a.s26)
        and fit["old_S57_PAID231_SHA256"]==sha(a.s57)
        and fit["old_S59_PAID2046_SHA256"]==sha(a.s59)
        and fit["old_S61_PAID2752_SHA256"]==sha(a.s61)
        and fit["old_S62_PAID2166_SHA256"]==sha(a.s62)
        and fit["original_Calibri43M_GT_and_all_old_paids_unchanged"]
        and fit["output_gate"]==0 and not fit["candidate_accepted"],
        "frozen original43M GT S63/older FP16 ancestral hash mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    archive=load_file(str(a.native),device="cpu")
    require(set(archive)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S64_HELD32"},
        "new32 original CPU archive includes unexpected role labels")
    x=archive["original_real43m_gt_word339_READONLY"].detach()
    valid=archive["original_native_word_valid_mask"].detach()
    unsafe=archive["native_BPE_causally_unsafe_word_mask"].detach()
    require(x.ndim==3 and x.shape[0]==32 and x.shape[-1]==339
            and x.dtype==torch.float32
            and valid.shape==unsafe.shape==x.shape[:2]
            and valid.dtype==unsafe.dtype==torch.bool
            and archive["source_row_indices_in_frozen_S64_HELD32"].tolist()
                ==list(range(32)),
            "S64 exact goldblind native339 original tensor ABI")
    oldsha=tensor_sha(x,valid,unsafe)
    old26=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(a.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    old59=S59NativeEventSemantic.from_fp16(
        load_file(str(a.s59),device="cpu")["S59_NATIVE320_SEMANTIC2046_FP16"])
    old60=S60Joint3.from_fp16(
        load_file(str(a.s60),device="cpu")["S60_SOURCE_LOCAL_TRICLASS2776_FP16"])
    old61=S61NativePairNull.from_fp16(
        load_file(str(a.s61),device="cpu")["S61_REAL43M_GT_PAIR_NULL_POLAR2752_FP16"])
    old62=S62JointEventRecipient.from_fp16(
        load_file(str(a.s62),device="cpu")["S62_JOINT_EVENT_TARGET2166_FP16"])
    old63=S63QuoteOuterSafe.from_fp16(
        load_file(str(a.s63),device="cpu")["S63_QUOTE_OUTER_STRICTBPE471_FP16"])
    models=(old26,old57,old59,old60,old61,old62,old63)
    require(all(not p.requires_grad for m in models for p in m.parameters())
            and all(m.output_gate==0 and not m.candidate_accepted
                    and not m.activation_allowed for m in models),
            "true original and readout weights not frozen/gate0")
    weights=[tensor_sha(*(z.detach() for z in m.parameters())) for m in models]
    prefix_risk=torch.tensor([
        bool(z["prefix_token_identifier_mismatch_words"]) or
        bool(z["strict_shared_native_words_unsafe"])
        for z in prefix["rows"]],dtype=torch.bool)
    require(prefix_risk.shape==(32,),"S64 prefix word probe row ABI")
    with torch.inference_mode():
        observed=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
        event=gold_blind_eventfirst_features(x,observed,raw_only_inputs())
        semantic=semantic_features(x,event)
        source=source_local_scope(semantic,raw_only_inputs())
        oldlogits=tuple(z.detach() for z in old57(event))
        p57=old57.decode(event)
        p59=old59.decode(semantic,oldlogits)
        p60=old60.decode(source,oldlogits,old59)
        p61=old61.decode(source,oldlogits,old59)
        b=source_only_frozen_beam(source,oldlogits,old59,old61)
        p62=old62.decode(b)
        overlay=quote_overlay(b,raw_only_inputs(),unsafe)
        p63=old63.decode(overlay,old62,strict_native_safety=True)
        # Matched gating across older model S62 and new S63 is necessary.
        p62matched=masked_out(p62,overlay.unsafe_row)
        combined_risk=overlay.unsafe_row|prefix_risk
        p62risk=masked_out(p62,combined_risk)
        p63risk=masked_out(p63,combined_risk)
        # NOT SAFE: predicted on fractional shared BPE without strict gate
        # for diagnostic only; this result NEVER substitutes safe score.
        p63_no_gate=old63.decode(overlay,old62,strict_native_safety=False)
        stable=(oldsha==tensor_sha(x,valid,unsafe)
                and all(tensor_sha(*(z.detach() for z in m.parameters()))==h
                        for m,h in zip(models,weights)))
    require(stable and p63["output_gate"]==0 and not p63["candidate_accepted"]
            and not bool((p63["directed_recipient"]&
                          overlay.unsafe_row[:,None,None]).any())
            and not bool((p63risk["directed_recipient"]&
                          combined_risk[:,None,None]).any())
            and not bool((p62risk["directed_recipient"]&
                          combined_risk[:,None,None]).any())
            and not bool((p63["directed_recipient"]&~event.valid_pair).any()),
            "genuine old43M GT mutated or unsafe/quote link leaked")
    # FIRST human annotation import ONLY AFTER ALL S57..S63 forwards.
    from research.nra3_s20.s64_new32_separate_manual_gold_NEVER_TRAIN import (
        held_gold32,gold_sha256,
    )
    manual=held_gold32()
    require(len(manual)==len(SOURCE_ROWS)==32,
            "precommitted separately annotated source rows incorrect")
    truth=torch.zeros_like(p63["directed_recipient"])
    neg=torch.zeros_like(truth)
    gold_event=torch.zeros(32,dtype=torch.long)
    diagnostics=[]
    for row,(raw,g) in enumerate(zip(SOURCE_ROWS,manual)):
        require(raw[0]==g.row_id,"S64 precommitted manual gold order changed")
        words=raw_surface_grammar(raw[2])
        require(len(words)==int(valid[row].sum()),
                "raw gold/native Calibri word alignment mismatch")
        def index(form):
            hits=[j for j,t in enumerate(words) if t.form==form]
            require(len(hits)==1,
                    "S64 manually frozen gold source word absent/not unique")
            return hits[0]
        src=index(g.main_predicate)
        gold_event[row]=src
        dst=None
        if g.transferred_recipient is not None:
            dst=index(g.transferred_recipient)
            truth[row,src,dst]=True
            if g.directed_transfer_polarity=="REJECTED":neg[row,src,dst]=True
        diagnostics.append({
            "id":g.row_id,"family":raw[1],
            "gold_event":g.main_predicate,
            "gold_recipient":g.transferred_recipient,
            "gold_rejected":g.directed_transfer_polarity,
            "predicted_S63_main_event":words[int(p63["chosen_event"][row])].form,
            "predicted_S62_main_event":words[int(p62["chosen_event"][row])].form,
            "gold_event_eligible_source_mask":bool(event.candidate_event[row,src]),
            "gold_pair_BPE_safe":bool(event.valid_pair[row,src,dst])
                    if dst is not None else None,
            "true_event_in_frozen_top8":bool(((b.sources[row]==src)
                                            &b.allowed[row]).any()),
            "strict_native_unsafe_or_unclosed_quote":bool(overlay.unsafe_row[row]),
            "native_prefix_token_identifier_unstable":bool(prefix_risk[row]),
            "S63_strict_link_count":int(p63["directed_recipient"][row].sum()),
            "S63_prefix_risk_guard_link_count":int(
                p63risk["directed_recipient"][row].sum()),
        })
    require(int(truth.sum())==20 and int(neg.sum())==3,
            "S64 new32 manually prereg 20 recipient/3 REJECTED gold drift")
    frozen={
        "S57":p57,"S59":p59,"S60":p60,"S61":p61,
        "S62_ungated":p62,
        "S62_Matched_BPE_and_unclosed_quote":p62matched,
        "S63_original_STRICT_BPE":p63,
        "S62_Matched_BPE_PLUS_prefix_risk_abstain":p62risk,
        "S63_BPE_PLUS_prefix_risk_abstain":p63risk,
    }
    results={}
    for label,pred in frozen.items():
        results[label+"_RECIPIENT"]=metrics(pred["directed_recipient"],truth)
        results[label+"_REJECTED"]=metrics(pred["directed_rejected"],neg)
        chosen=pred["chosen_outer_event_index"] if label=="S57" else pred["chosen_event"]
        results[label+"_MAIN_PREDICATE"]={
            "correct":int((chosen==gold_event).sum()),"total":32,
            "accuracy":float((chosen==gold_event).float().mean()),
        }
    strata={}
    for family in FAMILIES:
        ids=torch.tensor([i for i,z in enumerate(SOURCE_ROWS) if z[1]==family])
        require(len(ids)==8,"S64 four 8-case audit strata drift")
        strata[family]={}
        for label,pred in frozen.items():
            if label in ("S63_original_STRICT_BPE",
                         "S62_Matched_BPE_and_unclosed_quote",
                         "S62_Matched_BPE_PLUS_prefix_risk_abstain",
                         "S63_BPE_PLUS_prefix_risk_abstain"):
                strata[family][label+"_RECIPIENT"]=metrics(
                    pred["directed_recipient"][ids],truth[ids])
                strata[family][label+"_REJECTED"]=metrics(
                    pred["directed_rejected"][ids],neg[ids])
    result={
        "stage":STAGE,
        "new_S64_no_new_trainable_parameters":True,
        "original_12layer_43M_GT_real_CPU_616J7_BASELINE_replayed_before_new32":True,
        "original616_BASE_ONLY_not_S64_integrated_J7":True,
        "all_original_weights_and_paid_frozen_unmodified":stable,
        "S64_RAW_precommitted_commit":"a3e40feec6f321176d719498e054714fb615a779",
        "S64_RAW_precommitted_git_blob":"828bc717360407403764c39b9d659f4ea754685f",
        "S64_separate_manual_gold_commit":"cbef673ccc468cde8af02b995ee1911895d1f1b5",
        "S64_separate_manual_gold_git_blob":"e4dbccca8f8c5db3e285d17c9b2956a8b4c87299",
        "S64_RAW_SHA256":raw_sha256(),
        "S64_separate_manual_GOLD_SHA256":gold_sha256(),
        "S64_original_word339_SHA256":sha(a.native),
        "S64_original_native_prefix_stability_report_SHA256":sha(a.prefix),
        "S64_frozen_paid_S63_471FP16_SHA256":sha(a.s63),
        "human_gold_imported_ONLY_AFTER_ALL_preexisting_frozen_forwards":True,
        "first_READ_NEW32_now_PERMANENTLY_spent":True,
        "author_created_contrast_NOT_multi_author_or_population_independent":True,
        "strict_native_shared_BPE_or_unbalanced_quote_abstentions":int(
            overlay.unsafe_row.sum()),
        "original_native_piece_shared_count":int(cpu["S51_shared_native_BPE_piece_count"]),
        "conservative_prefix_token_identifier_risk_rows":int(prefix_risk.sum()),
        "combined_strict_BPE_and_prefix_risk_rows":int(combined_risk.sum()),
        "original_true_gold_source_BPE_eligible":sum(
            z["gold_event_eligible_source_mask"] for z in diagnostics),
        "original_true_gold_pair_BPE_safe":sum(
            z["gold_pair_BPE_safe"] is True for z in diagnostics),
        "frozen_top8_TRUE_event_coverage":sum(z["true_event_in_frozen_top8"]
                                              for z in diagnostics),
        "first_READ_all_frozen_S57_to_S63_and_matched_conservative_gate":results,
        "first_READ_by_four_challenge_strata":strata,
        "first_READ_row_diagnostics":diagnostics,
        "UNSAFE_UNGATED_native_fractional_ablation_NOT_CERTIFIED":{
            "not_safe_not_accepted_not_primary":True,
            "recipient":metrics(p63_no_gate["directed_recipient"],truth),
            "rejected":metrics(p63_no_gate["directed_rejected"],neg),
        },
        "prefix_BPE_stability_not_equivalent_to_original_hidden_GT_causal_cert":True,
        "original_model_full_context_GT_no_online_streaming_certification":True,
        "no_new32_optimizer_threshold_selection_reuse":True,
        "protected_EVAL128_EVAL32_S38_S47_S48_untouched":True,
        "no_live_Render_Supabase_Jessica_APK_changes":True,
        "active_J7_bpp_CPU_CB59_not_measured":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/REPORT).write_text(
        json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf-8")
    print("S64_REAL43M_GT_HELD32_FIRST_GENERALIZATION_MATCHED_GATE",
          json.dumps(results,sort_keys=True),flush=True)
    print("S64_REAL43M_GT_NATIVE_BPE_PREFIX_SAFETY_DIAGNOSTIC",
          json.dumps({"strict_abstain_rows":int(overlay.unsafe_row.sum()),
                      "prefix_risk_rows":int(prefix_risk.sum()),
                      "combined":int(combined_risk.sum()),
                      "shared":int(cpu["S51_shared_native_BPE_piece_count"]),
                      "frozen_S59_beam_gold_coverage":
                        result["frozen_top8_TRUE_event_coverage"],
                      "gate":0},sort_keys=True),flush=True)
    print("S64_NEW32_NOW_SPENT_RESEARCH_ONLY_NOT_ACCEPTED",flush=True)

if __name__=="__main__":main()
