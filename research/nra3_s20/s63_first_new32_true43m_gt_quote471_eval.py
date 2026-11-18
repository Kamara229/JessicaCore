"""S63 one-time first NEW32 original Calibri43M GT quoted-outer action audit.

S63 physical471FP16 weights frozen from OLD280 original43M+GT before
S63 new32 CPU and manual gold. Replay original 616 J7 on unchanged
base first, native capture gold-blind, then frozen S57/59/60/61/62/63
forwards; AFTER all forwards only import separately precommitted gold.
Measure BOTH strict-unsafe BPE fail-closed S63 (primary) and unchecked
offline residual (diagnostic, explicitly NOT SAFE/CERTIFIED). This is
researcher-authored 4×8 challenge, NOT independent population EVAL.
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
from research.nra3_s20.s61_native_pair_null_polarity_2752 import (
    S61NativePairNull,
)
from research.nra3_s20.s62_joint_event_target_beam2166 import (
    S62JointEventRecipient,source_only_frozen_beam,
)
from research.nra3_s20.s63_quote_outer_strictbpe_residual471 import (
    S63QuoteOuterSafe,quote_overlay,FP16,
)
from research.nra3_s20.s63_new32_raw_only_precommitted import (
    SOURCE_ROWS,FAMILIES,raw_only_inputs,raw_sha256,
)
STAGE="NRA3_S63_TRUE_ORIGINAL43M_GT_NEW32_QUOTE_OUTER_STRICTBPE471_FIRSTREAD"
NATIVE="NRA3_S63_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
PROV="NRA3_S63_GENUINE43M_GT_EVENTFIRST32_OFFSET_PROVENANCE.json"
CAPTURE="NRA3_S63_GENUINE43M_GT_EVENTFIRST32_NATIVE_J7_REPORT.json"
S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
S57="NRA3_S57_EVENTFIRST_SPENT96_ONLY_231FP16_UNACCEPTED.safetensors"
S59="NRA3_S59_REAL43M_GT_OLD152_NATIVE320_SEMANTIC2046_FP16.safetensors"
S60="NRA3_S60_ORIGINAL43M_GT_OLD184_TRICLASS2776_FP16.safetensors"
S61="NRA3_S61_REAL43M_GT_SPENT216_PAID2752_PAIR_NONE.safetensors"
S62="NRA3_S62_REAL43M_GT_SPENT248_JOINT_EVENT_TARGET2166_FP16.safetensors"
S63="NRA3_S63_REAL43M_GT_SPENT280_QUOTE_OUTER_BPE471_FP16.safetensors"
FIT59="NRA3_S59_OLD152_TRAIN_NATIVE2046_FP16_MANIFEST.json"
FIT60="NRA3_S60_OLD184_TRAIN_TRICLASS2776_MANIFEST.json"
FIT61="NRA3_S61_SPENT216_PAIR_NONE2752_TRAIN_MANIFEST.json"
FIT62="NRA3_S62_OLD248_JOINT2166_TRAIN_MANIFEST.json"
FIT63="NRA3_S63_SPENT280_QUOTE471_TRAIN_MANIFEST.json"
REPORT="NRA3_S63_TRUE43M_GT_NEW32_QUOTE471_FIRSTREAD_REPORT.json"

def require(ok,msg):
    if not bool(ok): raise ValueError("S63 original Calibri43M GT new32: "+msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(pred,gold):
    tp=int((pred&gold).sum());fp=int((pred&~gold).sum())
    fn=int((~pred&gold).sum());den=2*tp+fp+fn
    return {"TP":tp,"FP":fp,"FN":fn,"precision":tp/(tp+fp) if tp+fp else 0.,
            "recall":tp/(tp+fn) if tp+fn else 0.,
            "F1":2*tp/den if den else 0.}

def main():
    p=argparse.ArgumentParser()
    for key in ("native","prov","capture","s26","s57",
                "s59","fit59","s60","fit60","s61","fit61",
                "s62","fit62","s63","fit63","out"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"NEVER overwrite first S63 HELD32 score")
    files=(a.native,a.prov,a.capture,a.s26,a.s57,
           a.s59,a.fit59,a.s60,a.fit60,a.s61,a.fit61,
           a.s62,a.fit62,a.s63,a.fit63)
    names=(NATIVE,PROV,CAPTURE,S26,S57,S59,FIT59,S60,FIT60,S61,FIT61,
           S62,FIT62,S63,FIT63)
    require(all(z.is_file() and z.name==n for z,n in zip(files,names)),
            "true original GT, old parent or new FP16 file missing")
    prov=json.loads(a.prov.read_text())
    cap=json.loads(a.capture.read_text())
    m59=json.loads(a.fit59.read_text())
    m60=json.loads(a.fit60.read_text())
    m61=json.loads(a.fit61.read_text())
    m62=json.loads(a.fit62.read_text())
    m63=json.loads(a.fit63.read_text())
    require(
        cap["stage"]=="NRA3_S63_GENUINE43M_GT_EVENTFIRST32_NATIVE_CPU_FIRST_PASS"
        and cap["S63_FROZEN_HELD32_PLAINTEXT_SHA256"]==raw_sha256()
        and cap["S63_HELD32_SOURCE_COMMIT"]==
             "0f7b32b4c29f78fc3771e2e950bbdc4edadee2a3"
        and cap["S63_HELD32_SOURCE_BLOB"]==
             "a19ef64d01b05e2b26d4062f2f66232c691d846f"
        and cap["all_616_native_CPU_generated_token_digests_unchanged"]
        and cap["original_456_full_EOS"]==456 and cap["original_new128_full_EOS"]==128
        and cap["actual_43M_CPU_12_transformer_layers"]
        and cap["GT_residuals_before_readport_in_same_active_S93_scope"]
        and cap["actual_genuine_43M_CPU_new_HELD32_prefills"]==32
        and cap["S63_HELD_manual_gold_NOT_IMPORTED_into_model_capture"]
        and cap["NRA3_output_gate"]==0 and not cap["NRA3_candidate_accepted"]
        and prov["stage"]=="NRA3_S63_GENUINE43M_GT_EVENTFIRST32_BPE_SOURCE_ONLY"
        and prov["frozen_S63_HELD32_raw_source_sha256"]==raw_sha256()
        and prov["original_616_J7_replayed_before_read_only_quotes"]
        and prov["derived_REAL43M_WORD339_SHA256"]==sha(a.native)
        and prov["new_BPE_fractional_aligned_count"]==32
        and prov["S63_HUMAN_GOLD_NOT_IMPORTED_DURING_ANY_CAPTURE_STEP"],
        "real original43M GT replay 616J7 and source-only precommitted NEW32 SHA failed")
    require(
        m63["stage"]=="NRA3_S63_GENUINE43M_GT_SPENT280_QUOTE_OUTER_STRICT_BPE471FP16"
        and m63["new_S63_PAID471_SHA256"]==sha(a.s63)
        and m63["new_S63_FP16_parameters"]==FP16==471
        and m63["original_S26_PAID12384_SHA256"]==sha(a.s26)
        and m63["old_S57_PAID231_SHA256"]==sha(a.s57)
        and m63["old_S59_PAID2046_SHA256"]==sha(a.s59)
        and m63["old_S61_PAID2752_SHA256"]==sha(a.s61)
        and m63["old_S62_PAID2166_SHA256"]==sha(a.s62)
        and m63["new_S63_NEW32_raw_gold_NEVER_IMPORTED_TO_OLD280_TRAIN"]
        and m63["original_Calibri43M_GT_and_all_old_paids_unchanged"]
        and m63["output_gate"]==0 and not m63["candidate_accepted"]
        and not m63["activation_allowed"]
        and m59["new_S59_semantic_paid2046_sha256"]==sha(a.s59)
        and m60["new_S60_paid2776_sha256"]==sha(a.s60)
        and m61["new_S61_paid2752_SHA256"]==sha(a.s61)
        and m62["new_S62_joint2166_FP16_SHA256"]==sha(a.s62),
        "S63 frozen471 or ancestor S26/S57/S59/S61/S62 FP16 SHA failed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    native=load_file(str(a.native),device="cpu")
    require(set(native)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S63_HELD32"},
        "NEW32 original43M postGT capture must not contain manual gold tensor")
    x=native["original_real43m_gt_word339_READONLY"].detach()
    valid=native["original_native_word_valid_mask"].detach()
    unsafe=native["native_BPE_causally_unsafe_word_mask"].detach()
    require(x.ndim==3 and x.shape[0]==32 and x.shape[-1]==339
            and x.dtype==torch.float32 and
            valid.shape==unsafe.shape==x.shape[:2]
            and valid.dtype==unsafe.dtype==torch.bool
            and native["source_row_indices_in_frozen_S63_HELD32"].tolist()
                ==list(range(32)),
            "S63 protected original native BPE339 shape/type/original index")
    native_sha=tensor_sha(x,valid,unsafe)
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
    new63=S63QuoteOuterSafe.from_fp16(
        load_file(str(a.s63),device="cpu")["S63_QUOTE_OUTER_STRICTBPE471_FP16"])
    models=(old26,old57,old59,old60,old61,old62,new63)
    require(all(not p.requires_grad for model in models for p in model.parameters())
            and all(z.output_gate==0 and not z.candidate_accepted
                    and not z.activation_allowed for z in models),
            "old original43M GT and all S63 paid heads must be frozen/gate0")
    ancestry=[tensor_sha(*(p.detach() for p in model.parameters()))
              for model in models]
    with torch.inference_mode():
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
        ev=gold_blind_eventfirst_features(x,obs,raw_only_inputs())
        sem=semantic_features(x,ev)
        fus=source_local_scope(sem,raw_only_inputs())
        logits57=tuple(z.detach() for z in old57(ev))
        pred57=old57.decode(ev)
        pred59=old59.decode(sem,logits57)
        pred60=old60.decode(fus,logits57,old59)
        pred61=old61.decode(fus,logits57,old59)
        beam=source_only_frozen_beam(fus,logits57,old59,old61)
        pred62=old62.decode(beam)
        overlay=quote_overlay(beam,raw_only_inputs(),unsafe)
        pred63=new63.decode(overlay,old62,strict_native_safety=True)
        # NOT DEPLOYABLE: offline unsafe-ablation scored only for explaining
        # effect of fail-closed. Never use this as model selection metric.
        pred63_unsafe=new63.decode(overlay,old62,strict_native_safety=False)
        unchanged=native_sha==tensor_sha(x,valid,unsafe) and all(
            tensor_sha(*(p.detach() for p in m.parameters()))==z
            for m,z in zip(models,ancestry))
    require(unchanged and pred63["output_gate"]==0 and
            not pred63["candidate_accepted"] and
            int(pred63["directed_recipient"].sum())<=32 and
            not bool((pred63["directed_recipient"]&~ev.valid_pair).any())
            and not bool((pred63["directed_rejected"]&
                          ~pred63["directed_recipient"]).any())
            and not bool((pred63["directed_recipient"]&
                          overlay.unsafe_row[:,None,None]).any()),
            "first S63 true43M GT forward modified weights or unsafe native edge")
    # *** FIRST import manually annotated separate S63 new32 GOLD AFTER ALL MODEL FORWARDS ***
    from research.nra3_s20.s63_new32_separate_manual_gold_NEVER_TRAIN import (
        held_gold32,gold_sha256,
    )
    manual=held_gold32()
    require(len(manual)==len(SOURCE_ROWS)==32,"S63 precommitted gold/raw count drift")
    gold_link=torch.zeros_like(pred63["directed_recipient"])
    gold_rejected=torch.zeros_like(gold_link)
    gold_event=torch.zeros(32,dtype=torch.long)
    diagnostics=[]
    for i,(source,g) in enumerate(zip(SOURCE_ROWS,manual)):
        require(source[0]==g.row_id,"S63 manual precommitted gold row alignment")
        words=raw_surface_grammar(source[2])
        require(len(words)==int(valid[i].sum()),
                "S63 original native token text/annotation misaligned")
        def find(word):
            matches=[j for j,t in enumerate(words) if t.form==word]
            require(len(matches)==1,"S63 exact manual word missing/ambiguous")
            return matches[0]
        src=find(g.main_predicate)
        gold_event[i]=src
        dst=None
        if g.transferred_recipient is not None:
            dst=find(g.transferred_recipient)
            gold_link[i,src,dst]=True
            if g.directed_transfer_polarity=="REJECTED":
                gold_rejected[i,src,dst]=True
        diagnostics.append({
            "id":g.row_id,"family":source[1],
            "gold_predicate":g.main_predicate,
            "gold_recipient":g.transferred_recipient,
            "gold_polarity":g.directed_transfer_polarity,
            "gold_source_mask_allowed":bool(ev.candidate_event[i,src]),
            "gold_pair_BPE_mask_safe":bool(ev.valid_pair[i,src,dst])
                                      if dst is not None else None,
            "frozen_top8_source_covered":bool(
                ((beam.sources[i]==src)&beam.allowed[i]).any()),
            "strict_BPE_unsafe_row":bool(overlay.unsafe_row[i]),
            "S62_predicted_predicate":words[int(pred62["chosen_event"][i])].form,
            "S63_predicted_predicate":words[int(pred63["chosen_event"][i])].form,
            "S63_strict_link_count":int(pred63["directed_recipient"][i].sum()),
            "S63_strict_rejected_count":int(pred63["directed_rejected"][i].sum()),
            "S63_unsafe_ablation_link_count":int(
                pred63_unsafe["directed_recipient"][i].sum()),
        })
    require(int(gold_link.sum())==24 and int(gold_rejected.sum())==8,
            "S63 frozen NEW32 manual labels must be 24 directed 8 REJECTED")
    versions={
        "S57":pred57,"S59_native2046":pred59,"S60_local2776":pred60,
        "S61_pair_null2752":pred61,"S62_joint2166":pred62,
        "S63_quote_strict471":pred63,
    }
    totals={}
    for name,pred in versions.items():
        totals[name+"_RECIPIENT"]=metric(pred["directed_recipient"],gold_link)
        totals[name+"_REJECTED"]=metric(pred["directed_rejected"],gold_rejected)
        source=(pred["chosen_outer_event_index"] if name=="S57"
                else pred["chosen_event"])
        totals[name+"_MAIN_EVENT"]={
            "correct":int((source==gold_event).sum()),"total":32,
            "accuracy":float((source==gold_event).float().mean()),
        }
    strata={}
    for family in FAMILIES:
        ix=torch.tensor([j for j,z in enumerate(SOURCE_ROWS) if z[1]==family])
        require(len(ix)==8,"S63 frozen new32 4x8 structural families drift")
        strata[family]={}
        for name,pred in versions.items():
            strata[family][name+"_RECIPIENT"]=metric(
                pred["directed_recipient"][ix],gold_link[ix])
            strata[family][name+"_REJECTED"]=metric(
                pred["directed_rejected"][ix],gold_rejected[ix])
    # BPE unsafe ablation is post-score DIAGNOSTIC, not valid safe model output.
    diagnostic_ablation={
        "offline_UNSAFE_BPE_ablation_NOT_Causal_certified":True,
        "NOT_a_safe_or_accepted_candidate_score":True,
        "offline_S63_no_strict_BPE_RECIPIENT":metric(
            pred63_unsafe["directed_recipient"],gold_link),
        "offline_S63_no_strict_BPE_REJECTED":metric(
            pred63_unsafe["directed_rejected"],gold_rejected),
    }
    report={
        "stage":STAGE,
        "S63_true_original43M_GT_CPU_12layers_original616_J7_replayed":True,
        "original_616_J7_BASE_ONLY_not_active_S63_candidate":True,
        "original_43M_GT_and_old_readout_physical_SHA_unchanged":unchanged,
        "S63_RAW_precommitted_commit":"0f7b32b4c29f78fc3771e2e950bbdc4edadee2a3",
        "S63_RAW_precommitted_git_blob":"a19ef64d01b05e2b26d4062f2f66232c691d846f",
        "S63_GOLD_separately_precommitted_commit":"1cfd2892ced0d399a1a0708e012879334365bf1e",
        "S63_GOLD_precommitted_git_blob":"14ac6e9e62074deca49af80d481f12fd4232287e",
        "S63_RAW32_SHA256":raw_sha256(),
        "S63_SEPARATE_GOLD32_SHA256":gold_sha256(),
        "S63_original_real43M_GT_native339_sha256":sha(a.native),
        "S63_fixed_trained_471FP16_sha256":sha(a.s63),
        "first_NEW32_GOLD_imported_ONLY_AFTER_ALL_frozen_predictions":True,
        "S63_NEW32_NOW_PERMANENTLY_SPENT":True,
        "this_4x8_is_researcher_written_NOT_random_population_EVAL":True,
        "new_32_ambiguous_or_unbalanced_BPE_STRICT_abstained":int(overlay.unsafe_row.sum()),
        "new_32_original_shared_native_BPE_piece_count":
            int(cap["S51_shared_native_BPE_piece_count"]),
        "S63_strict_BPE_gate_no_directed_link_on_any_unsafe_row":True,
        "S63_beam8_true_event_coverage":sum(
            z["frozen_top8_source_covered"] for z in diagnostics),
        "S63_gold_main_predicate_source_mask_eligible":sum(
            z["gold_source_mask_allowed"] for z in diagnostics),
        "S63_gold_direction_pair_BPE_mask_safe":sum(
            z["gold_pair_BPE_mask_safe"] is True for z in diagnostics),
        "S63_strict_BPE_native_unsafe_no_streaming_causal_cert":True,
        "NEW32_FIRST_READ_all_frozen_models_metrics":totals,
        "NEW32_FIRST_READ_per_four_8case_families":strata,
        "NEW32_FIRST_READ_row_level_diagnostics":diagnostics,
        "POST_FORWARD_UNSAFE_BPE_ABLATION_NONCERTIFIABLE":diagnostic_ablation,
        "no_NEW32_model_selection_or_threshold_tuning":True,
        "protected_S38_S47_S48_EVAL32_EVAL128_not_loaded":True,
        "active_integrated_S63_616J7_bpp_CB59_not_certified":True,
        "full_context_original_CPU_readout_not_streaming_causal_certified":True,
        "LIVE_Render_Supabase_Jessica_APK_UNCHANGED":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/REPORT).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf-8")
    print("S63_TRUE_ORIGINAL43M_GT_NEW32_FIRSTREAD_QUOTE471_STRICT",
          json.dumps(totals,sort_keys=True),flush=True)
    print("S63_REAL43M_GT_NEW32_STRICT_BPE_GATE_AUDIT",
          json.dumps({
              "unsafe_rows":report["new_32_ambiguous_or_unbalanced_BPE_STRICT_abstained"],
              "shared_BPE_pieces":report["new_32_original_shared_native_BPE_piece_count"],
              "beam8_coverage":report["S63_beam8_true_event_coverage"],
              "unsafe_ablation_DO_NOT_CERTIFY":diagnostic_ablation,
              "gate":0,
          },sort_keys=True),flush=True)
    print("S63_NEW32_NOW_SPENT_GATE0_CANDIDATE_NOT_INTEGRATED",flush=True)

if __name__=="__main__":main()
