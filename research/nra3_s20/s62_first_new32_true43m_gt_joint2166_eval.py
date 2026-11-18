"""S62 one-time first-read true original Calibri43M + GreenTunnel CPU NEW32.

S62 2166FP16 physically trained only on previously scored old248,
SHA-locked BEFORE any new S62 held read. Original protected 616 J7
replayed on unchanged model before NEW32 gold-blind native CPU capture.
All S57/S59/S60/S61/S62 frozen prediction forwards done BEFORE manual
S62 NEW32 gold is imported. This is researcher-authored 4x8 challenge,
NOT population evaluation, NOT online autoregressive certification.
NEW32 permanently SPENT once this first read is scored.
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
    S62JointEventRecipient,source_only_frozen_beam,K,PAID,
)
from research.nra3_s20.s62_new32_preregistered_raw_only import (
    SOURCE_ROWS,FAMILIES,raw_only_inputs,raw_sha256,
)
STAGE="NRA3_S62_TRUE_ORIGINAL43M_GT_NEW32_JOINT_EVENT_TARGET2166_FIRSTREAD"
NATIVE="NRA3_S62_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
PROV="NRA3_S62_GENUINE43M_GT_EVENTFIRST32_OFFSET_PROVENANCE.json"
CAPTURE="NRA3_S62_GENUINE43M_GT_EVENTFIRST32_NATIVE_J7_REPORT.json"
S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
S57="NRA3_S57_EVENTFIRST_SPENT96_ONLY_231FP16_UNACCEPTED.safetensors"
S59="NRA3_S59_REAL43M_GT_OLD152_NATIVE320_SEMANTIC2046_FP16.safetensors"
S60="NRA3_S60_ORIGINAL43M_GT_OLD184_TRICLASS2776_FP16.safetensors"
S61="NRA3_S61_REAL43M_GT_SPENT216_PAID2752_PAIR_NONE.safetensors"
S62="NRA3_S62_REAL43M_GT_SPENT248_JOINT_EVENT_TARGET2166_FP16.safetensors"
FIT59="NRA3_S59_OLD152_TRAIN_NATIVE2046_FP16_MANIFEST.json"
FIT60="NRA3_S60_OLD184_TRAIN_TRICLASS2776_MANIFEST.json"
FIT61="NRA3_S61_SPENT216_PAIR_NONE2752_TRAIN_MANIFEST.json"
FIT62="NRA3_S62_OLD248_JOINT2166_TRAIN_MANIFEST.json"
REPORT="NRA3_S62_TRUE43M_GT_FIRST_NEW32_JOINT2166_REPORT.json"

def require(ok,msg):
    if not bool(ok):raise ValueError("S62 original43M GT first NEW32: "+msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def measure(y,g):
    tp=int((y&g).sum());fp=int((y&~g).sum());fn=int((~y&g).sum())
    den=2*tp+fp+fn
    return {"TP":tp,"FP":fp,"FN":fn,
            "precision":tp/(tp+fp) if tp+fp else 0.,
            "recall":tp/(tp+fn) if tp+fn else 0.,
            "F1":(2*tp/den) if den else 0.}

def main():
    p=argparse.ArgumentParser()
    for key in ("native","prov","capture","s26","s57",
                "s59","fit59","s60","fit60","s61","fit61",
                "s62","fit62","out"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"S62 physical first NEW32 once only, refuse overwrite")
    files=(a.native,a.prov,a.capture,a.s26,a.s57,
           a.s59,a.fit59,a.s60,a.fit60,a.s61,a.fit61,a.s62,a.fit62)
    names=(NATIVE,PROV,CAPTURE,S26,S57,S59,FIT59,S60,FIT60,S61,FIT61,S62,FIT62)
    require(all(p.is_file() and p.name==n for p,n in zip(files,names)),
            "original archived model/manifest immutable names incorrect")
    prov=json.loads(a.prov.read_text())
    cap=json.loads(a.capture.read_text())
    m59=json.loads(a.fit59.read_text())
    m60=json.loads(a.fit60.read_text())
    m61=json.loads(a.fit61.read_text())
    m62=json.loads(a.fit62.read_text())
    require(
        cap["stage"]=="NRA3_S62_GENUINE43M_GT_EVENTFIRST32_NATIVE_CPU_FIRST_PASS"
        and cap["S62_FROZEN_HELD32_PLAINTEXT_SHA256"]==raw_sha256()
        and cap["S62_HELD32_SOURCE_COMMIT"]==
            "6062a9a7f432a2e153dda85361c0e262e824f71b"
        and cap["S62_HELD32_SOURCE_BLOB"]==
            "ee9d8f988264bb54b7d7b0abc45365e8489fbe24"
        and cap["all_616_native_CPU_generated_token_digests_unchanged"]
        and cap["original_456_full_EOS"]==456 and cap["original_new128_full_EOS"]==128
        and cap["actual_43M_CPU_12_transformer_layers"]
        and cap["GT_residuals_before_readport_in_same_active_S93_scope"]
        and cap["actual_genuine_43M_CPU_new_HELD32_prefills"]==32
        and cap["S62_HELD_manual_gold_NOT_IMPORTED_into_model_capture"]
        and cap["NRA3_output_gate"]==0 and not cap["NRA3_candidate_accepted"]
        and prov["stage"]=="NRA3_S62_GENUINE43M_GT_EVENTFIRST32_BPE_SOURCE_ONLY"
        and prov["frozen_S62_HELD32_raw_source_sha256"]==raw_sha256()
        and prov["original_616_J7_replayed_before_read_only_quotes"]
        and prov["derived_REAL43M_WORD339_SHA256"]==sha(a.native)
        and prov["new_BPE_fractional_aligned_count"]==32
        and prov["S62_HUMAN_GOLD_NOT_IMPORTED_DURING_ANY_CAPTURE_STEP"],
        "S62 true original43M GT 616 protected J7 and raw32 gold-blind capture fail")
    require(
        m62["stage"]=="NRA3_S62_GENUINE43M_GT_SPENT248_JOINT_EVENT_TARGET_BEAM2166_TRAIN"
        and m62["new_S62_joint2166_FP16_SHA256"]==sha(a.s62)
        and m62["new_S62_paid_fp16_parameters"]==PAID==2166
        and m62["original_S26_PAID12384_SHA256"]==sha(a.s26)
        and m62["frozen_S57_PAID231_SHA256"]==sha(a.s57)
        and m62["frozen_S59_PAID2046_SHA256"]==sha(a.s59)
        and m62["frozen_S61_PAID2752_SHA256"]==sha(a.s61)
        and m62["S62_NEW32_RAW_AND_GOLD_NOT_IMPORTED_IN_TRAIN"]
        and m62["output_gate"]==0 and not m62["candidate_accepted"]
        and not m62["activation_allowed"]
        and m59["new_S59_semantic_paid2046_sha256"]==sha(a.s59)
        and m60["new_S60_paid2776_sha256"]==sha(a.s60)
        and m61["new_S61_paid2752_SHA256"]==sha(a.s61),
        "S62 fixed pretrained old248 FP16 or old originals ancestry corrupted")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    native=load_file(str(a.native),device="cpu")
    require(set(native)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S62_HELD32"},
        "original S62 new32 archive must NOT contain gold tensors or labels")
    x=native["original_real43m_gt_word339_READONLY"].detach()
    valid=native["original_native_word_valid_mask"].detach()
    unsafe=native["native_BPE_causally_unsafe_word_mask"].detach()
    require(x.ndim==3 and x.shape[0]==32 and x.shape[-1]==339
            and x.dtype==torch.float32 and
            valid.shape==unsafe.shape==x.shape[:2]
            and valid.dtype==unsafe.dtype==torch.bool
            and native["source_row_indices_in_frozen_S62_HELD32"].tolist()
                ==list(range(32)),
            "S62 original 43M+GT native339 BPE word shape drift")
    input_hash=tensor_sha(x,valid,unsafe)
    old26=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(a.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    old59=S59NativeEventSemantic.from_fp16(
        load_file(str(a.s59),device="cpu")["S59_NATIVE320_SEMANTIC2046_FP16"])
    old60=S60Joint3.from_fp16(
        load_file(str(a.s60),device="cpu")["S60_SOURCE_LOCAL_TRICLASS2776_FP16"])
    old61=S61NativePairNull.from_fp16(
        load_file(str(a.s61),device="cpu")["S61_REAL43M_GT_PAIR_NULL_POLAR2752_FP16"])
    new62=S62JointEventRecipient.from_fp16(
        load_file(str(a.s62),device="cpu")["S62_JOINT_EVENT_TARGET2166_FP16"])
    models=(old26,old57,old59,old60,old61,new62)
    require(all(not p.requires_grad for m in models for p in m.parameters())
            and all(m.output_gate==0 and not m.candidate_accepted
                    and not m.activation_allowed for m in models),
            "old S26/S57/S59/S60/S61 and new S62 must remain frozen gated")
    saved_hash=[tensor_sha(*(p.detach() for p in m.parameters())) for m in models]
    with torch.inference_mode():
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
        event=gold_blind_eventfirst_features(x,obs,raw_only_inputs())
        sem=semantic_features(x,event)
        fused=source_local_scope(sem,raw_only_inputs())
        prior57=tuple(q.detach() for q in old57(event))
        s57=old57.decode(event)
        s59=old59.decode(sem,prior57)
        s60=old60.decode(fused,prior57,old59)
        s61=old61.decode(fused,prior57,old59)
        beam=source_only_frozen_beam(fused,prior57,old59,old61)
        s62=new62.decode(beam)
        unchanged=input_hash==tensor_sha(x,valid,unsafe) and all(
            tensor_sha(*(p.detach() for p in m.parameters()))==h
            for m,h in zip(models,saved_hash))
    require(unchanged and s62["output_gate"]==0 and not s62["candidate_accepted"]
            and int(s62["directed_recipient"].sum())<=32
            and not bool((s62["directed_recipient"]&~event.valid_pair).any())
            and not bool((s62["directed_rejected"]&
                          ~s62["directed_recipient"]).any()),
            "S62 frozen first forward original43M GT mutation/unsafe edge violation")
    # -------- ONLY HERE: first read of separately frozen NEW32 MANUAL GOLD ----------
    from research.nra3_s20.s62_new32_separate_manual_gold_NEVER_TRAIN import (
        held_gold32,gold_sha256,
    )
    gold=held_gold32()
    require(len(gold)==len(SOURCE_ROWS)==32,"S62 precommitted raw/gold length mismatch")
    truth=torch.zeros_like(s62["directed_recipient"])
    rejected=torch.zeros_like(truth)
    true_event=torch.zeros(32,dtype=torch.long)
    rows=[]
    for i,(raw,g) in enumerate(zip(SOURCE_ROWS,gold)):
        require(raw[0]==g.row_id,"S62 newly frozen gold row identity changed")
        words=raw_surface_grammar(raw[2])
        require(len(words)==int(valid[i].sum()),"S62 true model native word offset drift")
        def get_word(value):
            matches=[j for j,t in enumerate(words) if t.form==value]
            require(len(matches)==1,
                    "S62 new gold predicate or receiver ambiguous/absent in native text")
            return matches[0]
        src=get_word(g.main_predicate)
        true_event[i]=src
        dst=None
        if g.transferred_recipient is not None:
            dst=get_word(g.transferred_recipient)
            truth[i,src,dst]=True
            if g.directed_transfer_polarity=="REJECTED":
                rejected[i,src,dst]=True
        rows.append({
            "id":g.row_id,"family":raw[1],
            "gold_predicate":g.main_predicate,
            "gold_recipient":g.transferred_recipient,
            "gold_rejected":g.directed_transfer_polarity,
            "gold_predicate_eligible":bool(event.candidate_event[i,src]),
            "gold_recipient_BPE_safe":bool(event.valid_pair[i,src,dst])
                                  if dst is not None else None,
            "gold_predicate_inside_precommitted_BEAM8":bool(
                ((beam.sources[i]==src)&beam.allowed[i]).any()),
            "S61_predicted_main":words[int(s61["chosen_event"][i])].form,
            "S62_predicted_main":words[int(s62["chosen_event"][i])].form,
            "S62_link_count":int(s62["directed_recipient"][i].sum()),
            "S62_rejected_count":int(s62["directed_rejected"][i].sum()),
        })
    require(int(truth.sum())==24 and int(rejected.sum())==8,
            "S62 NEW32 precommitted manual 24 directed/8 rejected totals drift")
    models_pred={
        "S57":s57,"S59_native2046":s59,"S60_local2776":s60,
        "S61_pair_null2752":s61,"S62_joint_event_target2166":s62,
    }
    metrics={}
    for name,pred in models_pred.items():
        metrics[name+"_RECIPIENT"]=measure(pred["directed_recipient"],truth)
        metrics[name+"_REJECTED"]=measure(pred["directed_rejected"],rejected)
        chosen=(pred["chosen_outer_event_index"] if name=="S57"
                else pred["chosen_event"])
        metrics[name+"_MAIN_EVENT"]={
            "correct":int((chosen==true_event).sum()),
            "total":32,"accuracy":float((chosen==true_event).float().mean()),
        }
    groups={}
    for family in FAMILIES:
        ids=torch.tensor([j for j,raw in enumerate(SOURCE_ROWS) if raw[1]==family])
        require(len(ids)==8,"S62 new32 4-family original source drift")
        groups[family]={}
        for name,pred in models_pred.items():
            groups[family][name+"_RECIPIENT"]=measure(
                pred["directed_recipient"][ids],truth[ids])
            groups[family][name+"_REJECTED"]=measure(
                pred["directed_rejected"][ids],rejected[ids])
    report={
        "stage":STAGE,
        "original_43M_GT_12layers_616_J7_TRUE_CPU_baseline_replayed_before_NEW32":True,
        "616_J7_preserved_original_ONLY_not_activated_S62_candidate":True,
        "original_Calibri43M_GT_and_old_paid_SHA_unchanged":unchanged,
        "precommitted_S62_NEW32_RAW_commit":"6062a9a7f432a2e153dda85361c0e262e824f71b",
        "precommitted_S62_NEW32_RAW_blob":"ee9d8f988264bb54b7d7b0abc45365e8489fbe24",
        "precommitted_S62_NEW32_GOLD_commit":"93c1df20cda67cd40e014539c22a99e21115ae0a",
        "precommitted_S62_NEW32_GOLD_blob":"2a483857ea0bd3f2bd6d4daae0c4923adf3dbd20",
        "new_S62_raw_sha256":raw_sha256(),
        "new_S62_gold_sha256":gold_sha256(),
        "new_S62_paid2166_sha256":sha(a.s62),
        "new_S62_original43M_GT_NATIVE339_sha256":sha(a.native),
        "first_NEW32_manual_GOLD_imported_ONLY_AFTER_all_prediction_forward":True,
        "all_NEW32_now_irrevocably_spent_after_first_read":True,
        "author_written_4x8_challenge_NOT_population_independent_EVAL":True,
        "precommitted_frozen_S59_top8_true_event_coverage":sum(
            q["gold_predicate_inside_precommitted_BEAM8"] for q in rows),
        "new_S62_beam8_limit_misses_are_unrecoverable":True,
        "new_S62_no_target_abstentions":s62["no_target_abstentions"],
        "gold_predicate_BPE_eligible":sum(q["gold_predicate_eligible"] for q in rows),
        "gold_recipient_BPE_safe":sum(q["gold_recipient_BPE_safe"] is True for q in rows),
        "NEW32_FIRST_READ_all_frozen_models_metrics":metrics,
        "NEW32_FIRST_READ_per_family":groups,
        "NEW32_FIRST_READ_row_diagnostics":rows,
        "no_new32_optimizer_model_selection_threshold_tuning":True,
        "protected_S38_S47_S48_EVAL32_EVAL128_not_touched":True,
        "active_S62_J7_bpp_CPU_CB59_streaming_NOT_certified":True,
        "full_context_prefill_not_online_causal_certified":True,
        "LIVE_Render_Supabase_Jessica_APK_original_UNCHANGED":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    a.out.mkdir(parents=True,exist_ok=False)
    dest=a.out/REPORT
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
                    encoding="utf-8")
    print("S62_REAL_ORIGINAL43M_GT_FIRST_NEW32_JOINT_EVENT_TARGET",
          json.dumps(metrics,sort_keys=True),flush=True)
    print("S62_REAL_ORIGINAL43M_GT_FIRST_NEW32_BEAM_DIAGNOSTICS",
          json.dumps({"precommitted_beam8_true_event":
                      report["precommitted_frozen_S59_top8_true_event_coverage"],
                      "gold_event_BPE_eligible":report["gold_predicate_BPE_eligible"],
                      "gold_pair_safe":report["gold_recipient_BPE_safe"],
                      "no_target":report["new_S62_no_target_abstentions"],
                      "gate":0},sort_keys=True),flush=True)
    print("S62_NEW32_NOW_PERMANENTLY_SPENT_ACTIVE_CANDIDATE_NOT_ACCEPTED",flush=True)

if __name__=="__main__":main()
