"""S59 genuine original Calibri43M+GT NEW32 FIRST-READ semantic evaluation.

Only precommitted S59 RAW32 enters true original CPU protected616 J7
gold-blind capture. Model weights/feature normalization/threshold fixed
before first NEW32 prediction. Original S26, S57, old S58 83 and NEW S59
2046 produce predictions and SHA-lock BEFORE the separately precommitted
S59 human gold module is imported. Author-built challenge, not unbiased
representative population benchmark. NEW32 becomes permanently spent.
NO MERGE, NO LIVE, no original/core/GT modifications.
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
from research.nra3_s20.s58_trainonly_event_transfer_residual83 import (
    S58EventTransferResidual,
)
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    S59NativeEventSemantic,semantic_features,P,
)
from research.nra3_s20.s59_new32_first_frozen_raw_only_NEVER_TRAIN import (
    SOURCE_ROWS,FAMILIES,raw_only_inputs,raw_sha256,
)

STAGE="NRA3_S59_TRUE_ORIGINAL43M_GT_NEW32_FIRSTREAD_NATIVE_SEMANTIC2046"
NATIVE="NRA3_S59_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
PROV="NRA3_S59_GENUINE43M_GT_EVENTFIRST32_OFFSET_PROVENANCE.json"
CAP="NRA3_S59_GENUINE43M_GT_EVENTFIRST32_NATIVE_J7_REPORT.json"
S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
S57="NRA3_S57_EVENTFIRST_SPENT96_ONLY_231FP16_UNACCEPTED.safetensors"
S58="NRA3_S58_OLD96_TRAIN_ONLY_EVENT_TRANSFER_RESIDUAL83_FP16.safetensors"
S59="NRA3_S59_REAL43M_GT_OLD152_NATIVE320_SEMANTIC2046_FP16.safetensors"
FIT59="NRA3_S59_OLD152_TRAIN_NATIVE2046_FP16_MANIFEST.json"
FIT57="NRA3_S57_EVENTFIRST_SPENT96_231FP16_TRAIN_REPORT.json"
FIT58="NRA3_S58_TRAIN_ONLY_OLD96_RESIDUAL83_MANIFEST.json"

def require(ok,msg):
    if not bool(ok):raise ValueError("S59 true43M GT new32 firstread: "+msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(pred,gold):
    tp=int((pred&gold).sum());fp=int((pred&~gold).sum())
    fn=int((~pred&gold).sum());d=2*tp+fp+fn
    return {
        "TP":tp,"FP":fp,"FN":fn,
        "precision":tp/(tp+fp) if tp+fp else 0.,
        "recall":tp/(tp+fn) if tp+fn else 0.,
        "F1":2*tp/d if d else 0.,
    }

def main():
    parser=argparse.ArgumentParser()
    for n in ("native","prov","capture","s26","s57","fit57","s58","fit58",
              "s59","fit59","out"):
        parser.add_argument("--"+n,type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(),"refuse overwrite FIRST S59 independent challenge result")
    files=(args.native,args.prov,args.capture,args.s26,args.s57,args.fit57,
           args.s58,args.fit58,args.s59,args.fit59)
    names=(NATIVE,PROV,CAP,S26,S57,FIT57,S58,FIT58,S59,FIT59)
    require(all(f.is_file() and f.name==name for f,name in zip(files,names)),
            "unexpected true source archive or paid checkpoint")
    prov=json.loads(args.prov.read_text())
    cap=json.loads(args.capture.read_text())
    f57=json.loads(args.fit57.read_text())
    f58=json.loads(args.fit58.read_text())
    f59=json.loads(args.fit59.read_text())
    require(
        cap["stage"]=="NRA3_S59_GENUINE43M_GT_EVENTFIRST32_NATIVE_CPU_FIRST_PASS"
        and cap["S59_FROZEN_HELD32_PLAINTEXT_SHA256"]==raw_sha256()
        and cap["S59_HELD32_SOURCE_COMMIT"]==
            "6b31095de76737b43f09f00c87e985e497a5b01e"
        and cap["S59_HELD32_SOURCE_BLOB"]==
            "1d36b159b58288991c9efb6529c389fda19468c0"
        and cap["all_616_native_CPU_generated_token_digests_unchanged"]
        and cap["original_456_full_EOS"]==456
        and cap["original_new128_full_EOS"]==128
        and cap["actual_43M_CPU_12_transformer_layers"]
        and cap["GT_residuals_before_readport_in_same_active_S93_scope"]
        and cap["actual_genuine_43M_CPU_new_HELD32_prefills"]==32
        and cap["S59_HELD_manual_gold_NOT_IMPORTED_into_model_capture"]
        and cap["NRA3_output_gate"]==0
        and not cap["NRA3_candidate_accepted"]
        and prov["stage"]=="NRA3_S59_GENUINE43M_GT_EVENTFIRST32_BPE_SOURCE_ONLY"
        and prov["frozen_S59_HELD32_raw_source_sha256"]==raw_sha256()
        and prov["original_616_J7_replayed_before_read_only_quotes"]
        and prov["derived_REAL43M_WORD339_SHA256"]==sha(args.native)
        and prov["new_BPE_fractional_aligned_count"]==32
        and prov["S59_HUMAN_GOLD_NOT_IMPORTED_DURING_ANY_CAPTURE_STEP"],
        "real original43M+GT 616 J7 and source goldblind native provenance mismatch")
    require(
        f59["stage"]==
           "NRA3_S59_FROZEN_ORIGINAL43M_GT_OLD152_NATIVE320_TRAIN2046_FP16"
        and f59["new_semantic_FP16_parameters"]==P==2046
        and f59["new_S59_semantic_paid2046_sha256"]==sha(args.s59)
        and f59["original_physical_S26_archive_SHA256"]==sha(args.s26)
        and f59["old_S57_paid231_archive_SHA256"]==sha(args.s57)
        and f59["never_imported_S59_new32_RAW_or_GOLD_into_training"]
        and f59["original_Calibri43M_GT_S26_S57_unchanged"]
        and f59["output_gate"]==0 and not f59["candidate_accepted"]
        and not f59["activation_allowed"]
        and f57["new_event_first_FP16_weights_SHA256"]==sha(args.s57)
        and f58["new_S58_paid83_SHA256"]==sha(args.s58),
        "old TRAIN/paid2046 physical SHA frozen BEFORE first NEW32 mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    archive=load_file(str(args.native),device="cpu")
    require(set(archive)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S59_HELD32"},
        "new S59 original native archive unexpectedly contains labels")
    x=archive["original_real43m_gt_word339_READONLY"].detach()
    valid=archive["original_native_word_valid_mask"].detach()
    unsafe=archive["native_BPE_causally_unsafe_word_mask"].detach()
    require(x.ndim==3 and x.shape[0]==32 and x.shape[-1]==339
            and valid.shape==unsafe.shape==x.shape[:2]
            and archive["source_row_indices_in_frozen_S59_HELD32"].tolist()
                ==list(range(32))
            and x.dtype==torch.float32
            and valid.dtype==unsafe.dtype==torch.bool,
            "new original real43M native word offsets and masks invalid")
    base_sha=tensor_sha(x,valid,unsafe)
    old26=EventGroundedInversionGraph.from_physical(args.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(args.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    old58=S58EventTransferResidual.from_fp16(
        load_file(str(args.s58),device="cpu")["S58_EVENT_TRANSFER_RESIDUAL83_FP16"])
    new59=S59NativeEventSemantic.from_fp16(
        load_file(str(args.s59),device="cpu")["S59_NATIVE320_SEMANTIC2046_FP16"])
    models=(old26,old57,old58,new59)
    require(all(not p.requires_grad for m in models for p in m.parameters())
            and old26.parent.output_gate==0
            and all(m.output_gate==0 and not m.candidate_accepted
                    for m in models),
            "frozen original S26 and S57 S58 S59 model unexpectedly active")
    paid_sha=[tensor_sha(*(p.detach() for p in m.parameters()))
              for m in models]
    with torch.inference_mode():
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
        features=gold_blind_eventfirst_features(x,obs,raw_only_inputs())
        native_semantic=semantic_features(x,features)
        frozen_s57=tuple(z.detach() for z in old57(features))
        baseline57=old57.decode(features)
        baseline58=old58.decode(features,frozen_s57)
        result59=new59.decode(native_semantic,frozen_s57)
        unchanged=(base_sha==tensor_sha(x,valid,unsafe) and all(
            tensor_sha(*(p.detach() for p in m.parameters()))==s
            for m,s in zip(models,paid_sha)))
    require(unchanged and all(
            not result59["directed_recipient"].is_floating_point()
            for _ in range(1)),
            "frozen original native and paid readouts modified in FIRST NEW32")
    require(int(result59["directed_recipient"].sum())<=32
            and not bool((result59["directed_recipient"]&~features.valid_pair).any())
            and not bool((result59["directed_rejected"]&
                          ~result59["directed_recipient"]).any()),
            "S59 emitted multiple unsafe or quoted directed links")
    # ********* FIRST READ OF NEW32 MANUAL GOLD HAPPENS ONLY NOW *********
    # No model/threshold update, training, or optimizer beyond this point.
    from research.nra3_s20.s59_new32_first_frozen_manual_gold_NEVER_TRAIN import (
        held_gold32,gold_sha256,
    )
    gold=held_gold32()
    require(len(gold)==len(SOURCE_ROWS)==32 and len(raw_only_inputs())==32,
            "frozen S59 gold missing at manual one-time evaluation")
    truth=torch.zeros_like(result59["directed_recipient"])
    rejected=torch.zeros_like(truth)
    truth_event=torch.zeros(32,dtype=torch.long)
    rows=[]
    for row,(source,manual) in enumerate(zip(SOURCE_ROWS,gold)):
        require(source[0]==manual.row_id,"NEW32 raw-only vs later frozen GOLD order changed")
        words=raw_surface_grammar(source[2])
        require(len(words)==int(valid[row].sum()),"original word/Gold raw Unicode mismatch")
        def find(form):
            hits=[i for i,t in enumerate(words) if t.form==form]
            require(len(hits)==1,"gold predicate or receiver not unique in native words")
            return hits[0]
        src=find(manual.main_predicate)
        truth_event[row]=src
        dest=None
        if manual.transferred_recipient is not None:
            dest=find(manual.transferred_recipient)
            truth[row,src,dest]=True
            if manual.directed_transfer_polarity=="REJECTED":
                rejected[row,src,dest]=True
        rows.append({
            "id":manual.row_id,"family":source[1],
            "gold_source":manual.main_predicate,
            "gold_recipient":manual.transferred_recipient,
            "gold_rejected":manual.directed_transfer_polarity,
            "old57_source":words[int(baseline57["chosen_outer_event_index"][row])].form,
            "old58_source":words[int(baseline58["chosen_event"][row])].form,
            "S59_source":words[int(result59["chosen_event"][row])].form,
            "S59_source_correct":int(result59["chosen_event"][row])==src,
            "gold_event_eligible":bool(features.candidate_event[row,src]),
            "gold_pair_safe":bool(features.valid_pair[row,src,dest])
                             if dest is not None else None,
            "old57_proposed_recipient":int(baseline57["directed_recipient"][row].sum()),
            "S58_proposed_recipient":int(baseline58["directed_recipient"][row].sum()),
            "S59_proposed_recipient":int(result59["directed_recipient"][row].sum()),
            "S59_proposed_REJECTED":int(result59["directed_rejected"][row].sum()),
        })
    require(int(truth.sum())==16 and int(rejected.sum())==8,
            "S59 precommitted first NEW32 true labels must be exactly 16/8")
    totals={}
    preds={
        "S57":baseline57,
        "S58":baseline58,
        "S59_native2046":result59,
    }
    for name,data in preds.items():
        totals[name+"_RECIPIENT"]=metric(data["directed_recipient"],truth)
        totals[name+"_REJECTED"]=metric(data["directed_rejected"],rejected)
        chosen=(data["chosen_outer_event_index"] if name=="S57"
                else data["chosen_event"])
        totals[name+"_MAIN_EVENT"]={
            "correct":int((chosen==truth_event).sum()),"total":32,
            "accuracy":float((chosen==truth_event).float().mean()),
        }
    families={}
    for family in FAMILIES:
        idx=torch.tensor([i for i,x in enumerate(SOURCE_ROWS) if x[1]==family])
        require(len(idx)==8,"S59 precommitted NEW32 family 8-case drift")
        families[family]={}
        for name,pred in preds.items():
            families[family][name+"_RECIPIENT"]=metric(
                pred["directed_recipient"].index_select(0,idx),
                truth.index_select(0,idx))
            families[family][name+"_REJECTED"]=metric(
                pred["directed_rejected"].index_select(0,idx),
                rejected.index_select(0,idx))
    report={
        "stage":STAGE,
        "first_read_NEW32_manual_gold_imported_AFTER_all_S57_S58_S59_predictions":True,
        "S59_NEW32_now_PERMANENTLY_spent":True,
        "author_written_structural_challenge_not_random_population_EVAL":True,
        "precommitted_RAW_commit":"6b31095de76737b43f09f00c87e985e497a5b01e",
        "precommitted_RAW_blob":"1d36b159b58288991c9efb6529c389fda19468c0",
        "precommitted_GOLD_commit":"1e6344eb14d3f388048657de18a35472196e45d4",
        "precommitted_GOLD_blob":"c8a519a94daf470c274c783bd708ff3053071eac",
        "raw_source_sha256":raw_sha256(),
        "manual_gold_sha256":gold_sha256(),
        "true_original43M_GT_native339_SHA256":sha(args.native),
        "previous_real_original616_J7_replayed_before_new32_CPU":True,
        "original_current_616J7_unchanged_base_only_NOT_active_S59":True,
        "original_GT_and_paid_model_hash_unchanged":unchanged,
        "original_S26_paid12384_SHA256":sha(args.s26),
        "original_S57_paid231_SHA256":sha(args.s57),
        "old_S58_paid83_SHA256":sha(args.s58),
        "new_S59_paid2046_SHA256":sha(args.s59),
        "gold_eligible_main_event":sum(r["gold_event_eligible"] for r in rows),
        "gold_safe_directed_recipient":sum(r["gold_pair_safe"] is True for r in rows),
        "new_S59_abstain_nontransfer":result59["abstain_nontransfer"],
        "new_S59_abstain_unsafe":result59["abstain_no_safe_target"],
        "all_FIRST_NEW32_directed_metrics":totals,
        "all_FIRST_NEW32_four_8case_families":families,
        "all_FIRST_NEW32_row_diagnostics":rows,
        "no_model_selection_or_new32_training_after_gold":True,
        "S59_full_prefill_original43M_is_not_online_causal_certified":True,
        "S59_active_616J7_bpp_CB59_NOT_certified":True,
        "no_protected_S38_S47_S48_HELD26_or_EVAL32_128":True,
        "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    args.out.mkdir(parents=True,exist_ok=False)
    result=args.out/"NRA3_S59_TRUE43M_GT_FIRST_NEW32_MANUAL_REPORT.json"
    result.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,
                                 indent=2)+"\n",encoding="utf-8")
    print("S59_REAL_ORIGINAL43M_GT_NEW32_SEMANTIC_FIRSTREAD",
          json.dumps(totals,ensure_ascii=False,sort_keys=True),flush=True)
    print("S59_TRUE43M_GT_NEW32_GOLD_AFTER_FORWARD_NO_ACCEPTANCE",
          json.dumps({"held32_now_spent":True,
                      "original_J7_unchanged_base_only":True,
                      "output_gate":0},sort_keys=True),flush=True)

if __name__=="__main__": main()
