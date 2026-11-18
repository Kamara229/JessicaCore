"""S58 FIRST-READ NEW32 author-built challenge on genuine original43M+GT CPU.

This first-read evaluator runs only after source-only CPU native capture and
after S58 83FP16 was physically frozen on OLD96 (no new32 text or gold).
Frozen S26, S57 231FP16, S58 no-param guard and S58 83FP16 produce
all new32 predictions BEFORE manual gold module is imported.
New32 is author-written challenge, NOT independent population benchmark.
A scored new32 becomes SPENT and may never be called fresh HELD again.
Research gate remains ZERO; never merge, activate or deploy.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from safetensors.torch import load_file
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s26_predicted_event_inversion_rescue import EventGroundedInversionGraph
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    S52QuoteAwareFrozenPaidGraphBridge, tensor_sha,
)
from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import (
    S57EventFirstRecipient, gold_blind_eventfirst_features,
)
from research.nra3_s20.s58_eventfirst_selective_abstention import (
    bridge_frozen_s57, selective_abstain,
)
from research.nra3_s20.s58_trainonly_event_transfer_residual83 import (
    S58EventTransferResidual,
)
from research.nra3_s20.s58_new32_preregistered_raw_only_do_not_train import (
    SOURCE_ROWS, FAMILIES, raw_only_inputs, raw_sha256,
)

STAGE="NRA3_S58_FIRST_NEW32_TRUE_ORIGINAL43M_GT_TRAIN96_PAID83"
NATIVE="NRA3_S58_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
PROV="NRA3_S58_GENUINE43M_GT_EVENTFIRST32_OFFSET_PROVENANCE.json"
CAP="NRA3_S58_GENUINE43M_GT_EVENTFIRST32_NATIVE_J7_REPORT.json"
S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
S57="NRA3_S57_EVENTFIRST_SPENT96_ONLY_231FP16_UNACCEPTED.safetensors"
FIT="NRA3_S57_EVENTFIRST_SPENT96_231FP16_TRAIN_REPORT.json"
S58="NRA3_S58_OLD96_TRAIN_ONLY_EVENT_TRANSFER_RESIDUAL83_FP16.safetensors"
FIT58="NRA3_S58_TRAIN_ONLY_OLD96_RESIDUAL83_MANIFEST.json"


def require(ok,msg):
    if not bool(ok):
        raise ValueError("S58 fresh32 research-only: "+msg)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def counts(pred,gold):
    tp=int((pred&gold).sum())
    fp=int((pred&~gold).sum())
    fn=int((~pred&gold).sum())
    return {"TP":tp,"FP":fp,"FN":fn,
            "precision":tp/(tp+fp) if tp+fp else 0.0,
            "recall":tp/(tp+fn) if tp+fn else 0.0,
            "F1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}


def main():
    p=argparse.ArgumentParser()
    for name in ("native","provenance","capture","s26","s57","fit","s58","s58-fit","out"):
        p.add_argument("--"+name,required=True,type=Path)
    args=p.parse_args()
    require(not args.out.exists(),"cannot overwrite first-read NEW32 result")
    require(all(x.is_file() for x in (
        args.native,args.provenance,args.capture,args.s26,args.s57,args.fit,args.s58,args.s58_fit)),
        "missing physically archived frozen model source")
    require((args.native.name,args.provenance.name,args.capture.name,
             args.s26.name,args.s57.name,args.fit.name,args.s58.name,args.s58_fit.name)==
            (NATIVE,PROV,CAP,S26,S57,FIT,S58,FIT58),"unexpected archive name")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    provenance=json.loads(args.provenance.read_text())
    capture=json.loads(args.capture.read_text())
    fit=json.loads(args.fit.read_text())
    fit58=json.loads(args.s58_fit.read_text())
    require(fit58["stage"]=="NRA3_S58_OLD96_REAL43M_GT_TRAIN_ONLY_EVENT_TRANSFER83"
            and fit58["new_S58_paid83_SHA256"]==sha(args.s58)
            and fit58["original_S26_FP16_SHA256"]==sha(args.s26)
            and fit58["old_original_S57_paid231_SHA256"]==sha(args.s57)
            and fit58["new_S58_parameters"]==83
            and fit58["new_S58_32_RAW_AND_GOLD_NEVER_IMPORTED"]
            and fit58["S57_spent24_dev_NOT_IMPORTED_IN_TRAIN"]
            and fit58["output_gate"]==0 and not fit58["candidate_accepted"],
            "S58 physically frozen old96-only model provenance failed")
    require(provenance["stage"]=="NRA3_S58_GENUINE43M_GT_EVENTFIRST32_BPE_SOURCE_ONLY"
            and provenance["derived_REAL43M_WORD339_SHA256"]==sha(args.native)
            and provenance["frozen_S58_HELD32_raw_source_sha256"]==raw_sha256()
            and provenance["original_616_J7_replayed_before_read_only_quotes"]
            and provenance["S58_HUMAN_GOLD_NOT_IMPORTED_DURING_ANY_CAPTURE_STEP"]
            and capture["stage"]=="NRA3_S58_GENUINE43M_GT_EVENTFIRST32_NATIVE_CPU_FIRST_PASS"
            and capture["all_616_native_CPU_generated_token_digests_unchanged"]
            and capture["actual_genuine_43M_CPU_new_HELD32_prefills"]==32
            and capture["S58_HELD32_SOURCE_COMMIT"]=="82596c03eb5a519f2409fd1f8eecddfb9df97cc6"
            and capture["S58_HELD32_SOURCE_BLOB"]=="0bdab37c6e3a634fd901e0e0c1d37bb36bcadee1"
            and capture["S58_HELD_manual_gold_NOT_IMPORTED_into_model_capture"]
            and capture["NRA3_output_gate"]==0 and not capture["NRA3_candidate_accepted"]
            and fit["new_event_first_FP16_weights_SHA256"]==sha(args.s57)
            and fit["new_event_first_FP16_model_parameters"]==231
            and fit["new_S57_first_HELD24_raw_and_manual_gold_NEVER_IMPORTED"],
            "S58 new32 genuine original CPU 43M+GT and paid archives provenance invalid")
    native=load_file(str(args.native),device="cpu")
    require(set(native)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S58_HELD32"}, "unexpected original model feature keys")
    x=native["original_real43m_gt_word339_READONLY"].detach()
    valid=native["original_native_word_valid_mask"].detach()
    unsafe=native["native_BPE_causally_unsafe_word_mask"].detach()
    require(x.ndim==3 and x.shape[0]==32 and x.shape[2]==339
            and valid.shape==unsafe.shape==x.shape[:2]
            and valid.dtype==unsafe.dtype==torch.bool
            and native["source_row_indices_in_frozen_S58_HELD32"].tolist()==list(range(32)),
            "native word alignment and safe masks drift")
    base_sha=tensor_sha(x,valid,unsafe)
    old26=EventGroundedInversionGraph.from_physical(args.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(args.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    new83=S58EventTransferResidual.from_fp16(
        load_file(str(args.s58),device="cpu")["S58_EVENT_TRANSFER_RESIDUAL83_FP16"])
    require(not any(p.requires_grad for model in (old26,old57,new83)
                    for p in model.parameters()),"old paid parameters unexpectedly unfrozen")
    model_sha=tensor_sha(*(p.detach() for p in old57.parameters()))
    new_sha=tensor_sha(*(p.detach() for p in new83.parameters()))
    with torch.inference_mode():
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
        features=gold_blind_eventfirst_features(x,obs,raw_only_inputs())
        old=old57.decode(features)
        new=selective_abstain(bridge_frozen_s57(old57,features))
        new83_pred=new83.decode(features,tuple(z.detach() for z in old57(features)))
        base_unmodified=(base_sha==tensor_sha(x,valid,unsafe))
        weights_unmodified=(model_sha==tensor_sha(*(p.detach() for p in old57.parameters())))
    require(base_unmodified and weights_unmodified
            and new_sha==tensor_sha(*(p.detach() for p in new83.parameters()))
            and old57.output_gate==0 and new83.output_gate==0
            and new["output_gate"]==0 and not new["candidate_accepted"]
            and not new["activation_allowed"],"read-only guard modified a frozen model")
    require(int(new83_pred["directed_recipient"].sum())<=32
            and not bool((new83_pred["directed_rejected"]&
                          ~new83_pred["directed_recipient"]).any()),
            "new paid83 model emitted unsafe directed pairs")
    require(int(new["directed_recipient"].sum())<=32
            and not bool((new["directed_rejected"]&~new["directed_recipient"]).any()),
            "S58 emitted unsafe or too many directed links")
    # Only after original 43M J7 capture AND S57/S58 forward, import frozen NEW32 GOLD.
    # This is a single first-read; nothing is tuned/selected afterwards.
    from research.nra3_s20.s58_new32_separately_frozen_manual_gold_NEVER_TRAIN import (
        held_gold32,gold_sha256,
    )
    manual=held_gold32()
    require(len(manual)==len(SOURCE_ROWS)==32,"new S58 manual labels drift")
    rec_truth=torch.zeros_like(old["directed_recipient"])
    rej_truth=torch.zeros_like(rec_truth)
    rows=[]
    main_correct_old=main_correct_new=0
    for k,(source,gold) in enumerate(zip(SOURCE_ROWS,manual)):
        words=raw_surface_grammar(source[2])
        require(source[0]==gold.row_id and len(words)==int(valid[k].sum()),
                "new S58 first-held row identity or tokenization drift")
        def find(token):
            ids=[j for j,tok in enumerate(words) if tok.form==token]
            require(len(ids)==1,"ambiguous new S58 gold word")
            return ids[0]
        src=find(gold.main_predicate)
        main_correct_old+=int(int(old["chosen_outer_event_index"][k])==src)
        main_correct_new+=int(int(new83_pred["chosen_event"][k])==src)
        if gold.transferred_recipient is not None:
            dest=find(gold.transferred_recipient)
            rec_truth[k,src,dest]=True
            if gold.directed_transfer_polarity=="REJECTED":
                rej_truth[k,src,dest]=True
        rows.append({
            "id":source[0],"family":source[1],
            "old_rec":int(old["directed_recipient"][k].sum()),
            "S58_rec":int(new["directed_recipient"][k].sum()),
            "old_rej":int(old["directed_rejected"][k].sum()),
            "S58_rej":int(new["directed_rejected"][k].sum()),
            "S58_paid83_rec":int(new83_pred["directed_recipient"][k].sum()),
            "S58_paid83_rej":int(new83_pred["directed_rejected"][k].sum()),
            "predicted_event_index":int(new["chosen_event_minus1_on_abstain"][k]),
            "S57_main_predicate_correct":int(old["chosen_outer_event_index"][k])==src,
            "S58_paid83_main_predicate_correct":int(new83_pred["chosen_event"][k])==src,
            "gold_recipient_safe":bool(features.valid_pair[k,src,dest]) if gold.transferred_recipient is not None else None
        })
    require(int(rec_truth.sum())==16 and int(rej_truth.sum())==8,
            "new precommitted gold directed 16/8 totals drift")
    totals={
        "S57_old_recipient":counts(old["directed_recipient"],rec_truth),
        "S58_guard_recipient":counts(new["directed_recipient"],rec_truth),
        "S57_old_rejected":counts(old["directed_rejected"],rej_truth),
        "S58_guard_rejected":counts(new["directed_rejected"],rej_truth),
        "S58_paid83_recipient":counts(new83_pred["directed_recipient"],rec_truth),
        "S58_paid83_rejected":counts(new83_pred["directed_rejected"],rej_truth),
    }
    families={}
    for family in FAMILIES:
        ix=torch.tensor([k for k,r in enumerate(SOURCE_ROWS) if r[1]==family])
        require(len(ix)==8,"incomplete new32 source family")
        families[family]={
            "S57_recipient":counts(old["directed_recipient"][ix],rec_truth[ix]),
            "S58_recipient":counts(new["directed_recipient"][ix],rec_truth[ix]),
            "S57_rejected":counts(old["directed_rejected"][ix],rej_truth[ix]),
            "S58_rejected":counts(new["directed_rejected"][ix],rej_truth[ix]),
            "S58_paid83_recipient":counts(new83_pred["directed_recipient"][ix],rec_truth[ix]),
            "S58_paid83_rejected":counts(new83_pred["directed_rejected"][ix],rej_truth[ix]),
        }
    report={
        "stage":STAGE,
        "first_new32_is_author_created_challenge_not_population_EVAL":True,
        "frozen_S58_NEW32_RAW_commit":"82596c03eb5a519f2409fd1f8eecddfb9df97cc6",
        "frozen_S58_NEW32_RAW_blob":"0bdab37c6e3a634fd901e0e0c1d37bb36bcadee1",
        "frozen_S58_NEW32_GOLD_commit":"4a13566295d6adf49cf3a7ab97975621a015a8ae",
        "frozen_S58_NEW32_GOLD_blob":"4f44b5946f6f534a2017337afa0370807e7a219d",
        "frozen_S58_NEW32_RAW_sha256":raw_sha256(),
        "frozen_S58_NEW32_GOLD_sha256":gold_sha256(),
        "new_HELD_gold_imported_AFTER_ALL_frozen_forward":True,
        "new_HELD32_previously_unscored_now_spent_after_first_read":True,
        "source_S57_original43M_GT_native_SHA256":sha(args.native),
        "source_S57_paid231_safetensors_SHA256":sha(args.s57),
        "new_S58_old96_trainonly_paid83_safetensors_SHA256":sha(args.s58),
        "source_S26_paid_archive_SHA256":sha(args.s26),
        "original_616_J7_replayed_on_unchanged_core_BEFORE_S58_NEW32_CPU_capture":True,
        "original_43M_GT_new32_CPU_prefill_complete":True,
        "S58_active_integrated_model_J7_NOT_tested":True,
        "S58_frozen_input_and_paid_models_unchanged":base_unmodified and weights_unmodified,
        "S58_structural_abstention_reasons":new["reasons"],
        "S58_syntactic_heuristics_can_reduce_recall":True,
        "S58_new83_never_train_on_S57_dev24":True,
        "S58_NEW32_gold_not_used_training_native_forward":True,
        "S57_main_predicate_correct":main_correct_old,
        "S58_paid83_main_predicate_correct":main_correct_new,
        "candidate_NOT_selected_on_new32":True,
        "results_NEW32_first_read":totals,"per_family_NEW32_first_read":families,
        "rows_NEW32_first_read":rows,
        "output_gate":0,"candidate_accepted":False,
        "activation_allowed":False,
        "causal_certified":False
    }
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"NRA3_S58_TRUE43M_GT_NEW32_FIRST_READ_MANUAL_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print("S58_FIRST_NEW32_REAL43M_GT_FROZEN_S57_S58_COMPARISON",
          json.dumps(totals,sort_keys=True),flush=True)
    print("S58_FIRST_NEW32_ABSTAIN_REASON_COUNTS",
          json.dumps(new["reasons"],sort_keys=True),flush=True)


if __name__=="__main__":
    main()
