"""S58 *spent S57* author-built 24-case REAL 43M+GT native DEV diagnostic.

S57 first-read HELD24 has already been inspected. It is permanently
SPENT. This audit is development ONLY, cannot count as fresh HELD/EVAL,
cannot certify performance, integration, J7 retention or online causality.

Inputs must be physically captured S57 frozen real original 43M+GT
native word339 + paid S26 and S57 archives from pinned CI runs.
S58 only changes the new deterministic source-only guard. Frozen source
features and weights remain immutable. Import spent S57 gold strictly
AFTER original S57 and guarded S58 predictions are obtained.
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
from research.nra3_s20.s57_frozen_eventfirst_held24_plain import (
    SOURCE_ROWS, FAMILIES, text_only_inputs, text_sha256,
)

STAGE="NRA3_S58_SPENT_S57_24_REAL43M_GT_DEV_ONLY_NOT_HELD"
NATIVE="NRA3_S57_GENUINE43M_GT_EVENTFIRST24_WORD339_AND_UNSAFE.safetensors"
PROV="NRA3_S57_GENUINE43M_GT_EVENTFIRST24_OFFSET_PROVENANCE.json"
CAP="NRA3_S57_GENUINE43M_GT_EVENTFIRST24_NATIVE_J7_REPORT.json"
S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
S57="NRA3_S57_EVENTFIRST_SPENT96_ONLY_231FP16_UNACCEPTED.safetensors"
FIT="NRA3_S57_EVENTFIRST_SPENT96_231FP16_TRAIN_REPORT.json"


def require(ok,msg):
    if not bool(ok):
        raise ValueError("S58 spent57 research-only: "+msg)


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
    for name in ("native","provenance","capture","s26","s57","fit","out"):
        p.add_argument("--"+name,required=True,type=Path)
    args=p.parse_args()
    require(not args.out.exists(),"cannot overwrite a spent-data report")
    require(all(x.is_file() for x in (
        args.native,args.provenance,args.capture,args.s26,args.s57,args.fit)),
        "missing physically archived frozen model source")
    require((args.native.name,args.provenance.name,args.capture.name,
             args.s26.name,args.s57.name,args.fit.name)==
            (NATIVE,PROV,CAP,S26,S57,FIT),"unexpected archive name")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    provenance=json.loads(args.provenance.read_text())
    capture=json.loads(args.capture.read_text())
    fit=json.loads(args.fit.read_text())
    require(provenance["derived_REAL43M_WORD339_SHA256"]==sha(args.native)
            and provenance["frozen_S57_HELD24_raw_source_sha256"]==text_sha256()
            and provenance["original_616_J7_replayed_before_read_only_quotes"]
            and capture["all_616_native_CPU_generated_token_digests_unchanged"]
            and capture["actual_genuine_43M_CPU_new_HELD24_prefills"]==24
            and fit["new_event_first_FP16_weights_SHA256"]==sha(args.s57)
            and fit["new_event_first_FP16_model_parameters"]==231
            and fit["new_S57_first_HELD24_raw_and_manual_gold_NEVER_IMPORTED"],
            "S57 original real CPU 43M+GT + saved paid231 provenance failed")
    native=load_file(str(args.native),device="cpu")
    require(set(native)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S57_HELD24"}, "unexpected original model feature keys")
    x=native["original_real43m_gt_word339_READONLY"].detach()
    valid=native["original_native_word_valid_mask"].detach()
    unsafe=native["native_BPE_causally_unsafe_word_mask"].detach()
    require(x.ndim==3 and x.shape[0]==24 and x.shape[2]==339
            and valid.shape==unsafe.shape==x.shape[:2]
            and valid.dtype==unsafe.dtype==torch.bool
            and native["source_row_indices_in_frozen_S57_HELD24"].tolist()==list(range(24)),
            "native word alignment and safe masks drift")
    base_sha=tensor_sha(x,valid,unsafe)
    old26=EventGroundedInversionGraph.from_physical(args.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(args.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    require(not any(p.requires_grad for model in (old26,old57)
                    for p in model.parameters()),"old paid parameters unexpectedly unfrozen")
    model_sha=tensor_sha(*(p.detach() for p in old57.parameters()))
    with torch.inference_mode():
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
        features=gold_blind_eventfirst_features(x,obs,text_only_inputs())
        old=old57.decode(features)
        new=selective_abstain(bridge_frozen_s57(old57,features))
        base_unmodified=(base_sha==tensor_sha(x,valid,unsafe))
        weights_unmodified=(model_sha==tensor_sha(*(p.detach() for p in old57.parameters())))
    require(base_unmodified and weights_unmodified and old57.output_gate==0
            and new["output_gate"]==0 and not new["candidate_accepted"]
            and not new["activation_allowed"],"read-only guard modified a frozen model")
    require(int(new["directed_recipient"].sum())<=24
            and not bool((new["directed_rejected"]&~new["directed_recipient"]).any()),
            "S58 emitted unsafe or too many directed links")
    # S57 has already been scored and publicly inspected. Use it as SPENT/dev.
    from research.nra3_s20.s57_frozen_eventfirst_held24_manual_gold import held_gold24
    manual=held_gold24()
    require(len(manual)==len(SOURCE_ROWS)==24,"spent S57 manual labels changed")
    rec_truth=torch.zeros_like(old["directed_recipient"])
    rej_truth=torch.zeros_like(rec_truth)
    rows=[]
    for k,(source,gold) in enumerate(zip(SOURCE_ROWS,manual)):
        words=raw_surface_grammar(source[2])
        require(source[0]==gold.row_id and len(words)==int(valid[k].sum()),
                "S57 spent row identity or tokenization drift")
        def find(token):
            ids=[j for j,tok in enumerate(words) if tok.form==token]
            require(len(ids)==1,"ambiguous spent gold word")
            return ids[0]
        src=find(gold.main_predicate)
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
            "predicted_event_index":int(new["chosen_event_minus1_on_abstain"][k])
        })
    require(int(rec_truth.sum())==12 and int(rej_truth.sum())==6,
            "spent gold directed totals unexpectedly changed")
    totals={
        "S57_old_recipient":counts(old["directed_recipient"],rec_truth),
        "S58_guard_recipient":counts(new["directed_recipient"],rec_truth),
        "S57_old_rejected":counts(old["directed_rejected"],rej_truth),
        "S58_guard_rejected":counts(new["directed_rejected"],rej_truth),
    }
    families={}
    for family in FAMILIES:
        ix=torch.tensor([k for k,r in enumerate(SOURCE_ROWS) if r[1]==family])
        require(len(ix)==6,"incomplete source family")
        families[family]={
            "S57_recipient":counts(old["directed_recipient"][ix],rec_truth[ix]),
            "S58_recipient":counts(new["directed_recipient"][ix],rec_truth[ix]),
            "S57_rejected":counts(old["directed_rejected"][ix],rej_truth[ix]),
            "S58_rejected":counts(new["directed_rejected"][ix],rej_truth[ix]),
        }
    report={
        "stage":STAGE,
        "source_S57_previous_FIRST_HELD_now_PERMANENTLY_SPENT":True,
        "SPENT24_dev_ONLY_no_independent_eval":True,
        "source_S57_original43M_GT_native_SHA256":sha(args.native),
        "source_S57_paid231_safetensors_SHA256":sha(args.s57),
        "source_S26_paid_archive_SHA256":sha(args.s26),
        "original_616_J7_from_PREVIOUS_S57_run_not_replayed_S58":True,
        "S58_new_CPU_616J7_NOT_tested":True,
        "S58_new_original_43M_GT_CPU_PREFILL_NOT_run":True,
        "S58_frozen_input_and_paid_models_unchanged":base_unmodified and weights_unmodified,
        "S58_structural_abstention_reasons":new["reasons"],
        "S58_syntactic_heuristics_can_reduce_recall":True,
        "results_SPENT_dev24":totals,"per_family_SPENT_dev24":families,
        "rows_SPENT_dev24":rows,
        "output_gate":0,"candidate_accepted":False,
        "activation_allowed":False,
        "causal_certified":False
    }
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"NRA3_S58_SPENT_S57_DEV24_RESEARCH_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print("S58_SPENT_S57_REAL43M_GT_READONLY_DEV24_NOT_INDEPENDENT",
          json.dumps(totals,sort_keys=True),flush=True)
    print("S58_SPENT_S57_ABSTAIN_REASON_COUNTS",
          json.dumps(new["reasons"],sort_keys=True),flush=True)


if __name__=="__main__":
    main()
