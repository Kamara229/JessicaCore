"""S59 train ONLY on six ALREADY INSPECTED original Calibri43M+GT groups.

Physical native post-layer8 word339: S53/54/55/56 24 each,
S57 previously-scored 24, S58 previously-scored 32 = OLD TRAIN 152.
True original S26 and S57 paid231 are immutable; train only NEW
S59 native320 semantic 2046FP16. S59 fresh HELD raw/gold NEVER imported.

S57 and S58 prior HELD have now been *spent*; fit on them is allowed
as TRAIN but cannot be cited as independent evaluation.
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
from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import (
    EventFirstInputs,S57EventFirstRecipient,gold_blind_eventfirst_features,
)
from research.nra3_s20.s57_fit_old96_real43m_gt_eventfirst231_cpu import (
    ARCHIVES as OLD_FOUR,OLD_S26,word_idx,
)
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    SemanticInputs,semantic_features,S59NativeEventSemantic,
    fit_only_PREVIOUSLY_INSPECTED_real43m_cases,P,
)

STAGE="NRA3_S59_FROZEN_ORIGINAL43M_GT_OLD152_NATIVE320_TRAIN2046_FP16"
OLD_S57="NRA3_S57_EVENTFIRST_SPENT96_ONLY_231FP16_UNACCEPTED.safetensors"
OLD_S57_MANIFEST="NRA3_S57_EVENTFIRST_SPENT96_231FP16_TRAIN_REPORT.json"
NATIVE57="NRA3_S57_GENUINE43M_GT_EVENTFIRST24_WORD339_AND_UNSAFE.safetensors"
NATIVE58="NRA3_S58_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
PROV57="NRA3_S57_GENUINE43M_GT_EVENTFIRST24_OFFSET_PROVENANCE.json"
PROV58="NRA3_S58_GENUINE43M_GT_EVENTFIRST32_OFFSET_PROVENANCE.json"
ALL_ARCHIVES=OLD_FOUR+(
 ("S57",NATIVE57,"source_row_indices_in_frozen_S57_HELD24"),
 ("S58",NATIVE58,"source_row_indices_in_frozen_S58_HELD32"),
)

def require(ok,msg):
    if not bool(ok):raise ValueError("S59 original real43M old152 TRAIN: "+msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def training_rows_and_old_gold():
    """Explicit ALL SPENT, no S59/new HELD imports."""
    from research.nra3_s20.s53_precommitted_negation_quote24_plain import SOURCE_ROWS as R53
    from research.nra3_s20.s53_separate_manual_gold_negation24 import gold24 as G53
    from research.nra3_s20.s54_frozen_new_held24_plain_source import SOURCE_ROWS as R54
    from research.nra3_s20.s54_frozen_new_held24_manual_gold import held_gold24 as G54
    from research.nra3_s20.s55_frozen_structural_shift_held24_plain import SOURCE_ROWS as R55
    from research.nra3_s20.s55_frozen_structural_shift_held24_manual_gold import held_gold24 as G55
    from research.nra3_s20.s56_frozen_new_scope24_plain import SOURCE_ROWS as R56
    from research.nra3_s20.s56_frozen_new_scope24_gold import held_gold24 as G56
    from research.nra3_s20.s57_frozen_eventfirst_held24_plain import SOURCE_ROWS as R57
    from research.nra3_s20.s57_frozen_eventfirst_held24_manual_gold import held_gold24 as G57
    from research.nra3_s20.s58_new32_preregistered_raw_only_do_not_train import SOURCE_ROWS as R58
    from research.nra3_s20.s58_new32_separately_frozen_manual_gold_NEVER_TRAIN import held_gold32 as G58
    rows=(R53,R54,R55,R56,R57,R58)
    gold=(G53(),G54(),G55(),G56(),G57(),G58())
    require([len(z) for z in rows]==[24,24,24,24,24,32]
            and [len(z) for z in gold]==[24,24,24,24,24,32],
            "S59 OLD native supervised rows must be exactly 152")
    return rows,gold


def main():
    parser=argparse.ArgumentParser()
    for k in ("s53","s54","s55","s56","s57-native","s58-native",
              "s26","s57-paid","s57-fit","s57-prov","s58-prov","out"):
        parser.add_argument("--"+k,type=Path,required=True)
    a=parser.parse_args()
    require(not a.out.exists(),"never overwrite S59 physical TRAIN checkpoint")
    paths=[a.s53,a.s54,a.s55,a.s56,a.s57_native,a.s58_native]
    require(all(file.name==tag[1] and file.is_file()
                for tag,file in zip(ALL_ARCHIVES,paths)),
            "S59 only 152 genuine original archived word339 files")
    require(a.s26.is_file() and a.s26.name==OLD_S26
            and a.s57_paid.is_file() and a.s57_paid.name==OLD_S57
            and a.s57_fit.is_file() and a.s57_fit.name==OLD_S57_MANIFEST
            and a.s57_prov.name==PROV57 and a.s57_prov.is_file()
            and a.s58_prov.name==PROV58 and a.s58_prov.is_file(),
            "original S26 and S57 paid or old provenance missing")
    s57_fit=json.loads(a.s57_fit.read_text())
    p57=json.loads(a.s57_prov.read_text())
    p58=json.loads(a.s58_prov.read_text())
    require(s57_fit["stage"]==
            "NRA3_S57_OLD_SPENT_S53_S54_S55_S56_TRAIN96_REAL43M_GT_EVENTFIRST_231FP16"
            and s57_fit["old_training_4x24_real43M_GT_native_archive_sha256"]==
            [sha(k) for k in paths[:4]]
            and s57_fit["old_S26_12384_FP16_physical_archive_SHA256"]==sha(a.s26)
            and s57_fit["new_event_first_FP16_weights_SHA256"]==sha(a.s57_paid)
            and p57["derived_REAL43M_WORD339_SHA256"]==sha(a.s57_native)
            and p58["derived_REAL43M_WORD339_SHA256"]==sha(a.s58_native)
            and p57["original_616_J7_replayed_before_read_only_quotes"]
            and p58["original_616_J7_replayed_before_read_only_quotes"]
            and p57["S57_HUMAN_GOLD_NOT_IMPORTED_DURING_ANY_CAPTURE_STEP"]
            and p58["S58_HUMAN_GOLD_NOT_IMPORTED_DURING_ANY_CAPTURE_STEP"],
            "original S53-S58 genuine CPU and trained S57 FP16 source SHA drift")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rows,golds=training_rows_and_old_gold()
    old26=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(a.s57_paid),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    require(all(not q.requires_grad for model in (old26,old57)
                for q in model.parameters()) and old26.parent.output_gate==0
            and old26.output_gate==old57.output_gate==0,
            "S26/S57 research frozen original paid model not locked")
    before=[tensor_sha(*(p.detach() for p in model.parameters()))
            for model in (old26,old57)]
    cases=[]
    gold_count=[]
    native_shas=[]
    for (tag,_,idx_key),path,raw,manual in zip(ALL_ARCHIVES,paths,rows,golds):
        saved=load_file(str(path),device="cpu")
        require(set(saved)=={
            "original_real43m_gt_word339_READONLY",
            "original_native_word_valid_mask",
            "native_BPE_causally_unsafe_word_mask",idx_key},
            "original GT gold-blind native arch includes unexpected labels")
        x=saved["original_real43m_gt_word339_READONLY"].detach().clone()
        valid=saved["original_native_word_valid_mask"].detach().clone()
        unsafe=saved["native_BPE_causally_unsafe_word_mask"].detach().clone()
        b=len(raw)
        require(x.ndim==3 and x.shape[0]==b and x.shape[-1]==339
                and valid.shape==unsafe.shape==x.shape[:2]
                and saved[idx_key].tolist()==list(range(b)),
                "native 43M/GT source token and word ABI mismatched")
        before_native=tensor_sha(x,valid,unsafe)
        with torch.no_grad():
            obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
            src=gold_blind_eventfirst_features(
                x,obs,tuple(row[2] for row in raw))
            semantic=semantic_features(x,src)
        # Training autograd must not reference frozen or inference-mode tensors.
        src=EventFirstInputs(
            src.event.detach().clone(),src.pair.detach().clone(),
            src.candidate_event.detach().clone(),src.valid_pair.detach().clone())
        semantic=SemanticInputs(src,semantic.semantic.detach().clone())
        require(not semantic.semantic.is_inference()
                and not semantic.semantic.requires_grad,
                "S59 semantic old model tensor not safe for new optimizer")
        event=torch.zeros(b,dtype=torch.long)
        links=torch.zeros_like(src.valid_pair)
        rej=torch.zeros_like(links)
        has=torch.zeros(b,dtype=torch.bool)
        for j,(r,g) in enumerate(zip(raw,manual)):
            require(r[0]==g.row_id,"S59 old REAL native TRAIN label row alignment drift")
            tok=raw_surface_grammar(r[2])
            require(len(tok)==int(valid[j].sum()),
                    "S59 old original word339 native text length mismatch")
            predicate=(g.predicate if tag in ("S53","S54") else g.main_predicate)
            recipient=(g.recipient if tag in ("S53","S54") else
                       g.transferred_recipient)
            polarity=(g.recipient_polarity if tag in ("S53","S54") else
                      g.transfer_polarity if tag=="S55" else
                      g.directed_transfer_polarity)
            i=word_idx(tok,predicate)
            require(bool(src.candidate_event[j,i]),"old TRUE main predicate masked unsafe")
            event[j]=i
            if recipient is not None:
                z=word_idx(tok,recipient)
                require(bool(src.valid_pair[j,i,z]),"old TRUE linked recipient BPE/quote unsafe")
                links[j,i,z]=True
                has[j]=True
                if polarity=="REJECTED":rej[j,i,z]=True
        require(tensor_sha(x,valid,unsafe)==before_native,
                "S59 old genuine original 43M GT bytes modified")
        cases.append((semantic,event,has,links,rej))
        gold_count.append({"group":tag,"rows":b,"recipients":int(links.sum()),
                           "rejected":int(rej.sum())})
        native_shas.append(sha(path))
    candidate,physical,fit=fit_only_PREVIOUSLY_INSPECTED_real43m_cases(cases,old57)
    require(physical.shape==(P,) and physical.dtype==torch.float16
            and not candidate.candidate_accepted
            and not candidate.activation_allowed
            and candidate.output_gate==0
            and all(not p.requires_grad for p in candidate.parameters())
            and all(tensor_sha(*(q.detach() for q in m.parameters()))==s
                    for m,s in zip((old26,old57),before)),
            "S59 old parent FP16 or original GT mutated during semantic TRAIN")
    a.out.mkdir(parents=True,exist_ok=False)
    weights=a.out/"NRA3_S59_REAL43M_GT_OLD152_NATIVE320_SEMANTIC2046_FP16.safetensors"
    save_file({"S59_NATIVE320_SEMANTIC2046_FP16":physical},str(weights))
    require(torch.equal(load_file(str(weights),device="cpu")[
                "S59_NATIVE320_SEMANTIC2046_FP16"],physical),
            "S59 physical FP16 semantic source SHA roundtrip failed")
    report={
        "stage":STAGE,
        "source_original_43M_GT_native_archived_runs":[
            44274389010,44279445613,44282672013,44308286512,
            44312765248,44324723824],
        "original_physical_S26_archive_SHA256":sha(a.s26),
        "old_S57_paid231_archive_SHA256":sha(a.s57_paid),
        "old_original_6_native_sources_SHA256":native_shas,
        "spent_OLD152_manual_training_gold_counts":gold_count,
        "new_S59_semantic_paid2046_file":weights.name,
        "new_S59_semantic_paid2046_sha256":sha(weights),
        "new_semantic_FP16_parameters":P,
        "train_source_only_old152":fit,
        "never_imported_S59_new32_RAW_or_GOLD_into_training":True,
        "S57_S58_old_HELD24_32_used_as_SPENT_TRAIN_not_independent_EVAL":True,
        "original_Calibri43M_GT_S26_S57_unchanged":True,
        "original_616J7_only_old_frozen_parent_verified_in_prior_runs":True,
        "active_S59_616J7_bpp_CB59_NOT_certified":True,
        "no_live_or_deployment":True,
        "no_protected_S38_S47_S48_EVAL32_128":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (a.out/"NRA3_S59_OLD152_TRAIN_NATIVE2046_FP16_MANIFEST.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
        encoding="utf-8")
    print("S59_REAL_NATIVE43M_GT_SEMANTIC2046_OLD152_ONLY",
          json.dumps({"SHA256":sha(weights),"old_gold":gold_count,
                      "paid_fp16":P,"output_gate":0,
                      "new_S59_first_held_NOT_imported":True},
                     ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":main()
