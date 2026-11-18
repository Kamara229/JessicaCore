"""Physically train NEW S60 2776FP16 ONLY on genuine native OLD184 S53-S59.

All seven corpora are previously scored/inspected SPENT TRAIN, never
independent HELD again. S60 raw/gold32 are explicitly absent from imports.
Original 43M+Green Tunnel, paid S26, S57 and S59 frozen unchanged.
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
    word_idx,OLD_S26,
)
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    SemanticInputs,S59NativeEventSemantic,semantic_features,
)
from research.nra3_s20.s59_fit_spent152_true43m_gt_native2046 import (
    ALL_ARCHIVES as PREVIOUS_6,training_rows_and_old_gold,
)
from research.nra3_s20.s60_native43m_local_triclass_model import (
    S60Input,source_local_scope,fit_old184,P,
)
STAGE="NRA3_S60_FROZEN_ORIGINAL43M_GT_SPENT184_NATIVE_3WAY2776_FP16"
S59_NATIVE="NRA3_S59_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
S59_PROV="NRA3_S59_GENUINE43M_GT_EVENTFIRST32_OFFSET_PROVENANCE.json"
S57="NRA3_S57_EVENTFIRST_SPENT96_ONLY_231FP16_UNACCEPTED.safetensors"
S57_FIT="NRA3_S57_EVENTFIRST_SPENT96_231FP16_TRAIN_REPORT.json"
S59="NRA3_S59_REAL43M_GT_OLD152_NATIVE320_SEMANTIC2046_FP16.safetensors"
S59_FIT="NRA3_S59_OLD152_TRAIN_NATIVE2046_FP16_MANIFEST.json"
FILES=PREVIOUS_6+(("S59",S59_NATIVE,"source_row_indices_in_frozen_S59_HELD32"),)

def require(x,s):
    if not bool(x):raise ValueError("S60 genuine43M GT spent184 train: "+s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ("s53","s54","s55","s56","s57-native","s58-native",
                 "s59-native","s57-prov","s58-prov","s59-prov","s26",
                 "s57","s57-fit","s59","s59-fit","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"refuse overwriting S60 old TRAIN model")
    files=[a.s53,a.s54,a.s55,a.s56,a.s57_native,a.s58_native,a.s59_native]
    require(all(f.is_file() and f.name==tag[1]
                for tag,f in zip(FILES,files)),
            "S60 requires precisely original S53-S59 physical native files")
    require(a.s26.name==OLD_S26 and a.s26.is_file()
            and a.s57.name==S57 and a.s57.is_file()
            and a.s57_fit.name==S57_FIT and a.s57_fit.is_file()
            and a.s59.name==S59 and a.s59.is_file()
            and a.s59_fit.name==S59_FIT and a.s59_fit.is_file()
            and a.s59_prov.name==S59_PROV and a.s59_prov.is_file()
            and all(x.is_file() for x in (a.s57_prov,a.s58_prov)),
            "S60 frozen original S26/S57/S59 provenance archive missing")
    fit57=json.loads(a.s57_fit.read_text())
    fit59=json.loads(a.s59_fit.read_text())
    provs=[json.loads(z.read_text()) for z in (
        a.s57_prov,a.s58_prov,a.s59_prov)]
    require(fit57["new_event_first_FP16_weights_SHA256"]==sha(a.s57)
            and fit57["old_S26_12384_FP16_physical_archive_SHA256"]==sha(a.s26)
            and fit59["new_S59_semantic_paid2046_sha256"]==sha(a.s59)
            and fit59["old_S57_paid231_archive_SHA256"]==sha(a.s57)
            and fit59["original_physical_S26_archive_SHA256"]==sha(a.s26)
            and fit59["never_imported_S59_new32_RAW_or_GOLD_into_training"]
            and all(q["derived_REAL43M_WORD339_SHA256"]==sha(f)
                    for q,f in zip(provs,files[4:]))
            and all(q["original_616_J7_replayed_before_read_only_quotes"]
                    for q in provs)
            and [q["stage"] for q in provs]==[
                "NRA3_S57_GENUINE43M_GT_EVENTFIRST24_BPE_SOURCE_ONLY",
                "NRA3_S58_GENUINE43M_GT_EVENTFIRST32_BPE_SOURCE_ONLY",
                "NRA3_S59_GENUINE43M_GT_EVENTFIRST32_BPE_SOURCE_ONLY"],
            "original CPU model or frozen paid ancestry SHA/manifest mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows,gold=training_rows_and_old_gold()
    from research.nra3_s20.s59_new32_first_frozen_raw_only_NEVER_TRAIN import SOURCE_ROWS as R59
    from research.nra3_s20.s59_new32_first_frozen_manual_gold_NEVER_TRAIN import held_gold32 as G59
    rows=rows+(R59,);gold=gold+(G59(),)
    require([len(z) for z in rows]==[24,24,24,24,24,32,32]
            and [len(z) for z in gold]==[24,24,24,24,24,32,32],
            "S60 only old152 + spent S59 32 = old184")
    old26=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(a.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    old59=S59NativeEventSemantic.from_fp16(
        load_file(str(a.s59),device="cpu")["S59_NATIVE320_SEMANTIC2046_FP16"])
    frozen=(old26,old57,old59)
    require(all(not p.requires_grad for m in frozen for p in m.parameters())
            and all(m.output_gate==0 and not m.candidate_accepted for m in frozen),
            "older paid models not fully frozen or gate0")
    old_hash=[tensor_sha(*(p.detach() for p in m.parameters())) for m in frozen]
    groups=[]
    counts=[]
    for (tag,_,index_key),path,raw,golds in zip(FILES,files,rows,gold):
        stored=load_file(str(path),device="cpu")
        require(set(stored)=={
            "original_real43m_gt_word339_READONLY",
            "original_native_word_valid_mask",
            "native_BPE_causally_unsafe_word_mask",index_key},
            "real native artifact leaked unexpected labels")
        x=stored["original_real43m_gt_word339_READONLY"].detach().clone()
        valid=stored["original_native_word_valid_mask"].detach().clone()
        unsafe=stored["native_BPE_causally_unsafe_word_mask"].detach().clone()
        b=len(raw)
        require(x.shape[0]==b and x.shape[-1]==339
                and valid.shape==unsafe.shape==x.shape[:2]
                and stored[index_key].tolist()==list(range(b)),
                "S60 native archival old group size/alignment invalid")
        before=tensor_sha(x,valid,unsafe)
        with torch.no_grad():
            obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
            event=gold_blind_eventfirst_features(x,obs,tuple(r[2] for r in raw))
            sem=semantic_features(x,event)
            scope=source_local_scope(sem,tuple(r[2] for r in raw))
        event=EventFirstInputs(event.event.detach().clone(),
            event.pair.detach().clone(),event.candidate_event.detach().clone(),
            event.valid_pair.detach().clone())
        sem=SemanticInputs(event,sem.semantic.detach().clone())
        scope=S60Input(sem,scope.scope.detach().clone(),scope.fused.detach().clone())
        require(not scope.fused.is_inference(),"S60 parent inference tensor in TRAIN")
        idx=torch.zeros(b,dtype=torch.long)
        classes=torch.zeros(b,dtype=torch.long)
        pairs=torch.zeros_like(event.valid_pair)
        rejected=torch.zeros_like(pairs)
        for i,(row,g) in enumerate(zip(raw,golds)):
            require(row[0]==g.row_id,"spent gold/native row order changed")
            tokens=raw_surface_grammar(row[2])
            require(len(tokens)==int(valid[i].sum()),"real original native BPE word drift")
            pred=g.predicate if tag in ("S53","S54") else g.main_predicate
            rec=g.recipient if tag in ("S53","S54") else g.transferred_recipient
            pol=(g.recipient_polarity if tag in ("S53","S54") else
                 g.transfer_polarity if tag=="S55" else
                 g.directed_transfer_polarity)
            src=word_idx(tokens,pred)
            require(bool(event.candidate_event[i,src]),"old TRUE predicate ineligible")
            idx[i]=src
            if rec is not None:
                dst=word_idx(tokens,rec)
                require(bool(event.valid_pair[i,src,dst]),"old TRUE recipient BPE unsafe")
                pairs[i,src,dst]=True
                classes[i]=2 if pol=="REJECTED" else 1
                if pol=="REJECTED":rejected[i,src,dst]=True
        require(before==tensor_sha(x,valid,unsafe),
                "original immutable postGT word features unexpectedly changed")
        groups.append((scope,idx,classes,pairs,rejected))
        counts.append({"tag":tag,"n":b,"NONE":int((classes==0).sum()),
                       "ASSERTED":int((classes==1).sum()),
                       "REJECTED":int((classes==2).sum())})
    obj,physical,metrics=fit_old184(groups,old57,old59)
    require(physical.shape==(2776,) and all(not p.requires_grad for p in obj.parameters())
            and not obj.candidate_accepted and obj.output_gate==0
            and all(tensor_sha(*(p.detach() for p in m.parameters()))==z
                    for m,z in zip(frozen,old_hash)),
            "S60 trained FP16 or original architecture mutation detected")
    a.out.mkdir(parents=True,exist_ok=False)
    destination=a.out/"NRA3_S60_ORIGINAL43M_GT_OLD184_TRICLASS2776_FP16.safetensors"
    save_file({"S60_SOURCE_LOCAL_TRICLASS2776_FP16":physical},str(destination))
    require(torch.equal(load_file(str(destination))[
                "S60_SOURCE_LOCAL_TRICLASS2776_FP16"],physical),
            "S60 actual FP16 persisted content drift")
    report={
       "stage":STAGE,
       "old_real43M_GT_native_sha256":[sha(x) for x in files],
       "original_S26_paid12384_sha256":sha(a.s26),
       "frozen_S57_paid231_sha256":sha(a.s57),
       "frozen_S59_paid2046_sha256":sha(a.s59),
       "new_S60_paid2776_sha256":sha(destination),
       "new_S60_FP16_parameters":2776,
       "old184_spent_training_gold_class_counts":counts,
       "training":metrics,
       "new_S60_RAW32_and_GOLD32_never_imported_into_OLD184_TRAIN":True,
       "protected_independent_EVAL32_EVAL128_or_S38_S47_S48_not_used":True,
       "real_Calibri43M_GreenTunnel_and_all_old_paid_weights_unchanged":True,
       "real_S60_active616J7_bpp_CB59_NOT_measured":True,
       "no_live_or_deployment":True,
       "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (a.out/"NRA3_S60_OLD184_TRAIN_TRICLASS2776_MANIFEST.json").write_text(
       json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("S60_REAL43M_GT_OLD184_TRAIN3CLASS_FP16",
          json.dumps({"sha":sha(destination),"paid":2776,"old184":counts,
                      "output_gate":0},sort_keys=True),flush=True)
if __name__=="__main__":main()
