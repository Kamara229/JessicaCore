"""Physically fit S61 NEW pair-aware 2752FP16 on OLD SPENT216 ONLY.

Re-read immutable original Calibri43M + Green Tunnel post-layer8 CPU
word339 archives S53-S60 (24×5 + 32×3), all previously inspected.
Source-only native and S26/S57/S59 older paid readouts are frozen.
S61 new fresh RAW32 and separate manual gold are not imported or seen.
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
    S57EventFirstRecipient,EventFirstInputs,gold_blind_eventfirst_features,
)
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    S59NativeEventSemantic,SemanticInputs,semantic_features,
)
from research.nra3_s20.s59_fit_spent152_true43m_gt_native2046 import (
    training_rows_and_old_gold,
)
from research.nra3_s20.s60_fit_old184_true43m_gt_triclass2776 import (
    FILES as OLD_SEVEN,S57,S59,word_idx,
)
from research.nra3_s20.s60_native43m_local_triclass_model import (
    S60Input,source_local_scope,
)
from research.nra3_s20.s61_native_pair_null_polarity_2752 import (
    S61NativePairNull,fit_SPENT_216_only,PAID,
)

STAGE="NRA3_S61_OLD216_REAL43M_GT_SOURCE_TARGET2752_TRAIN_ONLY"
S60_NATIVE="NRA3_S60_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
FILES=OLD_SEVEN+(("S60",S60_NATIVE,"source_row_indices_in_frozen_S60_HELD32"),)
S57_FIT="NRA3_S57_EVENTFIRST_SPENT96_231FP16_TRAIN_REPORT.json"
S59_FIT="NRA3_S59_OLD152_TRAIN_NATIVE2046_FP16_MANIFEST.json"
S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"

def require(ok,reason):
    if not bool(ok):raise ValueError("S61 actual43M GT old216 fitting: "+reason)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load_old_spent216():
    rows,gold=training_rows_and_old_gold()
    from research.nra3_s20.s59_new32_first_frozen_raw_only_NEVER_TRAIN import SOURCE_ROWS as R59
    from research.nra3_s20.s59_new32_first_frozen_manual_gold_NEVER_TRAIN import held_gold32 as G59
    from research.nra3_s20.s60_new32_precommitted_raw_only import SOURCE_ROWS as R60
    from research.nra3_s20.s60_new32_separate_precommitted_gold_NEVER_TRAIN import held_gold32 as G60
    rows=rows+(R59,R60)
    gold=gold+(G59(),G60())
    require([len(g) for g in rows]==[24,24,24,24,24,32,32,32]
            and [len(g) for g in gold]==[24,24,24,24,24,32,32,32],
            "S61 old train must be precisely 216 old inspected cases")
    return rows,gold

def main():
    p=argparse.ArgumentParser()
    for name in ("s53","s54","s55","s56","s57-native","s58-native",
                 "s59-native","s60-native","s57-prov","s58-prov",
                 "s59-prov","s60-prov","s26","s57","s57-fit",
                 "s59","s59-fit","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"S61 cannot overwrite trained physical state")
    paths=[a.s53,a.s54,a.s55,a.s56,a.s57_native,a.s58_native,
           a.s59_native,a.s60_native]
    require(all(path.is_file() and path.name==key[1]
                for path,key in zip(paths,FILES)),
            "must provide eight physically genuine43M GT archive files only")
    require(a.s26.is_file() and a.s26.name==S26
            and a.s57.is_file() and a.s57.name==S57
            and a.s59.is_file() and a.s59.name==S59
            and a.s57_fit.is_file() and a.s57_fit.name==S57_FIT
            and a.s59_fit.is_file() and a.s59_fit.name==S59_FIT,
            "original physically frozen GT S26/S57/S59 source missing")
    provs=[json.loads(path.read_text()) for path in (
        a.s57_prov,a.s58_prov,a.s59_prov,a.s60_prov)]
    require([v["stage"] for v in provs]==[
        f"NRA3_{tag}_GENUINE43M_GT_EVENTFIRST{size}_BPE_SOURCE_ONLY"
        for tag,size in (("S57",24),("S58",32),("S59",32),("S60",32))]
        and all(v["derived_REAL43M_WORD339_SHA256"]==sha(n)
                and v["original_616_J7_replayed_before_read_only_quotes"]
                for v,n in zip(provs,paths[4:])),
        "S61 original 43M GT S57-60 native CPU provenance corruption")
    f57=json.loads(a.s57_fit.read_text())
    f59=json.loads(a.s59_fit.read_text())
    require(f57["new_event_first_FP16_weights_SHA256"]==sha(a.s57)
            and f59["new_S59_semantic_paid2046_sha256"]==sha(a.s59)
            and f59["old_S57_paid231_archive_SHA256"]==sha(a.s57)
            and f59["original_physical_S26_archive_SHA256"]==sha(a.s26),
            "S61 old S26/S57/S59 physical FP16 ancestry altered")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows,gold=load_old_spent216()
    old26=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(a.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    old59=S59NativeEventSemantic.from_fp16(
        load_file(str(a.s59),device="cpu")["S59_NATIVE320_SEMANTIC2046_FP16"])
    olds=(old26,old57,old59)
    require(all(not z.requires_grad for m in olds for z in m.parameters())
            and all(m.output_gate==0 and not m.candidate_accepted for m in olds),
            "old original Calibri43M GT and S26/S57/S59 must be gate0")
    ancestry=[tensor_sha(*(p.detach() for p in m.parameters())) for m in olds]
    cases=[];gold_count=[]
    for (tag,_,idx_key),path,sources,manual in zip(FILES,paths,rows,gold):
        stored=load_file(str(path),device="cpu")
        require(set(stored)=={
            "original_real43m_gt_word339_READONLY",
            "original_native_word_valid_mask",
            "native_BPE_causally_unsafe_word_mask",idx_key},
            "native original postGT archive cannot include manual roles/gold")
        x=stored["original_real43m_gt_word339_READONLY"].detach().clone()
        valid=stored["original_native_word_valid_mask"].detach().clone()
        unsafe=stored["native_BPE_causally_unsafe_word_mask"].detach().clone()
        b=len(sources);width=x.shape[1]
        require(x.shape==(b,width,339) and
                valid.shape==unsafe.shape==(b,width) and
                stored[idx_key].tolist()==list(range(b)),
                "S61 archive native ABI shape/source row identity drift")
        native_sha=tensor_sha(x,valid,unsafe)
        with torch.no_grad():
            observed=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
            graph=gold_blind_eventfirst_features(
                x,observed,tuple(row[2] for row in sources))
            semantic=semantic_features(x,graph)
            fused=source_local_scope(semantic,tuple(row[2] for row in sources))
        graph=EventFirstInputs(graph.event.detach().clone(),
                               graph.pair.detach().clone(),
                               graph.candidate_event.detach().clone(),
                               graph.valid_pair.detach().clone())
        semantic=SemanticInputs(graph,semantic.semantic.detach().clone())
        fused=S60Input(semantic,fused.scope.detach().clone(),
                       fused.fused.detach().clone())
        require(not fused.fused.is_inference() and not fused.fused.requires_grad,
                "inference-mode original hidden cannot enter autograd")
        event=torch.zeros(b,dtype=torch.long)
        recipient=torch.full((b,),width,dtype=torch.long) # explicitly NONE = W
        rejected=torch.zeros((b,),dtype=torch.bool)
        for i,(row,g) in enumerate(zip(sources,manual)):
            require(row[0]==g.row_id,"old manual label row mismatch")
            words=raw_surface_grammar(row[2])
            require(len(words)==int(valid[i].sum()),"old source native word lengths drift")
            pred=(g.predicate if tag in ("S53","S54") else g.main_predicate)
            rece=(g.recipient if tag in ("S53","S54") else
                  g.transferred_recipient)
            pol=(g.recipient_polarity if tag in ("S53","S54") else
                 g.transfer_polarity if tag=="S55" else
                 g.directed_transfer_polarity)
            src=word_idx(words,pred)
            require(bool(graph.candidate_event[i,src]),
                    "gold old selected outer event excluded source native mask")
            event[i]=src
            if rece is not None:
                dest=word_idx(words,rece)
                require(bool(graph.valid_pair[i,src,dest]),
                        "old directed gold BPE pair unsafe or quoted")
                recipient[i]=dest
                rejected[i]=(pol=="REJECTED")
        require(native_sha==tensor_sha(x,valid,unsafe),
                "S61 original GT postlayer8 archive overwritten")
        cases.append((fused,event,recipient,rejected))
        gold_count.append({"tag":tag,"n":b,"with_recipient":int((recipient!=width).sum()),
                           "rejected":int(rejected.sum())})
    candidate,physical,fit=fit_SPENT_216_only(cases,old57,old59)
    require(physical.shape==(PAID,) and physical.dtype==torch.float16
            and candidate.output_gate==0 and not candidate.candidate_accepted
            and not candidate.activation_allowed
            and all(not p.requires_grad for p in candidate.parameters())
            and all(tensor_sha(*(p.detach() for p in m.parameters()))==s
                    for m,s in zip(olds,ancestry)),
            "S61 frozen parent or trained readout mutated/not gate0")
    a.out.mkdir(parents=True,exist_ok=False)
    weight=a.out/"NRA3_S61_REAL43M_GT_SPENT216_PAID2752_PAIR_NONE.safetensors"
    save_file({"S61_REAL43M_GT_PAIR_NULL_POLAR2752_FP16":physical},str(weight))
    require(torch.equal(load_file(str(weight),device="cpu")[
                "S61_REAL43M_GT_PAIR_NULL_POLAR2752_FP16"],physical),
            "S61 trained physical FP16 readback mismatch")
    report={
       "stage":STAGE,
       "old_genuine_original43M_GT_CPU_source_run_ids":[
           44274389010,44279445613,44282672013,44308286512,
           44312765248,44324723824,44358677983,44362903794],
       "old_spent216_native_sha256":[sha(z) for z in paths],
       "original_S26_SHA256":sha(a.s26),
       "old_S57_paid231_SHA256":sha(a.s57),
       "old_S59_paid2046_SHA256":sha(a.s59),
       "new_S61_paid2752_SHA256":sha(weight),
       "new_S61_paid_FP16_count":PAID,
       "old216_train_labels":gold_count,
       "fitting":fit,
       "S61_NEW32_raw_and_gold_never_imported_in_spent216_fitting":True,
       "no_S38_S47_S48_EVAL32_EVAL128_protected_data":True,
       "original43M_GT_old_paid_model_SHA_unchanged":True,
       "active_S61_616J7_bpp_CB59_not_certified":True,
       "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (a.out/"NRA3_S61_SPENT216_PAIR_NONE2752_TRAIN_MANIFEST.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print("S61_GENUINE_CALIBRI43M_GT_SPENT216_PAIR_NONE2752_TRAIN",
          json.dumps({"sha256":sha(weight),"paid2752":PAID,
                      "training":gold_count,"gate":0},sort_keys=True),flush=True)

if __name__=="__main__":main()
