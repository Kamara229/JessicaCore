"""Physically fit S58 paid83 on exactly four already-SPENT genuine43M GT TRAIN24 sources.

No S57 spent dev24 or S58 NEW32 raw/gold are imported at any point.
Original 43M, Green Tunnel, frozen S26 and S57 paid231 are read-only.
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
    ARCHIVES,OLD_S26,word_idx,
)
from research.nra3_s20.s58_trainonly_event_transfer_residual83 import (
    S58EventTransferResidual,train_on_old_S53_S54_S55_S56_ONLY,N_PAID,
)

def require(ok,msg):
    if not bool(ok):raise ValueError("S58 physical old96 train: "+msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ("s53","s54","s55","s56","s26","s57","s57-fit","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    src=[a.s53,a.s54,a.s55,a.s56]
    require(not a.out.exists(),"refuse overwrite")
    require(all(f.is_file() and f.name==ref[1]
                for ref,f in zip(ARCHIVES,src)),"S53-56 archive names drift")
    require(a.s26.name==OLD_S26 and a.s26.is_file()
            and a.s57.name=="NRA3_S57_EVENTFIRST_SPENT96_ONLY_231FP16_UNACCEPTED.safetensors"
            and a.s57_fit.name=="NRA3_S57_EVENTFIRST_SPENT96_231FP16_TRAIN_REPORT.json"
            and a.s57.is_file() and a.s57_fit.is_file(),
            "original S26/S57 frozen weights missing")
    fit=json.loads(a.s57_fit.read_text())
    require(fit["stage"]=="NRA3_S57_OLD_SPENT_S53_S54_S55_S56_TRAIN96_REAL43M_GT_EVENTFIRST_231FP16"
            and fit["old_S26_12384_FP16_physical_archive_SHA256"]==sha(a.s26)
            and fit["new_event_first_FP16_weights_SHA256"]==sha(a.s57)
            and fit["old_training_4x24_real43M_GT_native_archive_sha256"]==
                [sha(q) for q in src]
            and fit["new_S57_first_HELD24_raw_and_manual_gold_NEVER_IMPORTED"],
            "source SHA provenance mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    old26=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(a.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    require(all(not p.requires_grad for m in (old26,old57)
                for p in m.parameters()),"old model unfrozen")
    old_shas=[
        tensor_sha(*(v.detach() for v in old26.parameters())),
        tensor_sha(*(v.detach() for v in old57.parameters())),
    ]
    from research.nra3_s20.s53_precommitted_negation_quote24_plain import SOURCE_ROWS as R53
    from research.nra3_s20.s53_separate_manual_gold_negation24 import gold24 as G53
    from research.nra3_s20.s54_frozen_new_held24_plain_source import SOURCE_ROWS as R54
    from research.nra3_s20.s54_frozen_new_held24_manual_gold import held_gold24 as G54
    from research.nra3_s20.s55_frozen_structural_shift_held24_plain import SOURCE_ROWS as R55
    from research.nra3_s20.s55_frozen_structural_shift_held24_manual_gold import held_gold24 as G55
    from research.nra3_s20.s56_frozen_new_scope24_plain import SOURCE_ROWS as R56
    from research.nra3_s20.s56_frozen_new_scope24_gold import held_gold24 as G56
    rows=(R53,R54,R55,R56)
    gold=(G53(),G54(),G55(),G56())
    require(all(len(r)==len(g)==24 for r,g in zip(rows,gold)),
            "exact old96 labels required")
    cases=[]
    for (tag,_,index_key),path,raw,manual in zip(ARCHIVES,src,rows,gold):
        data=load_file(str(path),device="cpu")
        require(set(data)=={
            "original_real43m_gt_word339_READONLY",
            "original_native_word_valid_mask",
            "native_BPE_causally_unsafe_word_mask",index_key},
            "old gold-free genuine43M source keys drift")
        x=data["original_real43m_gt_word339_READONLY"].detach().clone()
        valid=data["original_native_word_valid_mask"].detach().clone()
        unsafe=data["native_BPE_causally_unsafe_word_mask"].detach().clone()
        require(x.shape[0]==24 and x.shape[2]==339
                and valid.shape==unsafe.shape==x.shape[:2]
                and data[index_key].tolist()==list(range(24)),
                "old native ABI mismatched")
        with torch.no_grad():
            obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
            f=gold_blind_eventfirst_features(x,obs,tuple(r[2] for r in raw))
        # trainable FP16 delta, all frozen features must be ordinary tensors
        f=EventFirstInputs(f.event.detach().clone(),f.pair.detach().clone(),
                           f.candidate_event.detach().clone(),
                           f.valid_pair.detach().clone())
        index=torch.zeros(24,dtype=torch.long)
        transferred=torch.zeros(24,dtype=torch.bool)
        for k,(row,g) in enumerate(zip(raw,manual)):
            require(row[0]==g.row_id,"old spent gold row drift")
            words=raw_surface_grammar(row[2])
            predicate=g.predicate if tag in ("S53","S54") else g.main_predicate
            recipient=(g.recipient if tag in ("S53","S54") else
                       g.transferred_recipient)
            index[k]=word_idx(words,predicate)
            require(bool(f.candidate_event[k,index[k]]),
                    "old main predicate not in safe eligible source")
            if recipient is not None:
                j=word_idx(words,recipient)
                require(bool(f.valid_pair[k,index[k],j]),
                        "old directed true recipient unsafe")
                transferred[k]=True
        cases.append((f,index,transferred))
    model,physical,train_report=train_on_old_S53_S54_S55_S56_ONLY(cases,old57)
    require(tensor_sha(*(p.detach() for p in old26.parameters()))==old_shas[0]
            and tensor_sha(*(p.detach() for p in old57.parameters()))==old_shas[1],
            "old frozen paid base unexpectedly modified")
    a.out.mkdir(parents=True,exist_ok=False)
    out=a.out/"NRA3_S58_OLD96_TRAIN_ONLY_EVENT_TRANSFER_RESIDUAL83_FP16.safetensors"
    save_file({"S58_EVENT_TRANSFER_RESIDUAL83_FP16":physical},str(out))
    require(torch.equal(
        load_file(str(out))["S58_EVENT_TRANSFER_RESIDUAL83_FP16"],physical),
        "frozen model FP16 safetensors reload failure")
    report={
        "stage":"NRA3_S58_OLD96_REAL43M_GT_TRAIN_ONLY_EVENT_TRANSFER83",
        "training_source_NATIVE_S53_S54_S55_S56_SHA256":[sha(q) for q in src],
        "original_S26_FP16_SHA256":sha(a.s26),
        "old_original_S57_paid231_SHA256":sha(a.s57),
        "new_S58_paid83_SHA256":sha(out),
        "new_S58_parameters":N_PAID,
        "train_statistics":train_report,
        "new_S58_32_RAW_AND_GOLD_NEVER_IMPORTED":True,
        "S57_spent24_dev_NOT_IMPORTED_IN_TRAIN":True,
        "no_new_original43M_GT_CPU_or_new_J7":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (a.out/"NRA3_S58_TRAIN_ONLY_OLD96_RESIDUAL83_MANIFEST.json").write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("S58_REAL43M_GT_OLD96_TRAIN_ONLY_NEW_83FP16",
          json.dumps({"sha":sha(out),"old96":96,"params":N_PAID,
                      "held32_unseen":True,"gate":0},sort_keys=True),flush=True)
if __name__=="__main__":main()
