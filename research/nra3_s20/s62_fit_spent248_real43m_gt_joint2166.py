"""S62 physical original Calibri43M+GreenTunnel OLD248 TRAIN-ONLY joint fit.

Exactly S53..S61 genuine previous post-layer8 CPU/GT word339 archives.
All OLD248 annotated examples are *permanently spent*, only TRAIN.
Original 43M, GT, S26, S57, S59, S61 frozen. S62 first NEW32 raw/gold
not imported, not used for topology, training, thresholds or reporting.
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
from research.nra3_s20.s60_native43m_local_triclass_model import (
    S60Input,source_local_scope,
)
from research.nra3_s20.s61_native_pair_null_polarity_2752 import (
    S61NativePairNull,
)
from research.nra3_s20.s61_fit_spent216_original43m_gt_pair2752 import (
    FILES as OLD_EIGHT,load_old_spent216,
)
from research.nra3_s20.s60_fit_old184_true43m_gt_triclass2776 import (
    S57,S59,word_idx,
)
from research.nra3_s20.s62_joint_event_target_beam2166 import (
    source_only_frozen_beam,fit_SPENT248_only,PAID,
)

STAGE="NRA3_S62_GENUINE43M_GT_SPENT248_JOINT_EVENT_TARGET_BEAM2166_TRAIN"
S61_NATIVE="NRA3_S61_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
FILES=OLD_EIGHT+(("S61",S61_NATIVE,"source_row_indices_in_frozen_S61_HELD32"),)
S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
S61="NRA3_S61_REAL43M_GT_SPENT216_PAID2752_PAIR_NONE.safetensors"
S61_FIT="NRA3_S61_SPENT216_PAIR_NONE2752_TRAIN_MANIFEST.json"
S57_FIT="NRA3_S57_EVENTFIRST_SPENT96_231FP16_TRAIN_REPORT.json"
S59_FIT="NRA3_S59_OLD152_TRAIN_NATIVE2046_FP16_MANIFEST.json"

def require(ok,msg):
    if not bool(ok):raise ValueError("S62 original43M GT old248 only: "+msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def old_spent248():
    row,gold=load_old_spent216()
    from research.nra3_s20.s61_new32_raw_only_preregistered import SOURCE_ROWS as R61
    from research.nra3_s20.s61_new32_separate_frozen_gold_DO_NOT_TRAIN import held_gold32 as G61
    row=row+(R61,)
    gold=gold+(G61(),)
    require([len(q) for q in row]==[24,24,24,24,24,32,32,32,32]
            and [len(q) for q in gold]==[24,24,24,24,24,32,32,32,32],
            "old248 must be exactly nine independently spent groups")
    return row,gold

def main():
    p=argparse.ArgumentParser()
    for name in ("s53","s54","s55","s56","s57-native","s58-native",
                 "s59-native","s60-native","s61-native",
                 "s57-prov","s58-prov","s59-prov","s60-prov","s61-prov",
                 "s26","s57","s57-fit","s59","s59-fit","s61","s61-fit","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"refuse overwriting physical S62 FP16")
    paths=[a.s53,a.s54,a.s55,a.s56,a.s57_native,a.s58_native,
           a.s59_native,a.s60_native,a.s61_native]
    require(all(path.is_file() and path.name==tag[1]
                for tag,path in zip(FILES,paths)),
            "S62 requires original S53..S61 true43M GT archived tensors")
    require(all(p.is_file() and p.name==tag for p,tag in (
        (a.s26,S26),(a.s57,S57),(a.s59,S59),(a.s61,S61),
        (a.s57_fit,S57_FIT),(a.s59_fit,S59_FIT),(a.s61_fit,S61_FIT))),
            "original frozen S26/S57/S59/S61 missing or unexpected file")
    prov=[json.loads(x.read_text()) for x in (
        a.s57_prov,a.s58_prov,a.s59_prov,a.s60_prov,a.s61_prov)]
    require([z["stage"] for z in prov]==[
        f"NRA3_{tag}_GENUINE43M_GT_EVENTFIRST{n}_BPE_SOURCE_ONLY"
        for tag,n in (("S57",24),("S58",32),("S59",32),("S60",32),("S61",32))]
        and all(z["derived_REAL43M_WORD339_SHA256"]==sha(x)
                and z["original_616_J7_replayed_before_read_only_quotes"]
                for z,x in zip(prov,paths[4:])),
            "real43M GT prior J7/native SHA ancestry drift")
    fit57=json.loads(a.s57_fit.read_text())
    fit59=json.loads(a.s59_fit.read_text())
    fit61=json.loads(a.s61_fit.read_text())
    require(fit57["new_event_first_FP16_weights_SHA256"]==sha(a.s57)
            and fit59["new_S59_semantic_paid2046_sha256"]==sha(a.s59)
            and fit61["new_S61_paid2752_SHA256"]==sha(a.s61)
            and fit61["original_S26_SHA256"]==sha(a.s26)
            and fit61["old_S57_paid231_SHA256"]==sha(a.s57)
            and fit61["old_S59_paid2046_SHA256"]==sha(a.s59),
            "real original43M S26/S57/S59/S61 frozen paid ancestry failed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    oldrow,gold=old_spent248()
    old26=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(a.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    old59=S59NativeEventSemantic.from_fp16(
        load_file(str(a.s59),device="cpu")["S59_NATIVE320_SEMANTIC2046_FP16"])
    old61=S61NativePairNull.from_fp16(
        load_file(str(a.s61),device="cpu")["S61_REAL43M_GT_PAIR_NULL_POLAR2752_FP16"])
    originals=(old26,old57,old59,old61)
    require(all(not p.requires_grad for m in originals for p in m.parameters())
            and all(m.output_gate==0 and not m.candidate_accepted
                    for m in originals),"all original 43M GT and paid mechanisms gate0")
    old_hash=[tensor_sha(*(z.detach() for z in m.parameters())) for m in originals]
    groups=[];counts=[]
    for (tag,_,idx_key),path,rows,manual in zip(FILES,paths,oldrow,gold):
        archive=load_file(str(path),device="cpu")
        require(set(archive)=={
            "original_real43m_gt_word339_READONLY",
            "original_native_word_valid_mask",
            "native_BPE_causally_unsafe_word_mask",idx_key},
            "S62 original GT source archived tensor contains gold/extra keys")
        x=archive["original_real43m_gt_word339_READONLY"].detach().clone()
        valid=archive["original_native_word_valid_mask"].detach().clone()
        unsafe=archive["native_BPE_causally_unsafe_word_mask"].detach().clone()
        n=len(rows)
        require(x.ndim==3 and x.shape[0]==n and x.shape[2]==339
                and valid.shape==unsafe.shape==x.shape[:2]
                and archive[idx_key].tolist()==list(range(n)),
                "S62 archived real GT word339 source or BPE alignment changed")
        unchanged=tensor_sha(x,valid,unsafe)
        with torch.no_grad():
            obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
            feat=gold_blind_eventfirst_features(x,obs,tuple(q[2] for q in rows))
            sem=semantic_features(x,feat)
            inp=source_local_scope(sem,tuple(q[2] for q in rows))
        feat=EventFirstInputs(feat.event.detach().clone(),
                              feat.pair.detach().clone(),
                              feat.candidate_event.detach().clone(),
                              feat.valid_pair.detach().clone())
        sem=SemanticInputs(feat,sem.semantic.detach().clone())
        inp=S60Input(sem,inp.scope.detach().clone(),inp.fused.detach().clone())
        require(not inp.fused.is_inference(),"S62 OLD248 autograd input in inference mode")
        with torch.no_grad():
            logit=tuple(q.detach() for q in old57(feat))
            beam=source_only_frozen_beam(inp,logit,old59,old61)
        event=torch.zeros(n,dtype=torch.long)
        recipient=torch.full((n,),x.shape[1],dtype=torch.long)
        neg=torch.zeros(n,dtype=torch.bool)
        for row,(raw,g) in enumerate(zip(rows,manual)):
            require(raw[0]==g.row_id,"original OLD248 annotated row mismatch")
            words=raw_surface_grammar(raw[2])
            require(len(words)==int(valid[row].sum()),
                    "original native 43M word-to-gold alignment failed")
            pred=(g.predicate if tag in ("S53","S54") else g.main_predicate)
            dest=(g.recipient if tag in ("S53","S54") else g.transferred_recipient)
            pol=(g.recipient_polarity if tag in ("S53","S54") else
                 g.transfer_polarity if tag=="S55" else
                 g.directed_transfer_polarity)
            src=word_idx(words,pred)
            require(bool(feat.candidate_event[row,src]),"true old gold source masked")
            event[row]=src
            if dest is not None:
                d=word_idx(words,dest)
                require(bool(feat.valid_pair[row,src,d]),
                        "old true recipient source-target BPE/quote unsafe")
                recipient[row]=d
                neg[row]=(pol=="REJECTED")
        require(tensor_sha(x,valid,unsafe)==unchanged,
                "original 43M GT hidden arrays mutated during source-only read")
        groups.append((beam,event,recipient,neg))
        counts.append({"tag":tag,"n":n,
                       "directed":int((recipient!=x.shape[1]).sum()),
                       "rejected":int(neg.sum()),
                       "gold_source_in_frozen_beam8":int(
                          ((beam.sources==event[:,None])&beam.allowed).any(-1).sum())})
    readout,z,stats=fit_SPENT248_only(groups)
    require(z.shape==(PAID,) and z.dtype==torch.float16
            and readout.output_gate==0 and not readout.candidate_accepted
            and not readout.activation_allowed
            and all(not p.requires_grad for p in readout.parameters())
            and all(tensor_sha(*(q.detach() for q in m.parameters()))==h
                    for m,h in zip(originals,old_hash)),
            "S62 FP16 physical model or original 43M paid weights modified")
    a.out.mkdir(parents=True,exist_ok=False)
    weight=a.out/"NRA3_S62_REAL43M_GT_SPENT248_JOINT_EVENT_TARGET2166_FP16.safetensors"
    save_file({"S62_JOINT_EVENT_TARGET2166_FP16":z},str(weight))
    require(torch.equal(load_file(str(weight),device="cpu")[
                "S62_JOINT_EVENT_TARGET2166_FP16"],z),
            "S62 physical FP16 readback differs")
    report={
        "stage":STAGE,
        "all_9_original43M_GT_previously_spent_NATIVE339_SHA256":[sha(p) for p in paths],
        "original_S26_PAID12384_SHA256":sha(a.s26),
        "frozen_S57_PAID231_SHA256":sha(a.s57),
        "frozen_S59_PAID2046_SHA256":sha(a.s59),
        "frozen_S61_PAID2752_SHA256":sha(a.s61),
        "new_S62_joint2166_FP16_SHA256":sha(weight),
        "new_S62_paid_fp16_parameters":PAID,
        "old248_per_group_annotated_gold_counts":counts,
        "fit":stats,
        "S62_NEW32_RAW_AND_GOLD_NOT_IMPORTED_IN_TRAIN":True,
        "protected_S38_S47_S48_EVAL32_128_NEVER_TOUCHED":True,
        "original_43M_GT_and_old_readouts_unchanged":True,
        "active_616J7_bpp_CPU_CB59_streaming_not_certified":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (a.out/"NRA3_S62_OLD248_JOINT2166_TRAIN_MANIFEST.json").write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("S62_GENUINE43M_GT_OLD248_JOINT2166_PHYSICAL_TRAIN",
          json.dumps({"SHA256":sha(weight),"train":stats,"groups":counts,
                      "gate":0},sort_keys=True),flush=True)

if __name__=="__main__":main()
