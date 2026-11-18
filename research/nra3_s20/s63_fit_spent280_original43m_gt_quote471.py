"""Fit S63 quoted outer-action residual471 using ONLY previously SPENT real43M GT 280.

OLD S53..S62 true physical post-layer8 original Calibri43M+GreenTunnel
word339, each prior CPU protected616J7 replay verified and source SHA
verified. Frozen S26/S57/S59/S61/S62 paid weights unchanged. S63 fresh
32 RAW or separately manually annotated GOLD NOT imported into training.
Six previous 0-based S62 BPE-unsafe indices excluded from new loss,
offline fractional BPE readout is NOT certified causal streaming safe.
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
from research.nra3_s20.s62_joint_event_target_beam2166 import (
    S62JointEventRecipient,source_only_frozen_beam,
)
from research.nra3_s20.s62_fit_spent248_real43m_gt_joint2166 import (
    FILES as OLD_NINE,old_spent248,
)
from research.nra3_s20.s60_fit_old184_true43m_gt_triclass2776 import (
    S57,S59,word_idx,
)
from research.nra3_s20.s63_quote_outer_strictbpe_residual471 import (
    quote_overlay,fit_PREVIOUSLY_SPENT280_only,FP16,
)

STAGE="NRA3_S63_GENUINE43M_GT_SPENT280_QUOTE_OUTER_STRICT_BPE471FP16"
S62_NATIVE="NRA3_S62_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
FILES=OLD_NINE+(("S62",S62_NATIVE,"source_row_indices_in_frozen_S62_HELD32"),)
S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
S61="NRA3_S61_REAL43M_GT_SPENT216_PAID2752_PAIR_NONE.safetensors"
S62="NRA3_S62_REAL43M_GT_SPENT248_JOINT_EVENT_TARGET2166_FP16.safetensors"
S57_FIT="NRA3_S57_EVENTFIRST_SPENT96_231FP16_TRAIN_REPORT.json"
S59_FIT="NRA3_S59_OLD152_TRAIN_NATIVE2046_FP16_MANIFEST.json"
S61_FIT="NRA3_S61_SPENT216_PAIR_NONE2752_TRAIN_MANIFEST.json"
S62_FIT="NRA3_S62_OLD248_JOINT2166_TRAIN_MANIFEST.json"

def require(ok,why):
    if not bool(ok):raise ValueError("S63 actual43M GT spent280 TRAIN: "+why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def old_spent280():
    sources,gold=old_spent248()
    from research.nra3_s20.s62_new32_preregistered_raw_only import SOURCE_ROWS as S62_RAW
    from research.nra3_s20.s62_new32_separate_manual_gold_NEVER_TRAIN import held_gold32 as S62_GOLD
    sources=sources+(S62_RAW,)
    gold=gold+(S62_GOLD(),)
    require([len(q) for q in sources]==[24,24,24,24,24,32,32,32,32,32]
            and [len(q) for q in gold]==[24,24,24,24,24,32,32,32,32,32],
            "S63 must train 280 old source rows exactly")
    return sources,gold

def main():
    p=argparse.ArgumentParser()
    for name in ("s53","s54","s55","s56","s57-native","s58-native",
                 "s59-native","s60-native","s61-native","s62-native",
                 "s57-prov","s58-prov","s59-prov","s60-prov","s61-prov","s62-prov",
                 "s26","s57","s57-fit","s59","s59-fit","s61","s61-fit",
                 "s62","s62-fit","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"never overwrite S63 source-only paid")
    paths=[a.s53,a.s54,a.s55,a.s56,a.s57_native,a.s58_native,
           a.s59_native,a.s60_native,a.s61_native,a.s62_native]
    require(all(x.is_file() and x.name==tag[1]
                for tag,x in zip(FILES,paths)),
            "original GT old S53..S62 native339 archive missing or modified")
    ancestors=((a.s26,S26),(a.s57,S57),(a.s59,S59),(a.s61,S61),
               (a.s62,S62),(a.s57_fit,S57_FIT),(a.s59_fit,S59_FIT),
               (a.s61_fit,S61_FIT),(a.s62_fit,S62_FIT))
    require(all(x.is_file() and x.name==expected for x,expected in ancestors),
            "S63 immutable old parent paid model/archive missing")
    provs=[json.loads(p.read_text()) for p in (
        a.s57_prov,a.s58_prov,a.s59_prov,a.s60_prov,a.s61_prov,a.s62_prov)]
    require([q["stage"] for q in provs]==[
        f"NRA3_{tag}_GENUINE43M_GT_EVENTFIRST{n}_BPE_SOURCE_ONLY"
        for tag,n in (("S57",24),("S58",32),("S59",32),
                      ("S60",32),("S61",32),("S62",32))]
        and all(q["derived_REAL43M_WORD339_SHA256"]==sha(path)
                and q["original_616_J7_replayed_before_read_only_quotes"]
                for q,path in zip(provs,paths[4:])),
            "prior S57-S62 authentic original 616 J7/native339 provenance changed")
    f57=json.loads(a.s57_fit.read_text())
    f59=json.loads(a.s59_fit.read_text())
    f61=json.loads(a.s61_fit.read_text())
    f62=json.loads(a.s62_fit.read_text())
    require(f57["new_event_first_FP16_weights_SHA256"]==sha(a.s57)
            and f59["new_S59_semantic_paid2046_sha256"]==sha(a.s59)
            and f61["new_S61_paid2752_SHA256"]==sha(a.s61)
            and f62["new_S62_joint2166_FP16_SHA256"]==sha(a.s62)
            and f62["original_S26_PAID12384_SHA256"]==sha(a.s26)
            and f62["frozen_S57_PAID231_SHA256"]==sha(a.s57)
            and f62["frozen_S59_PAID2046_SHA256"]==sha(a.s59)
            and f62["frozen_S61_PAID2752_SHA256"]==sha(a.s61),
            "physical source model S26/S57/S59/S61/S62 SHA mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    groups_raw,gold=old_spent280()
    old26=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(a.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    old59=S59NativeEventSemantic.from_fp16(
        load_file(str(a.s59),device="cpu")["S59_NATIVE320_SEMANTIC2046_FP16"])
    old61=S61NativePairNull.from_fp16(
        load_file(str(a.s61),device="cpu")["S61_REAL43M_GT_PAIR_NULL_POLAR2752_FP16"])
    old62=S62JointEventRecipient.from_fp16(
        load_file(str(a.s62),device="cpu")["S62_JOINT_EVENT_TARGET2166_FP16"])
    frozen=(old26,old57,old59,old61,old62)
    require(all(not p.requires_grad for z in frozen for p in z.parameters())
            and all(m.output_gate==0 and not m.candidate_accepted
                    for m in frozen),
            "S63 old original 43M GT and paid source must remain gate0")
    before=[tensor_sha(*(p.detach() for p in m.parameters())) for m in frozen]
    cases=[];counts=[]
    for (tag,_,index_key),path,rows,manual in zip(FILES,paths,groups_raw,gold):
        store=load_file(str(path),device="cpu")
        require(set(store)=={
            "original_real43m_gt_word339_READONLY",
            "original_native_word_valid_mask",
            "native_BPE_causally_unsafe_word_mask",index_key},
            "original 43M GT source tensor unexpectedly includes labels")
        x=store["original_real43m_gt_word339_READONLY"].detach().clone()
        valid=store["original_native_word_valid_mask"].detach().clone()
        unsafe=store["native_BPE_causally_unsafe_word_mask"].detach().clone()
        n=len(rows);w=x.shape[1]
        require(x.shape==(n,w,339)
                and valid.shape==unsafe.shape==(n,w)
                and store[index_key].tolist()==list(range(n)),
                "S63 source-only genuine CPU original BPE index shape")
        checksum=tensor_sha(x,valid,unsafe)
        with torch.no_grad():
            obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
            event=gold_blind_eventfirst_features(
                x,obs,tuple(row[2] for row in rows))
            sem=semantic_features(x,event)
            fused=source_local_scope(sem,tuple(row[2] for row in rows))
            old_logits=tuple(z.detach() for z in old57(event))
            beam=source_only_frozen_beam(fused,old_logits,old59,old61)
            overlay=quote_overlay(beam,tuple(row[2] for row in rows),unsafe)
        # Maintain standard torch grad autograd inputs outside inference contexts.
        require(not overlay.source_quote.is_inference()
                and not overlay.pair_quote.is_inference()
                and not overlay.source_quote.requires_grad
                and not overlay.pair_quote.requires_grad,
                "S63 input saved old GT should be graph-free")
        event_idx=torch.zeros((n,),dtype=torch.long)
        target=torch.full((n,),w,dtype=torch.long)
        rejected=torch.zeros((n,),dtype=torch.bool)
        for i,(raw,g) in enumerate(zip(rows,manual)):
            require(raw[0]==g.row_id,"old spent280 gold source row alignment changed")
            tokens=raw_surface_grammar(raw[2])
            require(len(tokens)==int(valid[i].sum()),
                    "original 43M true native word offset/token count mismatch")
            name=g.predicate if tag in ("S53","S54") else g.main_predicate
            rec=g.recipient if tag in ("S53","S54") else g.transferred_recipient
            pol=(g.recipient_polarity if tag in ("S53","S54") else
                 g.transfer_polarity if tag=="S55" else
                 g.directed_transfer_polarity)
            src=word_idx(tokens,name)
            require(bool(event.candidate_event[i,src]),"gold main event not eligible old")
            event_idx[i]=src
            if rec is not None:
                dest=word_idx(tokens,rec)
                require(bool(event.valid_pair[i,src,dest]),
                        "gold directed old S62 pair not BPE/quote safe")
                target[i]=dest
                rejected[i]=(pol=="REJECTED")
        require(tensor_sha(x,valid,unsafe)==checksum,
                "frozen original Calibri43M GT hidden representation changed")
        cases.append((overlay,event_idx,target,rejected))
        counts.append({"group":tag,"rows":n,"directed":int((target!=w).sum()),
                       "rejected":int(rejected.sum()),
                       "native_unsafe_rows":int(overlay.unsafe_row.sum()),
                       "source_gold_in_beam8":int(
                           ((overlay.original.sources==event_idx[:,None])
                            &overlay.original.allowed).any(-1).sum())})
    candidate,z,fit=fit_PREVIOUSLY_SPENT280_only(cases,old62)
    require(z.shape==(FP16,) and z.dtype==torch.float16
            and candidate.output_gate==0 and not candidate.candidate_accepted
            and not candidate.activation_allowed
            and all(not p.requires_grad for p in candidate.parameters())
            and all(tensor_sha(*(p.detach() for p in m.parameters()))==s
                    for m,s in zip(frozen,before)),
            "new S63 FP16 or original Calibri GT previous readout weights changed")
    a.out.mkdir(parents=True,exist_ok=False)
    weights=a.out/"NRA3_S63_REAL43M_GT_SPENT280_QUOTE_OUTER_BPE471_FP16.safetensors"
    save_file({"S63_QUOTE_OUTER_STRICTBPE471_FP16":z},str(weights))
    require(torch.equal(load_file(str(weights),device="cpu")[
                "S63_QUOTE_OUTER_STRICTBPE471_FP16"],z),
            "S63 trained FP16 checkpoint is not physically replayable")
    report={
        "stage":STAGE,
        "spent_10_prior_original43M_GT_postlayer8_sources_SHA256":[sha(p) for p in paths],
        "original_S26_PAID12384_SHA256":sha(a.s26),
        "old_S57_PAID231_SHA256":sha(a.s57),
        "old_S59_PAID2046_SHA256":sha(a.s59),
        "old_S61_PAID2752_SHA256":sha(a.s61),
        "old_S62_PAID2166_SHA256":sha(a.s62),
        "new_S63_PAID471_SHA256":sha(weights),
        "new_S63_FP16_parameters":FP16,
        "old_280_spent_training_gold_counts":counts,
        "training":fit,
        "new_S63_NEW32_raw_gold_NEVER_IMPORTED_TO_OLD280_TRAIN":True,
        "original_Calibri43M_GT_and_all_old_paids_unchanged":True,
        "strict_BPE_invalid_rows_excluded_from_new_adapter_train":True,
        "not_populational_or_general_unbiased_eval":True,
        "original_616J7_baseline_ONLY_not_S63_integrated_candidate":True,
        "active_streaming_J7_bpp_CPU_CB59_not_certified":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (a.out/"NRA3_S63_SPENT280_QUOTE471_TRAIN_MANIFEST.json").write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("S63_REAL43M_GT_SPENT280_QUOTE_OUTER471_TRAIN",
          json.dumps({"physical_SHA256":sha(weights),"old_counts":counts,
                      "training":fit,"gate":0},sort_keys=True),flush=True)
if __name__=="__main__":main()
