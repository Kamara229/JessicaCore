"""Physically fit S65 259 FP16 on ONLY prior SPENT344 original43M+GT.

S53..S64: genuine archived protected original CPU/GT post-layer8 word339;
all ten/eleven original protected 616 J7 provenance, immutable source SHA.
Old S63 quote471, S62 joint2166, S61 pair2752, S59 native2046, S57
event231, S26 graph12384 all frozen with read-back physical hashes.
Separate NEW S65 32 raw and manual gold are NEVER IMPORTED OR FITTED.
S64 data is already SPENT, may be used for OLD344 TRAIN ONLY.
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
from research.nra3_s20.s61_native_pair_null_polarity_2752 import S61NativePairNull
from research.nra3_s20.s62_joint_event_target_beam2166 import (
    S62JointEventRecipient,source_only_frozen_beam,
)
from research.nra3_s20.s63_quote_outer_strictbpe_residual471 import (
    S63QuoteOuterSafe,quote_overlay,
)
from research.nra3_s20.s63_fit_spent280_original43m_gt_quote471 import (
    FILES as OLD_TEN,old_spent280,
)
from research.nra3_s20.s60_fit_old184_true43m_gt_triclass2776 import (
    S57,S59,word_idx,
)
from research.nra3_s20.s65_factuality_attributed_event259 import (
    discourse_features,fit_SPENT344_only,PAID,
)

STAGE="NRA3_S65_REAL43M_GT_SPENT344_REPORTED_FACTUALITY3WAY259_FP16_TRAIN"
S63_NATIVE="NRA3_S63_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
S64_NATIVE="NRA3_S64_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
FILES=OLD_TEN+(("S63",S63_NATIVE,"source_row_indices_in_frozen_S63_HELD32"),
               ("S64",S64_NATIVE,"source_row_indices_in_frozen_S64_HELD32"),)
S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
S61="NRA3_S61_REAL43M_GT_SPENT216_PAID2752_PAIR_NONE.safetensors"
S62="NRA3_S62_REAL43M_GT_SPENT248_JOINT_EVENT_TARGET2166_FP16.safetensors"
S63="NRA3_S63_REAL43M_GT_SPENT280_QUOTE_OUTER_BPE471_FP16.safetensors"
S57_FIT="NRA3_S57_EVENTFIRST_SPENT96_231FP16_TRAIN_REPORT.json"
S59_FIT="NRA3_S59_OLD152_TRAIN_NATIVE2046_FP16_MANIFEST.json"
S61_FIT="NRA3_S61_SPENT216_PAIR_NONE2752_TRAIN_MANIFEST.json"
S62_FIT="NRA3_S62_OLD248_JOINT2166_TRAIN_MANIFEST.json"
S63_FIT="NRA3_S63_SPENT280_QUOTE471_TRAIN_MANIFEST.json"
S65_WEIGHT="NRA3_S65_REAL43M_GT_SPENT344_REPORTED3WAY259_FP16.safetensors"
S65_MANIFEST="NRA3_S65_REAL43M_GT_SPENT344_FACTUALITY259_MANIFEST.json"

def require(ok,why):
    if not bool(ok):raise ValueError("S65 true43M GT OLD344 fitting: "+why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def old_spent312():
    rows,gold=old_spent280()
    from research.nra3_s20.s63_new32_raw_only_precommitted import SOURCE_ROWS as R63
    from research.nra3_s20.s63_new32_separate_manual_gold_NEVER_TRAIN import held_gold32 as G63
    from research.nra3_s20.s64_new32_raw_only_preregistered import SOURCE_ROWS as R64
    from research.nra3_s20.s64_new32_separate_manual_gold_NEVER_TRAIN import held_gold32 as G64
    rows=rows+(R63,R64)
    gold=gold+(G63(),G64())
    require([len(q) for q in rows]==[24]*5+[32]*7
            and [len(q) for q in gold]==[24]*5+[32]*7,
            "old344 must be exactly S53-S64 already scored groups")
    return rows,gold

def main():
    p=argparse.ArgumentParser()
    for name in ("s53","s54","s55","s56","s57-native","s58-native",
                 "s59-native","s60-native","s61-native","s62-native","s63-native","s64-native",
                 "s57-prov","s58-prov","s59-prov","s60-prov","s61-prov","s62-prov","s63-prov","s64-prov",
                 "s26","s57","s57-fit","s59","s59-fit","s61","s61-fit",
                 "s62","s62-fit","s63","s63-fit","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"refuse overwriting old344 trained FP16 physical state")
    paths=[a.s53,a.s54,a.s55,a.s56,a.s57_native,a.s58_native,
           a.s59_native,a.s60_native,a.s61_native,a.s62_native,a.s63_native,a.s64_native]
    require(all(path.is_file() and path.name==meta[1]
                for path,meta in zip(paths,FILES)),
            "S65 source archive must be true original Calibri43M+GT REAL native339")
    models=((a.s26,S26),(a.s57,S57),(a.s59,S59),(a.s61,S61),
            (a.s62,S62),(a.s63,S63),(a.s57_fit,S57_FIT),
            (a.s59_fit,S59_FIT),(a.s61_fit,S61_FIT),
            (a.s62_fit,S62_FIT),(a.s63_fit,S63_FIT))
    require(all(path.is_file() and path.name==name for path,name in models),
            "S65 old frozen paid physical parent missing")
    prov=[json.loads(z.read_text()) for z in (
        a.s57_prov,a.s58_prov,a.s59_prov,a.s60_prov,
        a.s61_prov,a.s62_prov,a.s63_prov,a.s64_prov)]
    require([z["stage"] for z in prov]==[
        f"NRA3_{tag}_GENUINE43M_GT_EVENTFIRST{size}_BPE_SOURCE_ONLY"
        for tag,size in (("S57",24),("S58",32),("S59",32),
                         ("S60",32),("S61",32),("S62",32),("S63",32),("S64",32))]
        and all(z["derived_REAL43M_WORD339_SHA256"]==sha(path)
                and z["original_616_J7_replayed_before_read_only_quotes"]
                for z,path in zip(prov,paths[4:])),
            "S65 old S57-S64 genuine CPU 616J7/G-T native SHA drift")
    fit57=json.loads(a.s57_fit.read_text())
    fit59=json.loads(a.s59_fit.read_text())
    fit61=json.loads(a.s61_fit.read_text())
    fit62=json.loads(a.s62_fit.read_text())
    fit63=json.loads(a.s63_fit.read_text())
    require(fit57["new_event_first_FP16_weights_SHA256"]==sha(a.s57)
            and fit59["new_S59_semantic_paid2046_sha256"]==sha(a.s59)
            and fit61["new_S61_paid2752_SHA256"]==sha(a.s61)
            and fit62["new_S62_joint2166_FP16_SHA256"]==sha(a.s62)
            and fit63["new_S63_PAID471_SHA256"]==sha(a.s63)
            and fit63["original_S26_PAID12384_SHA256"]==sha(a.s26)
            and fit63["old_S57_PAID231_SHA256"]==sha(a.s57)
            and fit63["old_S59_PAID2046_SHA256"]==sha(a.s59)
            and fit63["old_S61_PAID2752_SHA256"]==sha(a.s61)
            and fit63["old_S62_PAID2166_SHA256"]==sha(a.s62),
            "S65 source parent S26/S57/S59/S61/S62/S63 SHA immutable ancestry")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows,gold=old_spent312()
    old26=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(a.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    old59=S59NativeEventSemantic.from_fp16(
        load_file(str(a.s59),device="cpu")["S59_NATIVE320_SEMANTIC2046_FP16"])
    old61=S61NativePairNull.from_fp16(
        load_file(str(a.s61),device="cpu")["S61_REAL43M_GT_PAIR_NULL_POLAR2752_FP16"])
    old62=S62JointEventRecipient.from_fp16(
        load_file(str(a.s62),device="cpu")["S62_JOINT_EVENT_TARGET2166_FP16"])
    old63=S63QuoteOuterSafe.from_fp16(
        load_file(str(a.s63),device="cpu")["S63_QUOTE_OUTER_STRICTBPE471_FP16"])
    old=(old26,old57,old59,old61,old62,old63)
    require(all(not z.requires_grad for model in old for z in model.parameters())
            and all(model.output_gate==0 and not model.candidate_accepted
                    for model in old),
            "all previous base43M GT and paid heads strictly gate0/frozen")
    oldhash=[tensor_sha(*(p.detach() for p in z.parameters())) for z in old]
    groups=[];per=[]
    for (tag,_,key),path,sources,manual in zip(FILES,paths,rows,gold):
        archive=load_file(str(path),device="cpu")
        require(set(archive)=={"original_real43m_gt_word339_READONLY",
               "original_native_word_valid_mask",
               "native_BPE_causally_unsafe_word_mask",key},
               "prior true original43M native cannot contain gold or roles")
        x=archive["original_real43m_gt_word339_READONLY"].detach().clone()
        valid=archive["original_native_word_valid_mask"].detach().clone()
        unsafe=archive["native_BPE_causally_unsafe_word_mask"].detach().clone()
        n=len(sources);w=x.shape[1]
        require(x.shape==(n,w,339) and
                valid.shape==unsafe.shape==(n,w)
                and archive[key].tolist()==list(range(n)),
                "old344 original CPU BPE/genuine word/native source shape")
        frozen_sha=tensor_sha(x,valid,unsafe)
        with torch.no_grad():
            obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
            ev=gold_blind_eventfirst_features(
                x,obs,tuple(z[2] for z in sources))
            sem=semantic_features(x,ev)
            fused=source_local_scope(sem,tuple(z[2] for z in sources))
            oldlogits=tuple(z.detach() for z in old57(ev))
            beam=source_only_frozen_beam(fused,oldlogits,old59,old61)
            overlay=quote_overlay(beam,tuple(z[2] for z in sources),unsafe)
            state=discourse_features(overlay,tuple(z[2] for z in sources))
        require(not state.factual_source.is_inference()
                and not state.factual_source.requires_grad,
                "S65 original CPU native source is inference-only tensor")
        predicate=torch.zeros(n,dtype=torch.long)
        recipients=torch.full((n,),w,dtype=torch.long)
        rejected=torch.zeros((n,),dtype=torch.bool)
        for i,(record,g) in enumerate(zip(sources,manual)):
            require(record[0]==g.row_id,"S65 OLD344 source manual row drift")
            words=raw_surface_grammar(record[2])
            require(len(words)==int(valid[i].sum()),
                    "old real native Calibri43M word boundary drift")
            word=(g.predicate if tag in ("S53","S54") else g.main_predicate)
            target=(g.recipient if tag in ("S53","S54")
                    else g.transferred_recipient)
            polar=(g.recipient_polarity if tag in ("S53","S54")
                   else g.transfer_polarity if tag=="S55"
                   else g.directed_transfer_polarity)
            src=word_idx(words,word)
            require(bool(ev.candidate_event[i,src]),"old gold event illegal")
            predicate[i]=src
            if target is not None:
                dest=word_idx(words,target)
                require(bool(ev.valid_pair[i,src,dest]),
                        "old gold pair original BPE pair not eligible")
                recipients[i]=dest
                rejected[i]=(polar=="REJECTED")
        require(tensor_sha(x,valid,unsafe)==frozen_sha,
                "S65 original native source tensor altered by readout")
        groups.append((state,predicate,recipients,rejected))
        per.append({"stage":tag,"old_rows":n,"true_directed":int((recipients!=w).sum()),
                    "true_rejected":int(rejected.sum()),
                    "native_or_unbalanced_quote_unsafe":int(state.strict_native_or_quote_unsafe.sum())})
    candidate,z,stats=fit_SPENT344_only(groups,old62,old63)
    require(z.shape==(PAID,) and z.dtype==torch.float16
            and not candidate.candidate_accepted and not candidate.activation_allowed
            and candidate.output_gate==0
            and all(not p.requires_grad for p in candidate.parameters())
            and all(tensor_sha(*(p.detach() for p in obj.parameters()))==h
                    for obj,h in zip(old,oldhash)),
            "S65 physically saved259 or frozen Calibri/GT parents modified")
    a.out.mkdir(parents=True,exist_ok=False)
    paid=a.out/S65_WEIGHT
    save_file({"S65_ATTRIBUTED_3WAY259_FP16":z},str(paid))
    require(torch.equal(load_file(str(paid),device="cpu")[
                "S65_ATTRIBUTED_3WAY259_FP16"],z),
            "S65 actual physical FP16 safetensors readback differs")
    report={
       "stage":STAGE,
       "previously_SPENT_original43M_GT_native339_SHA256":[sha(p) for p in paths],
       "frozen_S26_paid_SHA256":sha(a.s26),
       "frozen_S57_paid_SHA256":sha(a.s57),
       "frozen_S59_paid_SHA256":sha(a.s59),
       "frozen_S61_paid_SHA256":sha(a.s61),
       "frozen_S62_paid_SHA256":sha(a.s62),
       "frozen_S63_paid_SHA256":sha(a.s63),
       "new_S65_physically_fitted259FP16_SHA256":sha(paid),
       "new_S65_FP16_parameter_count":PAID,
       "old_344_spent_annotation_counts":per,
       "training":stats,
       "new_S65_NEW32_raw_and_gold_NEVER_IMPORTED_IN_OLD344_fitting":True,
       "original_Calibri43M_GT_and_older_paid_unchanged":True,
       "protected_original_616J7_BASE_only_not_active_candidate":True,
       "S65_research_only_not_autoregressive_online_causal_certified":True,
       "no_protected_EVAL32_EVAL128_S38_S47_S48":True,
       "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (a.out/S65_MANIFEST).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("S65_REAL43M_GT_SPENT344_REPORTED_FACTUALITY259_PHYSICAL_TRAIN",
          json.dumps({"sha256":sha(paid),"training":stats,
                      "source":per,"gate":0},sort_keys=True),flush=True)
if __name__=="__main__":main()
