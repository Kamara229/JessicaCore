"""S61 true ORIGINAL 12-layer Calibri43M GT NEW32 FIRST-READ manual audit.

S61 weights physically fitted on previous SPENT216 and SHA frozen.
Original 616 J7 replay on unchanged 43M+GT must precede NEW32 CPU
gold-blind native source capture. S57/S59/S60/S61 predictions are all
made and frozen BEFORE the separately precommitted NEW32 manual gold
module is imported. Oracle-source diagnostic only AFTER scored first
read; it is not a deployable metric and cannot select S61.
No original model/GT/paid weights touched, no external LIVE access.
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
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    S59NativeEventSemantic,semantic_features,
)
from research.nra3_s20.s60_native43m_local_triclass_model import (
    S60Joint3,source_local_scope,
)
from research.nra3_s20.s61_native_pair_null_polarity_2752 import (
    S61NativePairNull,PAID,
)
from research.nra3_s20.s61_new32_raw_only_preregistered import (
    SOURCE_ROWS,FAMILIES,raw_only_inputs,raw_sha256,
)

STAGE="NRA3_S61_GENUINE_ORIGINAL43M_GT_NEW32_FIRSTREAD_PAIR_NULL2752"
NATIVE="NRA3_S61_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
PROV="NRA3_S61_GENUINE43M_GT_EVENTFIRST32_OFFSET_PROVENANCE.json"
CAPTURE="NRA3_S61_GENUINE43M_GT_EVENTFIRST32_NATIVE_J7_REPORT.json"
S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
S57="NRA3_S57_EVENTFIRST_SPENT96_ONLY_231FP16_UNACCEPTED.safetensors"
S59="NRA3_S59_REAL43M_GT_OLD152_NATIVE320_SEMANTIC2046_FP16.safetensors"
S60="NRA3_S60_ORIGINAL43M_GT_OLD184_TRICLASS2776_FP16.safetensors"
S61="NRA3_S61_REAL43M_GT_SPENT216_PAID2752_PAIR_NONE.safetensors"
FIT59="NRA3_S59_OLD152_TRAIN_NATIVE2046_FP16_MANIFEST.json"
FIT60="NRA3_S60_OLD184_TRAIN_TRICLASS2776_MANIFEST.json"
FIT61="NRA3_S61_SPENT216_PAIR_NONE2752_TRAIN_MANIFEST.json"

def require(ok,msg):
    if not bool(ok): raise ValueError("S61 true43M GT new32 first read: "+msg)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def score(pred,ground):
    tp=int((pred&ground).sum());fp=int((pred&~ground).sum())
    fn=int((~pred&ground).sum());d=2*tp+fp+fn
    return {"TP":tp,"FP":fp,"FN":fn,
            "precision":tp/(tp+fp) if tp+fp else 0.,
            "recall":tp/(tp+fn) if tp+fn else 0.,
            "F1":2*tp/d if d else 0.}

def main():
    p=argparse.ArgumentParser()
    for key in ("native","prov","capture","s26","s57","s59","fit59",
                "s60","fit60","s61","fit61","out"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"do not rescore or overwrite first NEW32")
    files=(a.native,a.prov,a.capture,a.s26,a.s57,a.s59,a.fit59,
           a.s60,a.fit60,a.s61,a.fit61)
    names=(NATIVE,PROV,CAPTURE,S26,S57,S59,FIT59,S60,FIT60,S61,FIT61)
    require(all(x.is_file() and x.name==name for x,name in zip(files,names)),
            "actual frozen 43M GT and FP16 archive names unexpected")
    prov=json.loads(a.prov.read_text())
    cap=json.loads(a.capture.read_text())
    fit59=json.loads(a.fit59.read_text())
    fit60=json.loads(a.fit60.read_text())
    fit61=json.loads(a.fit61.read_text())
    require(
        cap["stage"]=="NRA3_S61_GENUINE43M_GT_EVENTFIRST32_NATIVE_CPU_FIRST_PASS"
        and cap["S61_FROZEN_HELD32_PLAINTEXT_SHA256"]==raw_sha256()
        and cap["S61_HELD32_SOURCE_COMMIT"]==
            "02f8bb787a309b065c09e1f2f881bc2ef8e1f3d7"
        and cap["S61_HELD32_SOURCE_BLOB"]==
            "5e0a40c2cdcfb95f7eea9e91879e68211c3b0afe"
        and cap["all_616_native_CPU_generated_token_digests_unchanged"]
        and cap["original_456_full_EOS"]==456
        and cap["original_new128_full_EOS"]==128
        and cap["actual_43M_CPU_12_transformer_layers"]
        and cap["GT_residuals_before_readport_in_same_active_S93_scope"]
        and cap["actual_genuine_43M_CPU_new_HELD32_prefills"]==32
        and cap["S61_HELD_manual_gold_NOT_IMPORTED_into_model_capture"]
        and cap["NRA3_output_gate"]==0 and not cap["NRA3_candidate_accepted"]
        and prov["stage"]=="NRA3_S61_GENUINE43M_GT_EVENTFIRST32_BPE_SOURCE_ONLY"
        and prov["frozen_S61_HELD32_raw_source_sha256"]==raw_sha256()
        and prov["original_616_J7_replayed_before_read_only_quotes"]
        and prov["derived_REAL43M_WORD339_SHA256"]==sha(a.native)
        and prov["new_BPE_fractional_aligned_count"]==32
        and prov["S61_HUMAN_GOLD_NOT_IMPORTED_DURING_ANY_CAPTURE_STEP"],
        "real original43M GT 616J7 gold-free CPU source provenance error")
    require(
        fit61["stage"]=="NRA3_S61_OLD216_REAL43M_GT_SOURCE_TARGET2752_TRAIN_ONLY"
        and fit61["new_S61_paid2752_SHA256"]==sha(a.s61)
        and fit61["original_S26_SHA256"]==sha(a.s26)
        and fit61["old_S57_paid231_SHA256"]==sha(a.s57)
        and fit61["old_S59_paid2046_SHA256"]==sha(a.s59)
        and fit61["new_S61_paid_FP16_count"]==PAID==2752
        and fit61["S61_NEW32_raw_and_gold_never_imported_in_spent216_fitting"]
        and fit61["output_gate"]==0 and not fit61["candidate_accepted"]
        and not fit61["activation_allowed"]
        and fit59["new_S59_semantic_paid2046_sha256"]==sha(a.s59)
        and fit60["new_S60_paid2776_sha256"]==sha(a.s60)
        and fit60["original_S26_paid12384_sha256"]==sha(a.s26)
        and fit60["frozen_S57_paid231_sha256"]==sha(a.s57)
        and fit60["frozen_S59_paid2046_sha256"]==sha(a.s59),
        "S61 real native FP16 paid2752 or older ancestry SHA changed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source=load_file(str(a.native),device="cpu")
    require(set(source)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S61_HELD32"},
        "S61 original native only must not contain gold tensors")
    x=source["original_real43m_gt_word339_READONLY"].detach()
    valid=source["original_native_word_valid_mask"].detach()
    unsafe=source["native_BPE_causally_unsafe_word_mask"].detach()
    require(x.ndim==3 and x.shape[0]==32 and x.shape[2]==339
            and x.dtype==torch.float32
            and valid.shape==unsafe.shape==x.shape[:2]
            and valid.dtype==unsafe.dtype==torch.bool
            and source["source_row_indices_in_frozen_S61_HELD32"].tolist()
                ==list(range(32)),
            "S61 genuine Calibri43M GT native source/word339 ABI invalid")
    native_sha=tensor_sha(x,valid,unsafe)
    old26=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(a.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    old59=S59NativeEventSemantic.from_fp16(
        load_file(str(a.s59),device="cpu")["S59_NATIVE320_SEMANTIC2046_FP16"])
    old60=S60Joint3.from_fp16(
        load_file(str(a.s60),device="cpu")["S60_SOURCE_LOCAL_TRICLASS2776_FP16"])
    new61=S61NativePairNull.from_fp16(
        load_file(str(a.s61),device="cpu")["S61_REAL43M_GT_PAIR_NULL_POLAR2752_FP16"])
    olds=(old26,old57,old59,old60,new61)
    require(all(not z.requires_grad for m in olds for z in m.parameters())
            and all(m.output_gate==0 and not m.candidate_accepted and
                    not m.activation_allowed for m in olds),
            "model gate0 and pretraining frozen state required")
    hashes=[tensor_sha(*(z.detach() for z in m.parameters())) for m in olds]
    with torch.inference_mode():
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
        event=gold_blind_eventfirst_features(x,obs,raw_only_inputs())
        semantic=semantic_features(x,event)
        inp=source_local_scope(semantic,raw_only_inputs())
        old_logits=tuple(v.detach() for v in old57(event))
        s57=old57.decode(event)
        s59=old59.decode(semantic,old_logits)
        s60=old60.decode(inp,old_logits,old59)
        s61=new61.decode(inp,old_logits,old59)
        immutable=(native_sha==tensor_sha(x,valid,unsafe)
                   and all(tensor_sha(*(z.detach() for z in m.parameters()))==v
                           for m,v in zip(olds,hashes)))
    require(immutable and s61["output_gate"]==0
            and not s61["candidate_accepted"]
            and int(s61["directed_recipient"].sum())<=32
            and not bool((s61["directed_recipient"]&~event.valid_pair).any())
            and not bool((s61["directed_rejected"]&
                          ~s61["directed_recipient"]).any()),
            "S61 frozen first forward changed original GT or emitted unsafe edge")

    # *** DO NOT MOVE: first manual gold is imported after ALL S57..S61 forward ***
    from research.nra3_s20.s61_new32_separate_frozen_gold_DO_NOT_TRAIN import (
        held_gold32,gold_sha256,
    )
    gold=held_gold32()
    require(len(gold)==len(SOURCE_ROWS)==32,"S61 raw and manual GOLD32 count drift")
    truth=torch.zeros_like(s61["directed_recipient"])
    rejected=torch.zeros_like(truth)
    gold_event=torch.zeros(32,dtype=torch.long)
    ids=[]
    for row,(raw,g) in enumerate(zip(SOURCE_ROWS,gold)):
        require(raw[0]==g.row_id,"S61 annotated row identity drift")
        words=raw_surface_grammar(raw[2])
        require(len(words)==int(valid[row].sum()),
                "S61 original BPE word339 alignment after manual gold mismatch")
        def find(form):
            hits=[i for i,z in enumerate(words) if z.form==form]
            require(len(hits)==1,
                    "S61 precommitted gold manual word not unique/absent")
            return hits[0]
        source_idx=find(g.main_predicate)
        gold_event[row]=source_idx
        dest_idx=None
        if g.transferred_recipient is not None:
            dest_idx=find(g.transferred_recipient)
            truth[row,source_idx,dest_idx]=True
            if g.directed_transfer_polarity=="REJECTED":
                rejected[row,source_idx,dest_idx]=True
        ids.append({
            "id":g.row_id,"family":raw[1],
            "gold_event":g.main_predicate,
            "gold_recipient":g.transferred_recipient,
            "gold_rejected":g.directed_transfer_polarity,
            "gold_event_source_mask_safe":bool(event.candidate_event[row,source_idx]),
            "gold_recipient_pair_BPE_safe":bool(event.valid_pair[row,source_idx,dest_idx])
                                           if dest_idx is not None else None,
            "S59_main_pred":words[int(s59["chosen_event"][row])].form,
            "S60_main_pred":words[int(s60["chosen_event"][row])].form,
            "S61_main_pred":words[int(s61["chosen_event"][row])].form,
            "S61_selected_link":int(s61["directed_recipient"][row].sum()),
            "S61_selected_rejected":int(s61["directed_rejected"][row].sum()),
        })
    require(int(truth.sum())==17 and int(rejected.sum())==8,
            "S61 new manual gold must have 17 directed and 8 REJECTED")
    preds={
        "S57":s57,"S59_native2046":s59,
        "S60_local2776":s60,"S61_pair_null2752":s61,
    }
    metrics={}
    for name,pred in preds.items():
        metrics[name+"_RECIPIENT"]=score(pred["directed_recipient"],truth)
        metrics[name+"_REJECTED"]=score(pred["directed_rejected"],rejected)
        event_ix=(pred["chosen_outer_event_index"] if name=="S57"
                  else pred["chosen_event"])
        metrics[name+"_MAIN_PREDICATE"]={
            "correct":int((event_ix==gold_event).sum()),
            "total":32,
            "accuracy":float((event_ix==gold_event).float().mean()),
        }
    by_family={}
    for family in FAMILIES:
        take=torch.tensor([i for i,r in enumerate(SOURCE_ROWS) if r[1]==family])
        require(len(take)==8,"S61 precommitted 4x8 family size drift")
        by_family[family]={}
        for name,pred in preds.items():
            by_family[family][name+"_RECIPIENT"]=score(
                pred["directed_recipient"][take],truth[take])
            by_family[family][name+"_REJECTED"]=score(
                pred["directed_rejected"][take],rejected[take])

    # Diagnostic upper bound ONLY: teacher-forced TRUE source performed
    # after gold read, never used to tune or select candidate. Not a
    # deployable prediction, labeled ORACLE and never certified.
    with torch.inference_mode():
        oracle_score,oracle_polar,oracle_mask=new61.score_sources(
            inp,old_logits,old59,gold_event)
        oracle_dst=oracle_score.argmax(-1)
    oracle_links=torch.zeros_like(truth)
    oracle_rej=torch.zeros_like(rejected)
    W=truth.shape[-1]
    for i in range(32):
        dst=int(oracle_dst[i])
        if dst==W:continue
        src=int(gold_event[i])
        require(bool(oracle_mask[i,dst]),"oracle source pair unsafe")
        oracle_links[i,src,dst]=True
        oracle_rej[i,src,dst]=float(oracle_polar[i,dst])>0
    diag={
        "uses_TRUE_main_predicate_after_manual_GOLD_not_model_prediction":True,
        "not_a_valid_predictive_model_metric_or_candidate_selection":True,
        "S61_oracle_event_recipient":score(oracle_links,truth),
        "S61_oracle_event_rejected":score(oracle_rej,rejected),
    }
    report={
        "stage":STAGE,
        "original_12layer_Calibri43M_GT_true_CPU_616_J7_before_NEW32":True,
        "original_core_and_frozen_paid_SHA_unchanged":immutable,
        "original_616_J7_BASE_ONLY_not_integrated_S61_replay":True,
        "S61_held32_raw_commit":"02f8bb787a309b065c09e1f2f881bc2ef8e1f3d7",
        "S61_held32_raw_blob":"5e0a40c2cdcfb95f7eea9e91879e68211c3b0afe",
        "S61_held32_gold_commit":"8ad5d37069c2904a83af6a433c29bbb2d403da1e",
        "S61_held32_gold_blob":"6ac9e0b863b54e76f86c5ed85c9d2e937bd3e5f2",
        "S61_original_raw_sha256":raw_sha256(),
        "S61_separately_frozen_gold_sha256":gold_sha256(),
        "S61_trained_paid2752_sha256":sha(a.s61),
        "S61_original_native339_sha256":sha(a.native),
        "manual_gold_imported_ONLY_AFTER_original_gt_and_all_paid_forward":True,
        "S61_NEW32_after_this_one_FIRST_READ_PERMANENTLY_SPENT":True,
        "NOT_random_or_representative_population_EVAL":True,
        "all_true_main_predicates_eligible":sum(bool(x["gold_event_source_mask_safe"])
                                                 for x in ids),
        "all_gold_directed_pairs_BPE_safe":sum(x["gold_recipient_pair_BPE_safe"] is True
                                                for x in ids),
        "S61_no_target_abstentions":s61["no_target_abstentions"],
        "FIRST_READ_NEW32_comparisons":metrics,
        "FIRST_READ_NEW32_four_8case_families":by_family,
        "FIRST_READ_NEW32_row_diagnostics":ids,
        "ORACLE_source_after_gold_only_DIAGNOSTIC_NOT_DEPLOYABLE":diag,
        "no_new_HELD_training_or_threshold_tuning":True,
        "no_protected_S38_S47_S48_EVAL32_128":True,
        "integrated_S61_candidate_J7_bpp_CB59_NOT_CERTIFIED":True,
        "original_full_prefill_not_streaming_causal_certified":True,
        "LIVE_Render_Supabase_Jessica_APK_UNCHANGED":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    a.out.mkdir(parents=True,exist_ok=False)
    name=a.out/"NRA3_S61_TRUE43M_GT_FIRST_NEW32_PAIR_NULL_REPORT.json"
    name.write_text(json.dumps(report,ensure_ascii=False,indent=2,
                               sort_keys=True)+"\n",encoding="utf-8")
    print("S61_TRUE_ORIGINAL43M_GT_NEW32_FIRST_READ_COMPARISON",
          json.dumps(metrics,sort_keys=True),flush=True)
    print("S61_TRUE_ORIGINAL43M_GT_NEW32_ORACLE_DIAGNOSTIC_NOT_VALID_PREDICTION",
          json.dumps(diag,sort_keys=True),flush=True)
    print("S61_FIRST_READ_NEW32_SPENT_GATE0",
          json.dumps({"spent":True,"output_gate":0,
                      "original_616_J7_base_only":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
