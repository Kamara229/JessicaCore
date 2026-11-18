"""S60 FIRST NEW32 author-created contrast: genuine 12-layer Calibri43M GT.

First replay protected original 616 J7, capture native post-GT word339
and produce S57/S59/S60 predictions with old paid/NEW2776 FP16 frozen.
Only AFTER all forwards, import precommitted S60 manual human gold.
No tuning, no new32 training, no original GT/Calibri/Jessica modification.
NEW32 turns SPENT after this one read; NOT population EVAL.
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
from research.nra3_s20.s60_new32_precommitted_raw_only import (
    SOURCE_ROWS,FAMILIES,raw_only_inputs,raw_sha256,
)
STAGE="NRA3_S60_REAL43M_GT_NATIVE3WAY2776_FIRST_NEW32_AUTHOR_CONTRAST"
NATIVE="NRA3_S60_GENUINE43M_GT_EVENTFIRST32_WORD339_AND_UNSAFE.safetensors"
PROV="NRA3_S60_GENUINE43M_GT_EVENTFIRST32_OFFSET_PROVENANCE.json"
CAPTURE="NRA3_S60_GENUINE43M_GT_EVENTFIRST32_NATIVE_J7_REPORT.json"
OLD26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
OLD57="NRA3_S57_EVENTFIRST_SPENT96_ONLY_231FP16_UNACCEPTED.safetensors"
OLD59="NRA3_S59_REAL43M_GT_OLD152_NATIVE320_SEMANTIC2046_FP16.safetensors"
MAN59="NRA3_S59_OLD152_TRAIN_NATIVE2046_FP16_MANIFEST.json"
NEW60="NRA3_S60_ORIGINAL43M_GT_OLD184_TRICLASS2776_FP16.safetensors"
MAN60="NRA3_S60_OLD184_TRAIN_TRICLASS2776_MANIFEST.json"

def require(ok,why):
    if not bool(ok):raise ValueError("S60 physical first NEW32: "+why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(y,g):
    tp=int((y&g).sum());fp=int((y&~g).sum());fn=int((~y&g).sum())
    d=2*tp+fp+fn
    return {"TP":tp,"FP":fp,"FN":fn,
            "precision":tp/(tp+fp) if tp+fp else 0.,
            "recall":tp/(tp+fn) if tp+fn else 0.,
            "F1":2*tp/d if d else 0.}

def main():
    a=argparse.ArgumentParser()
    for z in ("native","prov","capture","s26","s57","s59","fit59",
              "s60","fit60","out"):
        a.add_argument("--"+z,type=Path,required=True)
    q=a.parse_args()
    require(not q.out.exists(),"cannot overwrite once-scored NEW32 HELD")
    files=(q.native,q.prov,q.capture,q.s26,q.s57,q.s59,q.fit59,q.s60,q.fit60)
    names=(NATIVE,PROV,CAPTURE,OLD26,OLD57,OLD59,MAN59,NEW60,MAN60)
    require(all(p.is_file() and p.name==n for p,n in zip(files,names)),
            "genuine archived original checkpoint naming must be pinned")
    capture=json.loads(q.capture.read_text())
    provenance=json.loads(q.prov.read_text())
    train59=json.loads(q.fit59.read_text())
    train60=json.loads(q.fit60.read_text())
    require(capture["stage"]==
            "NRA3_S60_GENUINE43M_GT_EVENTFIRST32_NATIVE_CPU_FIRST_PASS"
            and capture["all_616_native_CPU_generated_token_digests_unchanged"]
            and capture["original_456_full_EOS"]==456
            and capture["original_new128_full_EOS"]==128
            and capture["actual_43M_CPU_12_transformer_layers"]
            and capture["GT_residuals_before_readport_in_same_active_S93_scope"]
            and capture["actual_genuine_43M_CPU_new_HELD32_prefills"]==32
            and capture["S60_FROZEN_HELD32_PLAINTEXT_SHA256"]==raw_sha256()
            and capture["S60_HELD32_SOURCE_COMMIT"]==
                "b1b80206a4b6e352b4262a909a0129769f4291e0"
            and capture["S60_HELD32_SOURCE_BLOB"]==
                "ef257f57a2a79a8a042ce48766f271648ee20b70"
            and capture["S60_HELD_manual_gold_NOT_IMPORTED_into_model_capture"]
            and capture["NRA3_output_gate"]==0 and not capture["activation_allowed"]
            and provenance["stage"]==
                "NRA3_S60_GENUINE43M_GT_EVENTFIRST32_BPE_SOURCE_ONLY"
            and provenance["frozen_S60_HELD32_raw_source_sha256"]==raw_sha256()
            and provenance["new_BPE_fractional_aligned_count"]==32
            and provenance["S60_HUMAN_GOLD_NOT_IMPORTED_DURING_ANY_CAPTURE_STEP"]
            and provenance["derived_REAL43M_WORD339_SHA256"]==sha(q.native)
            and provenance["original_616_J7_replayed_before_read_only_quotes"],
            "S60 actual original43M GT CPU J7 and label-free native capture failed")
    require(train60["stage"]==
              "NRA3_S60_FROZEN_ORIGINAL43M_GT_SPENT184_NATIVE_3WAY2776_FP16"
            and train60["new_S60_paid2776_sha256"]==sha(q.s60)
            and train60["frozen_S59_paid2046_sha256"]==sha(q.s59)
            and train60["frozen_S57_paid231_sha256"]==sha(q.s57)
            and train60["original_S26_paid12384_sha256"]==sha(q.s26)
            and train60["new_S60_FP16_parameters"]==2776
            and train60["new_S60_RAW32_and_GOLD32_never_imported_into_OLD184_TRAIN"]
            and train60["output_gate"]==0
            and not train60["candidate_accepted"]
            and train59["new_S59_semantic_paid2046_sha256"]==sha(q.s59),
            "S60 fixed true native FP16 must predate untouched new32 eval")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source=load_file(str(q.native),device="cpu")
    require(set(source)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S60_HELD32"},
        "S60 native capture unexpectedly contains gold source or label")
    x=source["original_real43m_gt_word339_READONLY"].detach()
    valid=source["original_native_word_valid_mask"].detach()
    unsafe=source["native_BPE_causally_unsafe_word_mask"].detach()
    require(x.shape[0]==32 and x.shape[-1]==339
            and valid.shape==unsafe.shape==x.shape[:2]
            and valid.dtype==unsafe.dtype==torch.bool
            and x.dtype==torch.float32
            and source["source_row_indices_in_frozen_S60_HELD32"].tolist()
                ==list(range(32)),"S60 original BPE native shape drift")
    native_hash=tensor_sha(x,valid,unsafe)
    old26=EventGroundedInversionGraph.from_physical(q.s26.read_bytes())
    old57=S57EventFirstRecipient.from_physical_fp16(
        load_file(str(q.s57),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    old59=S59NativeEventSemantic.from_fp16(
        load_file(str(q.s59),device="cpu")["S59_NATIVE320_SEMANTIC2046_FP16"])
    new60=S60Joint3.from_fp16(
        load_file(str(q.s60),device="cpu")["S60_SOURCE_LOCAL_TRICLASS2776_FP16"])
    models=(old26,old57,old59,new60)
    require(all(not v.requires_grad for m in models for v in m.parameters())
            and all(m.output_gate==0 and not m.candidate_accepted for m in models),
            "research-only old and new models must all be frozen gated")
    old_paid_hash=[tensor_sha(*(v.detach() for v in m.parameters())) for m in models]
    with torch.inference_mode():
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
        ev=gold_blind_eventfirst_features(x,obs,raw_only_inputs())
        semantic=semantic_features(x,ev)
        inp=source_local_scope(semantic,raw_only_inputs())
        logits57=tuple(t.detach() for t in old57(ev))
        pred57=old57.decode(ev)
        pred59=old59.decode(semantic,logits57)
        pred60=new60.decode(inp,logits57,old59)
        source_unmodified=(native_hash==tensor_sha(x,valid,unsafe))
        weights_unmodified=all(
            tensor_sha(*(v.detach() for v in m.parameters()))==s
            for m,s in zip(models,old_paid_hash))
    require(source_unmodified and weights_unmodified
            and pred60["output_gate"]==0 and not pred60["candidate_accepted"]
            and int(pred60["directed_recipient"].sum())<=32
            and not bool((pred60["directed_recipient"]&~ev.valid_pair).any())
            and not bool((pred60["directed_rejected"]&
                          ~pred60["directed_recipient"]).any()),
            "first read changed original GT/old paid or generated unsafe pair")
    # First gold read only after S57/S59/S60 native forward has completed.
    from research.nra3_s20.s60_new32_separate_precommitted_gold_NEVER_TRAIN import (
        held_gold32,gold_sha256,
    )
    gold=held_gold32()
    require(len(gold)==len(SOURCE_ROWS)==32,"new32 manual GOLD count/order drift")
    ground=torch.zeros_like(pred60["directed_recipient"])
    ground_rej=torch.zeros_like(ground)
    gold_event=torch.zeros(32,dtype=torch.long)
    families={}
    rows=[]
    for row,(raw,ref) in enumerate(zip(SOURCE_ROWS,gold)):
        require(raw[0]==ref.row_id,"new precommitted SHA gold changed row identity")
        words=raw_surface_grammar(raw[2])
        require(len(words)==int(valid[row].sum()),"S60 new original BPE source mismatch")
        def idx(form):
            matches=[j for j,tok in enumerate(words) if tok.form==form]
            require(len(matches)==1,"S60 gold manual raw predicate not unique")
            return matches[0]
        src=idx(ref.main_predicate);gold_event[row]=src
        if ref.transferred_recipient is not None:
            dst=idx(ref.transferred_recipient)
            ground[row,src,dst]=True
            if ref.directed_transfer_polarity=="REJECTED":
                ground_rej[row,src,dst]=True
        rows.append({
            "id":raw[0],"family":raw[1],
            "gold_main_predicate":ref.main_predicate,
            "gold_recipient":ref.transferred_recipient,
            "gold_polarity":ref.directed_transfer_polarity,
            "gold_event_eligible":bool(ev.candidate_event[row,src]),
            "gold_pair_safe":bool(ev.valid_pair[row,src,dst])
                             if ref.transferred_recipient is not None else None,
            "S59_predicted_main":words[int(pred59["chosen_event"][row])].form,
            "S60_predicted_main":words[int(pred60["chosen_event"][row])].form,
            "S60_predicted_class":int(pred60["chosen_event_class"][row]),
            "S60_predicted_link_count":int(pred60["directed_recipient"][row].sum()),
        })
    require(int(ground.sum())==16 and int(ground_rej.sum())==8,
            "new S60 manually precommitted 16 recipient/8 rejected cases drift")
    preds={"S57":pred57,"S59_native2046":pred59,"S60_local2776":pred60}
    totals={}
    for label,pred in preds.items():
        totals[label+"_RECIPIENT"]=metric(pred["directed_recipient"],ground)
        totals[label+"_REJECTED"]=metric(pred["directed_rejected"],ground_rej)
        chosen=(pred["chosen_outer_event_index"] if label=="S57"
                else pred["chosen_event"])
        totals[label+"_MAIN_PREDICATE"]={
            "correct":int((chosen==gold_event).sum()),"total":32,
            "accuracy":float((chosen==gold_event).float().mean())}
    for family in FAMILIES:
        ix=torch.tensor([k for k,r in enumerate(SOURCE_ROWS) if r[1]==family])
        require(len(ix)==8,"S60 four frozen 8-case families source drift")
        families[family]={}
        for key,p in preds.items():
            families[family][key+"_RECIPIENT"]=metric(
                p["directed_recipient"][ix],ground[ix])
            families[family][key+"_REJECTED"]=metric(
                p["directed_rejected"][ix],ground_rej[ix])
    report={
        "stage":STAGE,
        "source_S60_RAW_commit":"b1b80206a4b6e352b4262a909a0129769f4291e0",
        "source_S60_RAW_blob":"ef257f57a2a79a8a042ce48766f271648ee20b70",
        "source_S60_GOLD_commit":"322899c29bcb5695533ef795ad4213ad9ed8111e",
        "source_S60_GOLD_blob":"ea09f8ce3dea3d0c0a600d2c1b3e2aed5bf38c76",
        "S60_first_RAW32_sha256":raw_sha256(),
        "S60_first_manual_GOLD_sha256":gold_sha256(),
        "frozen_S60_paid2776_sha256":sha(q.s60),
        "genuine_original43M_GT_native339_sha256":sha(q.native),
        "all_616_original_J7_reproduced_before_NEW32_CPU_capture":True,
        "original616_J7_frozen_backbone_ONLY_not_ACTIVE_S60":True,
        "original_GT_and_all_paid_weights_unchanged":True,
        "human_manual_GOLD_imported_ONLY_AFTER_ALL_model_first_forward":True,
        "new_S60_HELD32_now_PERMANENTLY_SPENT":True,
        "not_random_population_generalization_eval":True,
        "gold_event_BPE_eligible_count":sum(z["gold_event_eligible"] for z in rows),
        "gold_directed_safe_BPE_count":sum(z["gold_pair_safe"] is True for z in rows),
        "S60_abstain_no_transfer":pred60["no_transfer_abstain"],
        "S60_abstain_no_safe_recipient":pred60["no_safe_target_abstain"],
        "first_NEW32_metrics":totals,"first_NEW32_family_metrics":families,
        "first_NEW32_row_diagnostics":rows,
        "never_tune_against_NEW32_after_first_manual_read":True,
        "active_S60_J7_bpp_CB59_not_certified":True,
        "full_prefill_not_online_autoregressive_causality_certified":True,
        "no_protected_HELD_S38_S47_S48_or_EVAL32_128":True,
        "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    q.out.mkdir(parents=True,exist_ok=False)
    (q.out/"NRA3_S60_TRUE43M_GT_FIRST_NEW32_MANUAL_REPORT.json").write_text(
       json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("S60_TRUE43M_GT_NEW32_FIRST_MANUAL_COMPARISON",
          json.dumps(totals,sort_keys=True),flush=True)
    print("S60_FIRST_NEW32_NOW_SPENT_RESEARCH_ONLY",
          json.dumps({"classes":pred60["chosen_event_class"].bincount(minlength=3).tolist(),
                      "output_gate":0,"j7_original_only":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
