"""S55 first structural-shift held24 evaluation of trained 36-FP16 real43M recipient residual.

Data isolation: S53 previously scored TRAIN24 is fitted IN A DIFFERENT
completed CI step and saved before this script can load S54 HELD data.
The genuine original Calibri43M+GT 12-layer CPU HELD24 prefill runs
after original 616 J7 replay, in a separate script importing ONLY
plain raw HELD texts. This script loads old S26+S54 fitted readout,
completes all HELD predictions and freezes them in memory BEFORE
importing HELD24 manual gold. No training, optimizer, threshold tuning
or model selection here. This is a precommitted structurally-shifted (not just lexical) researcher-authored first-read held, not representative independent real-world EVAL.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors.torch import load_file
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s19.s19f_real43m_train9_graph_targets import ROLE,STATUS
from research.nra3_s20.s26_predicted_event_inversion_rescue import EventGroundedInversionGraph
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    S52QuoteAwareFrozenPaidGraphBridge,tensor_sha,
)
from research.nra3_s20.s54_trainonly_real43m_recipient_residual import (
    S54DirectedRecipientResidual,pair_features_gold_blind,FEATURE_NAMES,
)
from research.nra3_s20.s55_frozen_structural_shift_held24_plain import (
    SOURCE_ROWS,text_only_inputs,text_sha256,
)

STAGE="NRA3_S55_STRUCTURAL_SHIFT_HELD24_REAL43M_GT_FROZEN_S54_PAID36_FIRST"
H_NATIVE="NRA3_S55_GENUINE43M_GT_STRUCTURAL24_WORD339_AND_UNSAFE.safetensors"
H_RAW="NRA3_S55_GENUINE43M_GT_STRUCTURAL24_OFFSET_PROVENANCE.json"
H_CAP="NRA3_S55_GENUINE43M_GT_STRUCTURAL24_NATIVE_J7_REPORT.json"
OLD_PAID="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
NEW_PAID="NRA3_S54_S53_TRAINONLY_36FP16_DIRECTED_RECIPIENT_REJECTED.safetensors"
TRAIN_REPORT="NRA3_S54_S53_TRAINONLY_NEW36_FP16_FIRST_TRAIN_REPORT.json"

def require(ok,why):
    if not bool(ok):raise ValueError("S54 first real43M+GT held diagnostic: "+why)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def count(p,g):
    tp=int((p&g).sum());fp=int((p&~g).sum());fn=int((~p&g).sum())
    den=2*tp+fp+fn
    return {"TP":tp,"FP":fp,"FN":fn,
            "F1":(2*tp/den if den else 0.),
            "precision":(tp/(tp+fp) if tp+fp else 0.),
            "recall":(tp/(tp+fn) if tp+fn else 0.)}

def sole_word(tokens,term):
    a=[i for i,t in enumerate(tokens) if t.form==term]
    require(len(a)==1,"precommitted held gold token must align with exact unique original word")
    return a[0]

def main():
    p=argparse.ArgumentParser()
    for field in ("held-native","held-provenance","held-capture",
                  "s26-paid","new-readout","train-manifest","out"):
        p.add_argument("--"+field,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"first precommitted HELD24 report already exists")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    h=a.held_native
    hprov=a.held_provenance
    hcap=a.held_capture
    require(all(z.is_file() for z in (
        h,hprov,hcap,a.s26_paid,a.new_readout,a.train_manifest))
        and h.name==H_NATIVE and hprov.name==H_RAW
        and hcap.name==H_CAP and a.s26_paid.name==OLD_PAID
        and a.new_readout.name==NEW_PAID and a.train_manifest.name==TRAIN_REPORT,
        "S54 expected exact previous physical true43M native and old/new trained weights")
    prov=json.loads(hprov.read_text())
    cap=json.loads(hcap.read_text())
    fit=json.loads(a.train_manifest.read_text())
    require(prov["stage"]=="NRA3_S55_GENUINE43M_GT_STRUCTURAL24_BPE_SOURCE_ONLY"
            and prov["frozen_S55_HELD24_raw_source_sha256"]==text_sha256()
            and prov["derived_REAL43M_WORD339_SHA256"]==sha(h)
            and prov["new_BPE_fractional_aligned_count"]==24
            and prov["original_616_J7_replayed_before_read_only_quotes"]
            and prov["S55_HUMAN_GOLD_NOT_IMPORTED_DURING_ANY_CAPTURE_STEP"]
            and cap["stage"]=="NRA3_S55_GENUINE43M_GT_STRUCTURAL24_NATIVE_CPU_FIRST_PASS"
            and cap["all_616_native_CPU_generated_token_digests_unchanged"]
            and cap["actual_genuine_43M_CPU_new_HELD24_prefills"]==24
            and cap["S55_FROZEN_HELD24_PLAINTEXT_SHA256"]==text_sha256()
            and cap["S55_HELD24_SOURCE_COMMIT"]=="943bdfe06dc4f5de04bf476210c05dac14ec236d"
            and cap["S55_HELD24_SOURCE_BLOB"]=="7d20bfb4aac8c768bc9ffceae65a8000b6926ab4"
            and cap["original_456_full_EOS"]==456
            and cap["original_new128_full_EOS"]==128
            and cap["original_new32_full_EOS"]==0
            and not cap["NRA3_candidate_accepted"]
            and cap["NRA3_output_gate"]==0,
            "S54 genuine 43M GT protected 616 J7 and independent raw held24 provenance drift")
    require(fit["stage"]=="NRA3_S54_TRAIN_ONLY_S53_PREVIOUSLY_SCORED24_GENUINE43M_GT_PAID36"
            and fit["physically_saved_36_FP16_readout_SHA256"]==sha(a.new_readout)
            and fit["newly_paid_trainable_FP16_params"]==36
            and fit["no_S54_HELD24_text_or_gold_imported_in_this_process"]
            and fit["output_gate"]==0
            and not fit["candidate_accepted"]
            and not fit["activation_allowed"],
            "S54 new 36 FP16 readout not trained solely on S53 before HELD read")
    tensors=load_file(str(h),device="cpu")
    require(set(tensors)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S55_HELD24"},
        "HELD native archive unexpectedly includes manual gold data")
    x=tensors["original_real43m_gt_word339_READONLY"].detach()
    valid=tensors["original_native_word_valid_mask"].detach()
    unsafe=tensors["native_BPE_causally_unsafe_word_mask"].detach()
    require(x.ndim==3 and x.shape[0]==24 and x.shape[-1]==339
            and valid.shape==unsafe.shape==x.shape[:2]
            and tensors["source_row_indices_in_frozen_S55_HELD24"].tolist()==list(range(24)),
            "HELD native row order or 43M feature structure drift")
    orig_sha=tensor_sha(x,valid,unsafe)
    old=EventGroundedInversionGraph.from_physical(a.s26_paid.read_bytes())
    param_sha=tensor_sha(*(z.detach() for z in old.parameters()))
    require(all(not z.requires_grad for z in old.parameters())
            and old.output_gate==0 and not old.candidate_accepted,
            "original 12,384 paid S26 was not frozen")
    paid=load_file(str(a.new_readout),device="cpu")
    require(set(paid)=={"S54_new36_FP16_readout"},"S54 research readout archive unexpected")
    new=S54DirectedRecipientResidual.from_physical_fp16(
        paid["S54_new36_FP16_readout"])
    require(new.output_gate==0 and not new.candidate_accepted
            and not new.activation_allowed
            and all(not z.requires_grad for z in new.parameters()),
            "S54 new trained 36 FP16 learner not frozen before HELD native inference")
    with torch.inference_mode():
        old_logits=old(x[...,:338],valid)
        watched=S52QuoteAwareFrozenPaidGraphBridge(old).observe(x,valid,unsafe)
        require(all(torch.equal(getattr(old_logits,k),
                                getattr(watched.original_paid_graph_logits,k))
                    for k in ("event_status","role","argument_polarity",
                              "coreference","event_relation")),
                "original S26 paid model changed under HELD S54 safety masks")
        feature=pair_features_gold_blind(x,watched)
        new_pred_rec,new_pred_rej=new.predict(
            feature,watched.eligible_directed_pair_mask)
        old_pred_role=(old_logits.role.argmax(-1)==ROLE.index("RECIPIENT"))
        old_pred_rej=old_pred_role & (old_logits.argument_polarity.argmax(-1)==2)
        base_pair=valid.unsqueeze(2)&valid.unsqueeze(1)
        base_pair&=~torch.eye(valid.shape[1],dtype=torch.bool).unsqueeze(0)
        old_pred_role=(old_pred_role&base_pair).clone()
        old_pred_rej=(old_pred_rej&base_pair).clone()
        old_safe_rec=(old_pred_role&watched.eligible_directed_pair_mask).clone()
        old_safe_rej=(old_pred_rej&watched.eligible_directed_pair_mask).clone()
        # All gold-blind predictions now detached and frozen BEFORE gold import.
        for z in (new_pred_rec,new_pred_rej,old_pred_role,old_pred_rej):
            require(not z.requires_grad and z.dtype==torch.bool,
                    "S54 first HELD predictions not frozen before manual labels")
        freeze_digests={
          "native_before_after":orig_sha==tensor_sha(x,valid,unsafe),
          "S26_original_12384_before_after":param_sha==tensor_sha(
              *(z.detach() for z in old.parameters())),
          "new_FP16_paid_before_after":torch.equal(
              new.physical_fp16(),paid["S54_new36_FP16_readout"]),
          "gate_zero":new.output_gate==old.output_gate==0,
        }
        require(all(freeze_digests.values()),
                "S54 HELD scoring accidentally updated original 43M/S26 or new readout")
    # ONLY NOW import *independently precommitted* HELD manual gold24.
    # This is the single first read and NEVER a hyperparameter selection
    # opportunity. No model forwards or optimizers AFTER this point.
    from research.nra3_s20.s55_frozen_structural_shift_held24_manual_gold import (
        held_gold24,gold_sha256,
    )
    gold=held_gold24()
    require(len(gold)==24 and len(SOURCE_ROWS)==24
            and len(text_only_inputs())==24 and len(gold_sha256())==64,
            "S54 separately frozen first HELD24 manual source mismatch")
    desired_rec=torch.zeros_like(new_pred_rec)
    desired_rej=torch.zeros_like(new_pred_rej)
    desired_event_neg=torch.zeros_like(valid)
    pred_gold_nodes=torch.zeros_like(valid)
    result_rows=[]
    for i,(row,g) in enumerate(zip(SOURCE_ROWS,gold)):
        require(row[0]==g.row_id,"first HELD row ID drift")
        words=raw_surface_grammar(row[2])
        require(len(words)==int(valid[i].sum()),
                "S54 first HELD original GT native token alignment drift")
        event_index=sole_word(words,g.main_predicate)
        pred_gold_nodes[i,event_index]=True
        if g.main_factuality=="NEGATED":desired_event_neg[i,event_index]=True
        rec_index=sole_word(words,g.transferred_recipient) if g.transferred_recipient is not None else None
        if rec_index is not None:
            desired_rec[i,event_index,rec_index]=True
            if g.transfer_polarity=="REJECTED":
                desired_rej[i,event_index,rec_index]=True
        result_rows.append({
            "id":g.row_id,"category":row[1],
            "precommitted_gold_predicate":g.main_predicate,
            "precommitted_gold_recipient":g.transferred_recipient,
            "precommitted_gold_REJECTED":g.transfer_polarity=="REJECTED",
            "original_S26_recipient_on_gold_pair":(
               bool(old_pred_role[i,event_index,rec_index]) if rec_index is not None else None),
            "new_S54_recipient_on_gold_pair":(
               bool(new_pred_rec[i,event_index,rec_index]) if rec_index is not None else None),
            "original_S26_REJECTED_on_gold_pair":(
               bool(old_pred_rej[i,event_index,rec_index]) if rec_index is not None else None),
            "new_S54_REJECTED_on_gold_pair":(
               bool(new_pred_rej[i,event_index,rec_index]) if rec_index is not None else None),
            "safety_eligible_true_recipient_pair":(
               bool(watched.eligible_directed_pair_mask[i,event_index,rec_index])
               if rec_index is not None else None),
            "precommitted_HELD_manual_gold_loaded_ONLY_AFTER_all_inference":True,
        })
    require(int(desired_rec.sum())==12 and int(desired_rej.sum())==6
            and int(desired_event_neg.sum())==12,
            "HELD24 author gold true directed recipient/REJECTED totals changed")
    stats={
        "old_S26_RECIPIENT_unmasked":count(old_pred_role,desired_rec),
        "old_S26_RECIPIENT_S52_safe":count(old_safe_rec,desired_rec),
        "new_S54_RECIPIENT_S52_safe":count(new_pred_rec,desired_rec),
        "old_S26_REJECTED_unmasked":count(old_pred_rej,desired_rej),
        "old_S26_REJECTED_S52_safe":count(old_safe_rej,desired_rej),
        "new_S54_REJECTED_S52_safe":count(new_pred_rej,desired_rej),
        "original_S26_NEGATED_event":count(
            (old_logits.event_status.argmax(-1)==STATUS.index("NEGATED"))&valid,
            desired_event_neg),
    }
    groups={}
    for category in ("negated_main_complex","affirmative_with_other_negation",
                     "negated_dative_nonrecipient","reported_quote_no_assertion"):
        ids=torch.tensor([i for i,r in enumerate(SOURCE_ROWS) if r[1]==category])
        require(len(ids)==6,"S54 predeclared HELD group support changed")
        groups[category]={
            "baseline_S26_REJECTED":count(old_safe_rej.index_select(0,ids),
                                          desired_rej.index_select(0,ids)),
            "new_S54_REJECTED":count(new_pred_rej.index_select(0,ids),
                                     desired_rej.index_select(0,ids)),
            "baseline_S26_RECIPIENT":count(old_safe_rec.index_select(0,ids),
                                           desired_rec.index_select(0,ids)),
            "new_S54_RECIPIENT":count(new_pred_rec.index_select(0,ids),
                                      desired_rec.index_select(0,ids)),
            "true_S54_gold_REJECTED":int(desired_rej.index_select(0,ids).sum()),
            "true_S54_gold_RECIPIENT":int(desired_rec.index_select(0,ids).sum()),
        }
    report={
      "stage":STAGE,
      "fresh_HELD_source_plain_sha256":text_sha256(),
      "fresh_HELD_source_gold_sha256":gold_sha256(),
      "frozen_HELD_plain_source_commit":"943bdfe06dc4f5de04bf476210c05dac14ec236d",
      "frozen_HELD_plain_source_blob":"7d20bfb4aac8c768bc9ffceae65a8000b6926ab4",
      "frozen_HELD_gold_source_commit":"0b5a667003cbf921b08755d15a285ecb6000874a",
      "frozen_HELD_gold_source_blob":"fc9c2f96e491fd8d8b99f6df53bd8f8573bd2ea0",
      "S53_TRAIN_only_36FP16_readout_SHA256":sha(a.new_readout),
      "old_S26_physically_paid12384_archive_SHA256":sha(a.s26_paid),
      "new_original_true43M_GT_native_first_HELD24_source_SHA256":sha(h),
      "original_Calibri43M_GreenTunnel_12layers_protected616J7_replayed_prior_capture":True,
      "original_frozen_43M_S26_36FP16_inference_all_completed_before_new_held_gold_import":True,
      "physical_model_and_GT_no_train_no_optimizer_during_HELD_forward":True,
      "new_S54_36FP16_readout_frozen_during_HELD_first_forward":True,
      "new_S54_first_HELD24_model_and_NATIVE_same_SHA_before_after":freeze_digests,
      "all24_true_directed_gold_RECIPIENT":int(desired_rec.sum()),
      "all24_true_directed_gold_REJECTED":int(desired_rej.sum()),
      "all24_true_main_event_NEGATED":int(desired_event_neg.sum()),
      "original_S26_vs_new_S54_actual_true43M_GT_first_HELD_counts_NOT_independent_realworld_EVAL":stats,
      "first_HELD_metrics_by_predeclared_family":groups,
      "first_HELD_all24_manual_gold_pair_comparison":result_rows,
      "HELD_is_author_constructed_STRUCTURAL_shift_not_random_independent_realworld":True,
      "HELD_not_used_to_train_select_fit_or_change_thresholds":True,
      "S53_previously_scored_TRAIN24_only_in_new36_gradients":True,
      "S38_S47_S48_HELD26_EVAL32_EVAL128_not_accessed":True,
      "original_S26_bidirectional_context_NOT_streaming_causal":True,
      "616J7_active_candidate_bpp_CB59_not_verified":True,
      "LIVE_Render_Supabase_Jessica_APK_not_modified":True,
      "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"NRA3_S55_REAL43M_GT_STRUCTURAL24_FROZEN_S54_RECIPIENT_REJECTED_FIRST_HELD.json").write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf-8")
    print("S55_STRUCTURAL_SHIFT_HELD24_FROZEN_S54_RECIPIENT_REJECTED",
          json.dumps(stats,sort_keys=True),flush=True)
    print("S55_STRUCTURAL_SHIFT_BY_PREDECLARED_FAMILY",
          json.dumps(groups,sort_keys=True),flush=True)
    print("S55_FROZEN_S54_PAID36_NO_STRUCTURAL_HELD_TRAIN_NO_ACTIVE_MODEL",flush=True)

if __name__=="__main__":
    main()
