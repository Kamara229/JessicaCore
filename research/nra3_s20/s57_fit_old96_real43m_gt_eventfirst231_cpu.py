"""S57 physical TRAIN ONLY on four already inspected old source 24x4 native sets.

The NEW S57 HELD source and gold modules are never imported here.
All original 43M+GT native data is genuine frozen post-layer8 CPU from
S53,S54,S55,S56, each after previously reproduced original 616 J7.
Old paid S26 12,384 FP16 is read-only. Four-head event-first separate
231 FP16 readout is the ONLY trainable model. No LIVE, no activation.
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
    EVENT_DIM,PAIR_DIM,N_PAID,
    gold_blind_eventfirst_features,train_fixed_on_previously_inspected_96_only,
)

STAGE="NRA3_S57_OLD_SPENT_S53_S54_S55_S56_TRAIN96_REAL43M_GT_EVENTFIRST_231FP16"
ARCHIVES=(
("S53","NRA3_S53_GENUINE43M_GT_NEGATION24_WORD339_AND_UNSAFE.safetensors",
 "source_row_indices_in_frozen_S53_NEW24"),
("S54","NRA3_S54_GENUINE43M_GT_HELD24_WORD339_AND_UNSAFE.safetensors",
 "source_row_indices_in_frozen_S54_HELD24"),
("S55","NRA3_S55_GENUINE43M_GT_STRUCTURAL24_WORD339_AND_UNSAFE.safetensors",
 "source_row_indices_in_frozen_S55_HELD24"),
("S56","NRA3_S56_GENUINE43M_GT_SCOPE24_WORD339_AND_UNSAFE.safetensors",
 "source_row_indices_in_frozen_S56_HELD24"),
)
OLD_S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"


def require(ok,why):
    if not bool(ok):raise ValueError("S57 old inspected TRAIN96 genuine 43M GT: "+why)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def word_idx(words,term):
    matches=[i for i,t in enumerate(words) if t.form==term]
    require(len(matches)==1,"S57 old human TRAIN word not present as exactly one original token")
    return matches[0]


def main():
    p=argparse.ArgumentParser()
    for name in ("s53","s54","s55","s56","s26","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"S57 old TRAIN physical checkpoint must not be overwritten")
    src=(a.s53,a.s54,a.s55,a.s56)
    require(a.s26.name==OLD_S26 and a.s26.is_file()
            and all(f.name==archive and f.is_file()
                    for (_,archive,_),f in zip(ARCHIVES,src)),
            "only exact old physically frozen real43M GT S53..56 source archives allowed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    # These old human annotations were ALREADY inspected/reported as EVAL.
    # They are now explicitly OLD TRAIN. S57 brand-new HELD24 not imported.
    from research.nra3_s20.s53_precommitted_negation_quote24_plain import SOURCE_ROWS as ROW53
    from research.nra3_s20.s53_separate_manual_gold_negation24 import gold24 as GOLD53
    from research.nra3_s20.s54_frozen_new_held24_plain_source import SOURCE_ROWS as ROW54
    from research.nra3_s20.s54_frozen_new_held24_manual_gold import held_gold24 as GOLD54
    from research.nra3_s20.s55_frozen_structural_shift_held24_plain import SOURCE_ROWS as ROW55
    from research.nra3_s20.s55_frozen_structural_shift_held24_manual_gold import held_gold24 as GOLD55
    from research.nra3_s20.s56_frozen_new_scope24_plain import SOURCE_ROWS as ROW56
    from research.nra3_s20.s56_frozen_new_scope24_gold import held_gold24 as GOLD56
    rows=(ROW53,ROW54,ROW55,ROW56)
    golds=(GOLD53(),GOLD54(),GOLD55(),GOLD56())
    require(all(len(r)==len(g)==24 for r,g in zip(rows,golds)),
            "S57 exactly four spent manually scored groups of 24 required")
    old=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    require(old.output_gate==old.parent.output_gate==0
            and all(not p.requires_grad for p in old.parameters()),
            "old S26 source 12,384 FP16 must be frozen")
    old_sha=tensor_sha(*(p.detach() for p in old.parameters()))
    trained=[]
    previous_native_shas=[]
    old_valid_word_counts=[]
    for (tag,_,index_key),file,raw,refs in zip(ARCHIVES,src,rows,golds):
        arch=load_file(str(file),device="cpu")
        require(set(arch)=={
             "original_real43m_gt_word339_READONLY",
             "original_native_word_valid_mask",
             "native_BPE_causally_unsafe_word_mask",index_key},
             "original43M source cannot contain any old human gold tensor")
        x=arch["original_real43m_gt_word339_READONLY"].detach().clone()
        valid=arch["original_native_word_valid_mask"].detach().clone()
        unsafe=arch["native_BPE_causally_unsafe_word_mask"].detach().clone()
        require(x.shape[0]==24 and x.shape[-1]==339
                and valid.shape==unsafe.shape==x.shape[:2]
                and arch[index_key].tolist()==list(range(24))
                and valid.dtype==unsafe.dtype==torch.bool,
                "original43M+GT native shape or label-free source alignment drift")
        native_sha=tensor_sha(x,valid,unsafe)
        with torch.inference_mode():
            old_logits=old(x[...,:338],valid)
            obs=S52QuoteAwareFrozenPaidGraphBridge(old).observe(x,valid,unsafe)
            require(all(torch.equal(getattr(old_logits,k),
                                    getattr(obs.original_paid_graph_logits,k))
                        for k in ("event_status","role","argument_polarity",
                                  "coreference","event_relation")),
                    "S57 old frozen 12,384 FP16 S26 five-head logits changed")
            representations=gold_blind_eventfirst_features(
                x,obs,tuple(r[2] for r in raw))
        # PyTorch S54 physical bug guarded: clone OUTSIDE inference_mode
        from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import EventFirstInputs
        representations=EventFirstInputs(
            representations.event.detach().clone(),
            representations.pair.detach().clone(),
            representations.candidate_event.detach().clone(),
            representations.valid_pair.detach().clone(),
        )
        require(not representations.event.is_inference()
                and not representations.pair.is_inference(),
                "S57 training feature born under inference_mode cannot autograd")
        m=representations.valid_pair
        gold_event=torch.zeros(24,dtype=torch.long)
        truth=torch.zeros_like(m)
        rejected=torch.zeros_like(m)
        has=torch.zeros(24,dtype=torch.bool)
        for i,(row,g) in enumerate(zip(raw,refs)):
            require(row[0]==g.row_id,"S57 spent original 43M gold source order mismatch")
            words=raw_surface_grammar(row[2])
            require(len(words)==int(valid[i].sum()),"S57 old source word boundary mismatch")
            predicate=g.predicate if tag in ("S53","S54") else g.main_predicate
            receiver=(g.recipient if tag in ("S53","S54") else
                      g.transferred_recipient)
            pol=(g.recipient_polarity if tag in ("S53","S54") else
                 g.transfer_polarity if tag=="S55" else
                 g.directed_transfer_polarity)
            event=word_idx(words,predicate)
            require(bool(representations.candidate_event[i,event]),
                    "S57 old gold main predicate excluded by hard quote/BPE/punctuation event mask")
            gold_event[i]=event
            if receiver is not None:
                dst=word_idx(words,receiver)
                require(bool(m[i,event,dst]),
                        "S57 old directed gold pair excluded from original native safety gate")
                truth[i,event,dst]=True
                has[i]=True
                if pol=="REJECTED":rejected[i,event,dst]=True
        require(tensor_sha(x,valid,unsafe)==native_sha,
                "S57 frozen original true43M/GT source changed during old TRAIN derivation")
        trained.append((representations,gold_event,truth,rejected,has))
        previous_native_shas.append(sha(file))
        old_valid_word_counts.append(int(valid.sum()))
    model,physical,training=train_fixed_on_previously_inspected_96_only(trained)
    require(physical.dtype==torch.float16 and physical.shape==(N_PAID,)
            and N_PAID==231
            and tensor_sha(*(p.detach() for p in old.parameters()))==old_sha
            and model.output_gate==0
            and not model.candidate_accepted and not model.activation_allowed
            and all(not p.requires_grad for p in model.parameters()),
            "S57 new event-first FP16 or old 43M/S26 changed")
    a.out.mkdir(parents=True,exist_ok=False)
    path=a.out/"NRA3_S57_EVENTFIRST_SPENT96_ONLY_231FP16_UNACCEPTED.safetensors"
    save_file({"S57_EVENTFIRST_PAID231_FP16":physical},str(path))
    require(torch.equal(
        load_file(str(path),device="cpu")["S57_EVENTFIRST_PAID231_FP16"],
        physical),"S57 old spent96 train-only FP16 physical bytes not reloadable")
    report={
      "stage":STAGE,
      "previously_separately_inspected_real43M_GT_source_CPU_runs":[
            44274389010,44279445613,44282672013,44308286512],
      "old_frozen_physically_paid_S26_run":44196720655,
      "old_training_4x24_real43M_GT_native_archive_sha256":previous_native_shas,
      "old_original_GT_native_word_counts":old_valid_word_counts,
      "old_S26_12384_FP16_physical_archive_SHA256":sha(a.s26),
      "new_event_first_FP16_model_parameters":N_PAID,
      "new_event_feature_dimension":EVENT_DIM,
      "old_pair_feature_dimension":PAIR_DIM,
      "new_event_first_FP16_weights_filename":path.name,
      "new_event_first_FP16_weights_SHA256":sha(path),
      "old_spent_TRAIN96_fixed_optimization_only":training,
      "old_previously_inspected_manual_gold_only_in_old_train_supervision":True,
      "new_S57_first_HELD24_raw_and_manual_gold_NEVER_IMPORTED":True,
      "old_original_true43M_GT_native_vectors_unchanged":True,
      "old_S26_12384_loaded_frozen_parameters_unchanged":True,
      "no_original_GreenTunnel_or_transformer_training":True,
      "S57_causal_streaming_decoder_NOT_CERTIFIED":True,
      "new_active_model_616_J7_bpp_CB59_not_evaluated":True,
      "no_protected_S38_S47_S48_HELD26_or_independent_EVAL32_128":True,
      "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    manifest=a.out/"NRA3_S57_EVENTFIRST_SPENT96_231FP16_TRAIN_REPORT.json"
    manifest.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                        encoding="utf-8")
    print("S57_EVENT_FIRST_OLD_REAL43M_GT_SPENT96_TRAINONLY",
          json.dumps({"old_true43m_word_counts":old_valid_word_counts,
                      "old_RECIPIENT":training["old_true_directed_recipient_edges"],
                      "old_REJECTED":training["old_true_directed_rejected_edges"],
                      "new_params_FP16":N_PAID,
                      "new_sha":sha(path),
                      "new_S57_HELD_raw_and_gold_unseen":True,
                      "gate":0},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
