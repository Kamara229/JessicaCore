"""S57 physical train-only event-first structured decoder on spent old96 (S53-56).

All four sources were previously inspected and are now TRAIN ONLY.
Loads physically captured genuine original Calibri43M+GreenTunnel
post-layer8 CPU native339 from S53,54,55,56 and frozen old S26 paid12,384
and S56 paid108. No original model/GT/S26/S54/S56 weights update.
Fresh S57 raw/gold HELD24 NOT IMPORTED in this process; no new inference
on S57 held until after saved FP16 new model is immutable.

Gold event predicate and target only used in old TRAIN96 loss, never
in feature creation nor inference. Gate0 candidate not accepted.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s26_predicted_event_inversion_rescue import (
    EventGroundedInversionGraph,
)
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    S52QuoteAwareFrozenPaidGraphBridge,tensor_sha,
)
from research.nra3_s20.s56_grammar_clause_scope_semantic_readout import (
    S56HierarchicalClauseRecipient,
)
from research.nra3_s20.s57_event_first_native_semantic_decoder import (
    create_event_first_features,fixed_train_old96_only,
    NODE_FEATURE_NAMES,TRAINABLE_PARAMS,
)

STAGE="NRA3_S57_EVENT_FIRST_OLD_SPENT96_GENUINE43M_GT_TRAIN_ONLY"
FILES=(
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
OLD_S56="NRA3_S56_SPENT72_ONLY_108FP16_SCOPE_RECIPIENT_REJECTED.safetensors"
OLD_S56_SHA="0ddb5a4739d883c7e6209a99e5ba4e398192ed6057746f15f36b7d87745609e5"

def require(ok,why):
    if not bool(ok):raise ValueError("S57 spent96 genuine43M training isolation: "+why)

def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()

def unique(words,needle):
    idx=[i for i,t in enumerate(words) if t.form==needle]
    require(len(idx)==1,"S57 previously scored TRAIN gold unique word not matched")
    return idx[0]

def main():
    ap=argparse.ArgumentParser()
    for key in ("s53","s54","s55","s56","s26","old-s56-paid","out"):
        ap.add_argument("--"+key,type=Path,required=True)
    a=ap.parse_args()
    require(not a.out.exists(),"S57 first old96 train result already exists")
    source_files=(a.s53,a.s54,a.s55,a.s56)
    require(all(f.name==info[1] and f.is_file()
                for f,info in zip(source_files,FILES))
            and a.s26.name==OLD_S26 and a.s26.is_file()
            and a.old_s56_paid.name==OLD_S56 and a.old_s56_paid.is_file()
            and sha(a.old_s56_paid)==OLD_S56_SHA,
            "S57 original GT old96 sources and frozen S56 FP16 SHA drift")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    # No fresh S57 HELD24 import. Old S53/54/55/56 have ALL been scored,
    # so previous HELD only becomes TRAIN here; never independent EVAL.
    from research.nra3_s20.s53_precommitted_negation_quote24_plain import SOURCE_ROWS as R53
    from research.nra3_s20.s53_separate_manual_gold_negation24 import gold24 as G53
    from research.nra3_s20.s54_frozen_new_held24_plain_source import SOURCE_ROWS as R54
    from research.nra3_s20.s54_frozen_new_held24_manual_gold import held_gold24 as G54
    from research.nra3_s20.s55_frozen_structural_shift_held24_plain import SOURCE_ROWS as R55
    from research.nra3_s20.s55_frozen_structural_shift_held24_manual_gold import held_gold24 as G55
    from research.nra3_s20.s56_frozen_new_scope24_plain import SOURCE_ROWS as R56
    from research.nra3_s20.s56_frozen_new_scope24_gold import held_gold24 as G56
    raw=(R53,R54,R55,R56)
    gold=(G53(),G54(),G55(),G56())
    require(all(len(r)==len(g)==24 for r,g in zip(raw,gold)),
            "S57 previous 4 x 24 original manually audited TRAIN data missing")
    old_s26=EventGroundedInversionGraph.from_physical(a.s26.read_bytes())
    w56=load_file(str(a.old_s56_paid),device="cpu")
    require(set(w56)=={"S56_SCOPE_PAIR108_FP16"},
            "old S56 FP16 paid research head archive contains unknown source")
    old_s56=S56HierarchicalClauseRecipient.from_physical_fp16(w56["S56_SCOPE_PAIR108_FP16"])
    require(old_s26.output_gate==old_s26.parent.output_gate==old_s56.output_gate==0
            and all(not p.requires_grad for mdl in (old_s26,old_s56)
                    for p in mdl.parameters()),
            "S57 prior 43M GT paid sources not frozen")
    old_s26_sha=tensor_sha(*(p.detach() for p in old_s26.parameters()))
    old_s56_sha=tensor_sha(*(p.detach() for p in old_s56.parameters()))
    nodes=[];pairs=[];validity=[];eligibility=[];src_targets=[];pair_targets=[];neg_targets=[]
    source_checks=[]
    rawword_counts=[]
    for (tag,name,rows_key),path,row_set,gold_set in zip(FILES,source_files,raw,gold):
        tensor=load_file(str(path),device="cpu")
        require(set(tensor)=={
            "original_real43m_gt_word339_READONLY",
            "original_native_word_valid_mask",
            "native_BPE_causally_unsafe_word_mask",rows_key},
            "source old real43M GT feature shard unexpectedly includes gold")
        x=tensor["original_real43m_gt_word339_READONLY"].detach().clone()
        valid=tensor["original_native_word_valid_mask"].detach().clone()
        unsafe=tensor["native_BPE_causally_unsafe_word_mask"].detach().clone()
        require(x.ndim==3 and x.shape[0]==24 and x.shape[2]==339
                and valid.shape==unsafe.shape==x.shape[:2]
                and valid.dtype==unsafe.dtype==torch.bool
                and tensor[rows_key].tolist()==list(range(24)),
                "S57 old original 43M GT native row counts/ABI changed")
        source_sha=tensor_sha(x,valid,unsafe)
        with torch.inference_mode():
            old_logits=old_s26(x[...,:338],valid)
            observed=S52QuoteAwareFrozenPaidGraphBridge(old_s26).observe(x,valid,unsafe)
            require(all(torch.equal(getattr(old_logits,k),
                                    getattr(observed.original_paid_graph_logits,k))
                        for k in ("event_status","role","argument_polarity",
                                  "coreference","event_relation")),
                    "S57 frozen S26 logits changed under research text safety mask")
            n,p,mask=create_event_first_features(
                x,observed,tuple(r[2] for r in row_set),old_s56)
        # Clone to normal non-inference tensors before autograd training.
        n=n.detach().clone()
        p=p.detach().clone()
        mask=mask.detach().clone()
        valid_word=observed.eligible_word_mask.detach().clone()
        require(not n.is_inference() and not p.is_inference()
                and not mask.is_inference() and not valid_word.is_inference(),
                "S57 old TRAIN96 feature tensors are still inference tensors")
        source_gold=torch.zeros_like(valid_word)
        directed_gold=torch.zeros_like(mask)
        negative_gold=torch.zeros_like(mask)
        for i,(sentence,g) in enumerate(zip(row_set,gold_set)):
            require(sentence[0]==g.row_id,"S57 past TRAIN manual source ID drift")
            words=raw_surface_grammar(sentence[2])
            require(len(words)==int(valid[i].sum()),
                    "S57 old frozen original CPU word source drift")
            if tag in ("S53","S54"):
                predicate=g.predicate
                receiver=g.recipient
                polarity=g.recipient_polarity
            elif tag=="S55":
                predicate=g.main_predicate
                receiver=g.transferred_recipient
                polarity=g.transfer_polarity
            else:
                predicate=g.main_predicate
                receiver=g.transferred_recipient
                polarity=g.directed_transfer_polarity
            if receiver is None:continue
            source_i=unique(words,predicate)
            target_j=unique(words,receiver)
            require(bool(valid_word[i,source_i])
                    and bool(mask[i,source_i,target_j]),
                    "S57 clause hard-mask mistakenly excludes old TRAIN true recipient!")
            source_gold[i,source_i]=True
            directed_gold[i,source_i,target_j]=True
            if polarity=="REJECTED":negative_gold[i,source_i,target_j]=True
        require(tensor_sha(x,valid,unsafe)==source_sha,
                "S57 source-only event features modified original 43M GT native")
        nodes.append(n);pairs.append(p);validity.append(valid_word);eligibility.append(mask)
        src_targets.append(source_gold);pair_targets.append(directed_gold);neg_targets.append(negative_gold)
        source_checks.append(sha(path));rawword_counts.append(int(valid.sum()))
    require(tensor_sha(*(p.detach() for p in old_s26.parameters()))==old_s26_sha
            and tensor_sha(*(p.detach() for p in old_s56.parameters()))==old_s56_sha,
            "S57 old physically trained S26/S56 paid semantic model changed")
    model,paid,diag=fixed_train_old96_only(
        nodes,pairs,validity,eligibility,src_targets,pair_targets,neg_targets)
    require(paid.dtype==torch.float16 and len(paid)==TRAINABLE_PARAMS
            and not model.candidate_accepted and model.output_gate==0
            and tensor_sha(*(p.detach() for p in old_s26.parameters()))==old_s26_sha
            and tensor_sha(*(p.detach() for p in old_s56.parameters()))==old_s56_sha,
            "S57 new event-first learned weights violated old frozen model constraints")
    a.out.mkdir(parents=True,exist_ok=False)
    file=a.out/"NRA3_S57_SPENT96_TRAINONLY_EVENT_FIRST_FP16.safetensors"
    save_file({"S57_EVENT_FIRST_FP16":paid},str(file))
    loaded=load_file(str(file),device="cpu")["S57_EVENT_FIRST_FP16"]
    require(torch.equal(loaded,paid),"S57 physically paid first 96-only decoder not roundtrip exact")
    report={
      "stage":STAGE,
      "old_96_original_Calibri43M_GT_native_source_runs":[44274389010,44279445613,44282672013,44308286512],
      "real_original_Calibri43M_GT_old96_native_SHA256":source_checks,
      "old96_true_original_GreenTunnel_native_word_counts":rawword_counts,
      "old_frozen_S26_paid_archive_SHA256":sha(a.s26),
      "old_frozen_S56_108FP16_archive_SHA256":sha(a.old_s56_paid),
      "S57_new_trainable_FP16_parameter_count":TRAINABLE_PARAMS,
      "S57_new_trainable_node_features":list(NODE_FEATURE_NAMES),
      "S57_new_FP16_file":file.name,
      "S57_new_FP16_file_SHA256":sha(file),
      "S57_old96_only_TRAIN_not_independent":diag,
      "original_Calibri43M_GreenTunnel_weights_unchanged":True,
      "original_S26_paid12384_and_S56_paid108_weights_unchanged":True,
      "S57_NEW_HELD24_source_and_gold_NOT_IMPORTED_in_training":True,
      "at_most_one_transfer_edge_per_utterance_by_design":True,
      "event_source_and_target_gold_never_passed_into_model_forward":True,
      "fullcontext_offline_noncausal_real43M_not_streaming":True,
      "active_candidate_616J7_bpp_CB59_NOT_certified":True,
      "no_protected_S38_S47_S48_HELD26_EVAL32_128":True,
      "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    p=a.out/"NRA3_S57_SPENT96_EVENT_FIRST_TRAIN_ONLY_REPORT.json"
    p.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                 encoding="utf-8")
    print("S57_EVENT_FIRST_TRAIN_OLD96_COMPLETED",
          json.dumps({"src_gold":diag["old_spent96_gold_source_events"],
                      "rej_gold":diag["old_spent96_gold_REJECTED_edges"],
                      "new_trainable":TRAINABLE_PARAMS,
                      "old_NATIVE_original_GT":rawword_counts,
                      "saved_SHA256":sha(file),
                      "S57_fresh_HELD_unseen":True,"output_gate":0},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
