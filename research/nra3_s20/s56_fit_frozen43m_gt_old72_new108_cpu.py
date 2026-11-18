"""S56 fixed 108 FP16 tiny grammar/semantic readout training on SPENT S53+S54+S55.

Three physically completed original Calibri43M+Green Tunnel post-layer8
native gold-BLIND captures and existing paid frozen S26 five-head model.
Human gold from already inspected S53, S54, S55 may supply gradients.
The fresh S56 HELD24 plaintext/manual gold must not be imported, present
in arguments or read by this training process.

Original backbone/GT, paid S26 and S54 are never modified. No merging,
LIVE, or streaming decoder; new output gate0.
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
from research.nra3_s20.s56_grammar_clause_scope_semantic_readout import (
    FEATURE_NAMES,PAID_PARAM_COUNT,real43m_semantic_structure_features,
    fixed_train_from_SPENT_S53_S54_S55_only,
)

STAGE="NRA3_S56_SPENT_S53_S54_S55_TRAIN72_TRUE43M_GT_SCOPE_108FP16"
CASES=(
    ("S53","NRA3_S53_GENUINE43M_GT_NEGATION24_WORD339_AND_UNSAFE.safetensors",
     "source_row_indices_in_frozen_S53_NEW24"),
    ("S54","NRA3_S54_GENUINE43M_GT_HELD24_WORD339_AND_UNSAFE.safetensors",
     "source_row_indices_in_frozen_S54_HELD24"),
    ("S55","NRA3_S55_GENUINE43M_GT_STRUCTURAL24_WORD339_AND_UNSAFE.safetensors",
     "source_row_indices_in_frozen_S55_HELD24"),
)
OLD_S26="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"


def require(ok,why):
    if not bool(ok):raise ValueError("S56 real43M GT spent72 train: "+why)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def find(tokens,name):
    match=[i for i,t in enumerate(tokens) if t.form==name]
    require(len(match)==1,
            "previously inspected TRAIN gold word missing exact original native mapping")
    return match[0]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s53",type=Path,required=True)
    p.add_argument("--s54",type=Path,required=True)
    p.add_argument("--s55",type=Path,required=True)
    p.add_argument("--s26",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"cannot overwrite S56 frozen first train output")
    require(a.s26.name==OLD_S26 and a.s26.is_file(),
            "old trained original S26 12384 physical FP16 missing")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data=(a.s53,a.s54,a.s55)
    require(all(file.name==name and file.is_file()
                for (key,name,idx),file in zip(CASES,data)),
            "S56 only previous real original43M S53+S54+S55 native files accepted")
    # Direct imports ONLY from previously inspected, spent TRAIN labels.
    from research.nra3_s20.s53_precommitted_negation_quote24_plain import (
        SOURCE_ROWS as SOURCE53,
    )
    from research.nra3_s20.s53_separate_manual_gold_negation24 import (
        gold24 as GOLD53,
    )
    from research.nra3_s20.s54_frozen_new_held24_plain_source import (
        SOURCE_ROWS as SOURCE54,
    )
    from research.nra3_s20.s54_frozen_new_held24_manual_gold import (
        held_gold24 as GOLD54,
    )
    from research.nra3_s20.s55_frozen_structural_shift_held24_plain import (
        SOURCE_ROWS as SOURCE55,
    )
    from research.nra3_s20.s55_frozen_structural_shift_held24_manual_gold import (
        held_gold24 as GOLD55,
    )
    rows=(SOURCE53,SOURCE54,SOURCE55)
    truths=(GOLD53(),GOLD54(),GOLD55())
    require(all(len(x)==24 and len(y)==24 for x,y in zip(rows,truths)),
            "S56 training must only consume exactly 72 PREVIOUSLY SCORED sentences")
    old_bytes=a.s26.read_bytes()
    old=EventGroundedInversionGraph.from_physical(old_bytes)
    require(old.output_gate==old.parent.output_gate==0
            and all(not p.requires_grad for p in old.parameters()),
            "S56 original S26 paid graph was not frozen")
    model_sha=tensor_sha(*(p.detach() for p in old.parameters()))
    inputs_sha=[]
    features=[]
    safe_masks=[]
    targets=[]
    neg_targets=[]
    native_counts=[]
    for (tag,filename,row_name),file,raw,gold in zip(CASES,data,rows,truths):
        native=load_file(str(file),device="cpu")
        require(set(native)=={
            "original_real43m_gt_word339_READONLY",
            "original_native_word_valid_mask",
            "native_BPE_causally_unsafe_word_mask",row_name},
            "original 43M actual native source "+tag+" cannot contain gold fields")
        x=native["original_real43m_gt_word339_READONLY"].detach().clone()
        valid=native["original_native_word_valid_mask"].detach().clone()
        unsafe=native["native_BPE_causally_unsafe_word_mask"].detach().clone()
        require(x.ndim==3 and x.shape[0]==24 and x.shape[-1]==339
                and valid.shape==unsafe.shape==x.shape[:2]
                and native[row_name].tolist()==list(range(24))
                and valid.dtype==unsafe.dtype==torch.bool,
                "real 43M+GT old native S53/S54/S55 rows not aligned")
        source_sha=tensor_sha(x,valid,unsafe)
        inputs_sha.append(sha(file))
        with torch.inference_mode():
            baseline=old(x[...,:338],valid)
            obs=S52QuoteAwareFrozenPaidGraphBridge(old).observe(x,valid,unsafe)
            require(all(torch.equal(getattr(baseline,k),
                                    getattr(obs.original_paid_graph_logits,k))
                        for k in ("event_status","role","argument_polarity",
                                  "coreference","event_relation")),
                    "S56 old paid S26 five logits changed under grammar bridge")
            # Clone OUTSIDE inference mode before new readout autograd.
            z=real43m_semantic_structure_features(
                 x,obs,tuple(r[2] for r in raw))
        z=z.detach().clone()
        m=obs.eligible_directed_pair_mask.detach().clone()
        require(not z.is_inference() and not m.is_inference(),
                "S56 train features or eligible mask are inference tensors")
        r_gold=torch.zeros_like(m)
        neg_gold=torch.zeros_like(m)
        for i,(sentence,ref) in enumerate(zip(raw,gold)):
            require(sentence[0]==ref.row_id,
                    "S56 TRAIN original raw/gold row order mismatch")
            words=raw_surface_grammar(sentence[2])
            require(len(words)==int(valid[i].sum()),
                    "S56 TRAIN original native+text word positions differ")
            receiver=(ref.transferred_recipient if tag=="S55" else ref.recipient)
            predicate=(ref.main_predicate if tag=="S55" else ref.predicate)
            polarity=(ref.transfer_polarity if tag=="S55" else ref.recipient_polarity)
            if receiver is None:continue
            src=find(words,predicate)
            dst=find(words,receiver)
            require(bool(m[i,src,dst]),
                    "true TRAIN directed recipient excluded by original BPE/quote mask")
            r_gold[i,src,dst]=True
            if polarity=="REJECTED":neg_gold[i,src,dst]=True
        require(tensor_sha(x,valid,unsafe)==source_sha,
                "original 43M+GT native tensors changed by source-only feature derivation")
        features.append(z)
        safe_masks.append(m)
        targets.append(r_gold)
        neg_targets.append(neg_gold)
        native_counts.append(int(valid.sum()))
    learner,physical,diag=fixed_train_from_SPENT_S53_S54_S55_only(
        features,safe_masks,targets,neg_targets)
    require(physical.shape==(2,len(FEATURE_NAMES))
            and physical.numel()==PAID_PARAM_COUNT==108
            and physical.dtype==torch.float16
            and tensor_sha(*(p.detach() for p in old.parameters()))==model_sha
            and old.output_gate==old.parent.output_gate==0
            and learner.output_gate==0
            and not learner.candidate_accepted
            and not learner.activation_allowed
            and all(not p.requires_grad for p in learner.parameters()),
            "S56 actual GT/S26 unchanged or candidate somehow activated")
    a.out.mkdir(parents=True,exist_ok=False)
    file=a.out/"NRA3_S56_SPENT72_ONLY_108FP16_SCOPE_RECIPIENT_REJECTED.safetensors"
    save_file({"S56_SCOPE_PAIR108_FP16":physical},str(file))
    verify=load_file(str(file),device="cpu")
    require(torch.equal(verify["S56_SCOPE_PAIR108_FP16"],physical),
            "S56 physically persisted 108 FP16 train-only candidate drift")
    report={
      "stage":STAGE,
      "true_original_Calibri43M_GT_SRC_CPU_runs":[44274389010,44279445613,44282672013],
      "old_physically_paid_S26_12384_FP16_run":44196720655,
      "original_real43M_GT_training_source_native_shas":inputs_sha,
      "genuine_original43M_GT_training_source_raw_word_counts":native_counts,
      "old_paid_S26_source_SHA256":sha(a.s26),
      "new_108_FP16_feature_names":list(FEATURE_NAMES),
      "new_108_FP16_standalone_weights_filename":file.name,
      "new_108_FP16_standalone_weights_SHA256":sha(file),
      "fixed_optimizer_and_training_CONSUMED_SPENT72_ONLY":diag,
      "sum_training_gold_RECIPIENT":sum(int(x.sum()) for x in targets),
      "sum_training_gold_REJECTED":sum(int(x.sum()) for x in neg_targets),
      "source_original43M_GT_postlayer8_tensor_unchanged":True,
      "old_12384_FP16_S26_paid_parameter_tensors_unchanged":True,
      "S56_fresh_new_scope_HELD24_text_and_gold_NOT_IMPORTED":True,
      "new_readout_not_43M_transformer_GreenTunnel_or_original_S26":True,
      "not_streaming_causal_online_decoder":True,
      "active_candidate_616J7_bpp_CB59_NOT_VERIFIED":True,
      "no_S38_S47_S48_held26_or_independent_EVAL128_EVAL32":True,
      "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (a.out/"NRA3_S56_SPENT72_SCOPE_RECIPIENT_108FP16_TRAIN_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf-8")
    print("S56_FROZEN_OLD_REAL43M_GT_S53_S54_S55_TRAIN72_SCOPE_READOUT",
          json.dumps({"native_words":native_counts,
                      "old_gold_recipient":report["sum_training_gold_RECIPIENT"],
                      "old_gold_REJECTED":report["sum_training_gold_REJECTED"],
                      "new_paid_params":108,
                      "trained_fp16_sha":sha(file),
                      "original43M_and_GT_unchanged":True,
                      "S56_fresh_HELD24_unseen":True,
                      "gate":0},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
