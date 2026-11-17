"""S19d: contextual future/NEGATED event scope TRAIN-only test on true CPU features.

Twelve fixed S18 TRAIN-internal splits with fully train-side parent grammar
retraining. Current architecture was inspired by the S19c TRAIN CV failure:
these folds are DEVELOPMENT, NOT pristine evaluation or novel EVAL. The
original frozen 43M+GT / 616 J7 remains untouched. 9 S19 TRAIN-only event
graphs are NOT neural input yet: genuine frozen original43M hidden320 for them
has not been captured. The entire S18 original source96 remains unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from research.nra3_s18.real43m_train96_physical_source import (
    checked_real_train96, grouped_train96_split_contract, S18B_TRUE_CPU_RUN,
)
from research.nra3_s19.s19b_train96_true43m_grammar import (
    assemble_true43m_train96, subset,
    train_fixed as train_grammar_fixed, STEPS as GRAMMAR_STEPS,
)
from research.nra3_s19.paid_grammar_parser import (
    load_paid_fp16_bytes, trainable_parameter_count,
)
from research.nra3_s19.s19c_paid_event_factuality import STATUS
from research.nra3_s19.s19c_train96_real43m_event_factuality import (
    build_train_targets, event_subset, evaluate_frozen, aggregate,
)
from research.nra3_s19.s19d_paid_contextual_scope import (
    PaidContextualEventScope, EXTRA_PARAMETERS, require,
)

SEED = 319104
STEPS = 160
LR = 0.012
WEIGHTS = torch.tensor([.25, 1., 2., 2.],dtype=torch.float32)
PARENT_S19B_SHA="7341becd1693fa0de09d1bce8252e2f5f7a00ae96c255664112277b024bc3915"
S19C_BASELINE_TRAIN_LEX=0.6355140186915887
S19C_BASELINE_TRAIN_SYNTAX=0.4549763033175355
S19C_BASELINE_CI=44126483606
S19C_FINAL_PHYSICAL_SHA="a3a8a4de32e8669c851437ad09340512609d193a607bcadc58dc5be73cb80d2a"


def fixed_train_context(frozen_grammar,word_batch,gold):
    require(all(not p.requires_grad for p in frozen_grammar.parameters())
            and frozen_grammar.output_gate==0 and not frozen_grammar.candidate_accepted,
            "do not train previously frozen grammar weights in scope stage")
    torch.manual_seed(SEED)
    model=PaidContextualEventScope(frozen_grammar)
    optimizer=torch.optim.AdamW(
        list(model.context.parameters())+list(model.event_head.parameters()),
        lr=LR,weight_decay=.0005)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits=model(word_batch.input_338,word_batch.valid)
        truth=gold.status[word_batch.valid]
        loss=F.cross_entropy(logits[word_batch.valid], truth, weight=WEIGHTS)
        require(bool(torch.isfinite(loss)), "nonfinite TRAIN-only scope loss")
        loss.backward()
        require(all(p.grad is None for p in model.grammar.parameters()) and
                all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in [*model.context.parameters(),*model.event_head.parameters()]),
                "scope weights must learn, grammar weights must stay frozen")
        torch.nn.utils.clip_grad_norm_(
            list(model.context.parameters())+list(model.event_head.parameters()),1.)
        optimizer.step()
    physical=model.physical_bytes()
    paid=PaidContextualEventScope.from_physical(physical).eval()
    require(paid.head_physical()==model.head_physical()
            and not any(p.requires_grad for p in paid.parameters()),
            "complete physically billed contextual head must reload to FP16")
    return paid,physical


def main():
    parser=argparse.ArgumentParser()
    for label in ("s18b-hidden","s18b-report","s19b-full-grammar","out"):
        parser.add_argument("--"+label,required=True,type=Path)
    args=parser.parse_args()
    require(not args.out.exists(),"never overwrite original research artifacts")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    samples,report=checked_real_train96(args.s18b_hidden,args.s18b_report)
    words=assemble_true43m_train96(samples,report)
    truth=build_train_targets(samples,words)
    folds=grouped_train96_split_contract()
    require(len(folds)==12 and S18B_TRUE_CPU_RUN==43910479658
            and report["all_616_native_CPU_generated_token_digests_unchanged"]
            and EXTRA_PARAMETERS==2804, "source or physically paid scope ABI drift")
    entries=[]
    for axis,fold_id,held_groups,training_ids,dev_ids in folds:
        require(set(training_ids).isdisjoint(dev_ids)
                and sorted((*training_ids,*dev_ids))==list(range(96)),
                "TRAIN-internal leave-group-out violation")
        tr=subset(words,training_ids)
        dev=subset(words,dev_ids)
        gtr=event_subset(truth,training_ids)
        gdev=event_subset(truth,dev_ids)
        # Critical: the full96 pretrained parser is NOT loaded in any CV fold.
        local_grammar,grammar_bytes=train_grammar_fixed(tr)
        paid,joined=fixed_train_context(local_grammar,tr,gtr)
        result=evaluate_frozen(paid,dev,gdev)
        entries.append({
          "axis":axis,"fold":fold_id,"held_groups":list(held_groups),
          "training_rows":len(training_ids),"internal_dev_rows":len(dev_ids),
          "TRAIN_only_gold_status_counts":{
            x:int((gtr.status[tr.valid]==i).sum())
            for i,x in enumerate(STATUS)
          },
          "all_grammar_weights_refit_using_train_rows_only":True,
          "held_rows_never_in_grammar_or_event_loss":True,
          "parent_grammar_FP16_SHA":hashlib.sha256(grammar_bytes).hexdigest(),
          "physical_joint_FP16_SHA":hashlib.sha256(joined).hexdigest(),
          **result
        })
        print("S19D_TRAIN_INTERNAL",fold_id,
              "joint_F1",result["joint_predicate_factuality_F1"],
              "negated_TP",result["per_status"]["NEGATED"]["TP"],
              "future_TP",result["per_status"]["FUTURE_UNREALIZED"]["TP"],
              flush=True)
    root=args.s19b_full_grammar.read_bytes()
    require(hashlib.sha256(root).hexdigest()==PARENT_S19B_SHA,
            "full96 original paid grammar SHA modified")
    full_grammar=load_paid_fp16_bytes(root)
    require(trainable_parameter_count(full_grammar)==6528,
            "frozen parent real learned grammar parameter count drift")
    full,physical=fixed_train_context(full_grammar,words,truth)
    args.out.mkdir(parents=True,exist_ok=False)
    name="NRA3_S19D_TRUE43M_GT_CONTEXT_SCOPE_GRAMMAR_AND_EVENT_TRAINONLY_FP16.n3s19d"
    (args.out/name).write_bytes(physical)
    require((args.out/name).read_bytes()==physical
            and PaidContextualEventScope.from_physical(physical).head_physical()
                ==full.head_physical(),"physical FP16 training output corrupt")
    lex=aggregate([z for z in entries if z["axis"]=="lexical"])
    syntax=aggregate([z for z in entries if z["axis"]=="syntax"])
    require(lex["all_raw_words"]==syntax["all_raw_words"]==600
            and lex["event_gold_count"]==syntax["event_gold_count"]==108,
            "heldout TRAIN-only groups must cover original 96 exactly")
    summary={
      "stage":"NRA3_S19D_REAL43M_GT_TRAIN96_CONTEXTUAL_EVENT_SCOPE",
      "S18b_true_CPU_genuine_original43M_GT_source_run":S18B_TRUE_CPU_RUN,
      "S19b_full_train_frozen_grammar_sha":PARENT_S19B_SHA,
      "S19c_frozen_original_comparison_run":S19C_BASELINE_CI,
      "S19c_frozen_original_final_fp16_sha":S19C_FINAL_PHYSICAL_SHA,
      "S19c_baseline_lexical_TRAIN_INTERNAL_micro_F1":S19C_BASELINE_TRAIN_LEX,
      "S19c_baseline_syntax_TRAIN_INTERNAL_micro_F1":S19C_BASELINE_TRAIN_SYNTAX,
      "new_context_lexical_TRAIN_INTERNAL_micro_F1":lex["joint_predicate_factuality_micro_F1"],
      "new_context_syntax_TRAIN_INTERNAL_micro_F1":syntax["joint_predicate_factuality_micro_F1"],
      "delta_from_previous_lexical_TRAIN_INTERNAL":
         lex["joint_predicate_factuality_micro_F1"]-S19C_BASELINE_TRAIN_LEX,
      "delta_from_previous_syntax_TRAIN_INTERNAL":
         syntax["joint_predicate_factuality_micro_F1"]-S19C_BASELINE_TRAIN_SYNTAX,
      "TRAIN_96_source_sha":words.source_sha,
      "original_frozen_43M_GT_hidden_sha":words.hidden_sha,
      "original_616_J7_PARENT_digest_unchanged":True,
      "input_real_hidden320_and_plain_text_raw18_only":True,
      "all_predicates_and_status_predicted_without_gold":True,
      "grammar_weights_frozen_in_context_training":True,
      "crossval_grammar_from_scratch_on_train_rows_only":True,
      "train_only_fixed_grammar_steps":GRAMMAR_STEPS,
      "train_only_fixed_context_steps":STEPS,
      "train_only_lexical_aggregate":lex,
      "train_only_syntax_aggregate":syntax,
      "train_only_folds":entries,
      "train_only_full96_resubstitution_NOT_generalization":
         evaluate_frozen(full,words,truth),
      "grammar_6528_FP16_parameters":6528,
      "context_2804_paid_FP16_parameters":EXTRA_PARAMETERS,
      "combined_9332_FP16_parameters":6528+EXTRA_PARAMETERS,
      "complete_combined_physical_bytes":len(physical),
      "combined_FP16_SHA256":hashlib.sha256(physical).hexdigest(),
      "physical_filename":name,
      "negative_role_recipient_or_coreference_NOT_TRAINED":True,
      "S19_train9_new_human_event_source_not_captured_on_true43M":True,
      "previous_TRAIN_internal_folds_informed_new_architecture":
          "YES_NOT_PRISTINE_EVAL",
      "NO_consumed_EVAL32_EVAL128_S17c_S15c_S12c_reused":True,
      "strict_online_word_causality_NOT_PROVEN":True,
      "candidate_active_616_J7_NOT_CERTIFIED":True,
      "all_in_bpp_le07_NOT_CERTIFIED":True,
      "full_CPU_ge90pct_CB59_NOT_CERTIFIED":True,
      "NRA2_event_graph_connected_to_decoder":False,
      "output_gate":0,"candidate_accepted":False,
      "activation_allowed":False,
      "original_Calibri_GT_LIVE_Render_Supabase_Jessica_APK_unchanged":True,
    }
    dest=args.out/"NRA3_S19D_TRUE43M_GT_TRAIN96_CONTEXTUAL_SCOPE_REPORT.json"
    dest.write_text(json.dumps(summary,indent=2,sort_keys=True,ensure_ascii=False)+"\n",
                    encoding="utf-8")
    print("S19D_FINAL_LEXICAL_TRAIN_INTERNAL",json.dumps(lex,sort_keys=True),flush=True)
    print("S19D_FINAL_SYNTAX_TRAIN_INTERNAL",json.dumps(syntax,sort_keys=True),flush=True)
    print("S19D_CONTEXT_FINAL_FULL_FP16_SHA",summary["combined_FP16_SHA256"],flush=True)


if __name__=="__main__":
    main()
