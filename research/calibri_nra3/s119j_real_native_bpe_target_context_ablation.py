"""S119j: move from invented hidden-state targets to ACTUAL next native BPE IDs.

Directly follows the user's demand for real universal-response progress rather
than proliferating neural regions or reporting only gradient/representation
smoke tests. We still cannot certify universal answer quality: original
Calibri 43M+GreenTunnel decoder is NOT dynamically run or updated here.
S114 contains physically measured real all12 hidden + ACTUAL native BPE IDs
from original CB59+GT on licensed SynTagRus-A TRAIN ONLY. This test uses
those true next-token IDs in a narrowly restricted **TRAIN-derived candidate
lexicon**. It is teacher-forced, not autoregressive generation and the
restricted prototype-vocabulary score is NOT full LM next-token likelihood.

Compared under identical training work and exact identical total trainable
parameters: context-aware structured 8-role NRA3+NPA1+core versus token-local
(only relational lookback disabled). No probe text/gold used in construction
of class keys; keys are frozen 320D lexical prototypes from original TRAIN
rows only. Probe rows are from the SAME TRAIN corpus and may have been read
in prior research; this is not independent unseen generalization or
full native LM softmax. Coverage of known vocabulary is always reported;
OOV labels NEVER silently remapped or counted as correct.

Original old 53,751 learned NRA3 FP16 weights, 616 baseline, LIVE,
Render, Supabase, Jessica, main, S118 SynTagRus-B GOLD, previous spent
developer/EVAL data are never mutated or opened.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119i_real12_contextual_structured_operator_ablation import (
    UnifiedCalibriWithStructuredInterface,
    sha, HIDDEN, FROZEN_PRETRAINED, PARENT_PARAMETERS,
)
from research.calibri_nra3.s119g_unified_planning_internal_action_neural_area import (
    STAGE as PARENT_STAGE, CHECKPOINT as PARENT_CHECKPOINT,
    REPORT as PARENT_REPORT, PREFIX as PARENT_PREFIX,
    teacher_targets,
)
from research.calibri_nra3.s119d_reuse_old37087_true12_inweight_eventgraph import (
    PARENT_H, PARENT_H_SHA, PARENT_REPORT as SOURCE_REPORT,
)

STAGE = "CALIBRI_S119J_REAL_ORIGINAL_NATIVE_BPE_TARGET_CONTEXTUAL_VS_LOCAL_TRAIN_ONLY"
REPORT = "CALIBRI_S119J_REAL_NATIVE_NEXT_BPE_CONTEXT_VS_LOCAL_PROBE_ONLY.json"
CHILD = "CALIBRI_S119J_CONTEXTUAL_REAL_NATIVE_BPE_TRAINONLY_NOT_ACCEPTED.safetensors"
VOCAB = "CALIBRI_S119J_TRAIN_ONLY_FROZEN_NATIVE_BPE_PROTOTYPES.safetensors"
KEYS_FROM = range(0, 128)
TRAIN_FROM = range(0, 64)
PROBE_FROM = range(128, 160)
TRAIN_EXAMPLES = 32
PROBE_EXAMPLES = 48
EPOCHS = 2
PROJECTION_SCALE = 10.
SEED = 119123


def original_source_valid(hidden, mask, ids):
    check(hidden.ndim == 4 and
          hidden.shape[0] == 256 and hidden.shape[1] == 12 and
          hidden.shape[-1] == HIDDEN and
          hidden.dtype == torch.float16 and
          ids.dtype == torch.int32 and mask.dtype == torch.bool and
          mask.shape == ids.shape == (256, hidden.shape[2]) and
          bool(torch.equal(ids[~mask],
                           torch.full_like(ids[~mask], -1))) and
          bool((ids[mask] >= 0).all()) and
          bool(torch.isfinite(hidden).all()),
          "real S114 original CB59+GT H320/native vocabulary geometry required")


def build_frozen_train_lexicon(hidden, mask, ids):
    """Exact native tokenizer IDs, prototypes from ONLY TRAIN(0..127).

    Using original earliest-layer states as lexical keys does NOT
    constitute the true compressed original token embedding / LM head.
    No labels or states from PROBE(128..159) can enter candidate keys.
    """
    counts = torch.zeros((0,), dtype=torch.long)
    words = []
    features = []
    for row in KEYS_FROM:
        n = int(mask[row].sum())
        check(n >= 8 and bool(mask[row, :n].all()) and
              not bool(mask[row, n:].any()),
              "TRAIN-only S114 native token rows contiguous")
        words.append(ids[row, :n].long())
        features.append(hidden[row, 0, :n].float())
    vocabulary = torch.unique(torch.cat(words), sorted=True)
    check(vocabulary.ndim == 1 and vocabulary.numel() > 100 and
          bool((vocabulary >= 0).all()), "frozen real native BPE vocab sparse")
    sums = torch.zeros((vocabulary.numel(), HIDDEN))
    counts = torch.zeros((vocabulary.numel(),), dtype=torch.long)
    for token_ids, row_embeds in zip(words, features):
        ix = torch.searchsorted(vocabulary, token_ids)
        sums.index_add_(0, ix, row_embeds)
        counts.index_add_(0, ix, torch.ones_like(ix))
    prototypes = F.normalize(sums / counts.clamp_min(1).unsqueeze(-1), dim=-1)
    check(bool(torch.isfinite(prototypes).all()) and
          bool((counts > 0).all()) and
          bool((prototypes.abs().sum(-1) > 0).all()) and
          prototypes.shape == (vocabulary.numel(), 320),
          "training-only source lexical-prototype construction")
    return vocabulary, prototypes, counts


def label_to_index(vocabulary, token):
    value = int(token)
    j = int(torch.searchsorted(vocabulary, value))
    return j if (j < vocabulary.numel() and
                 int(vocabulary[j]) == value) else None


def collect_targets(mask, ids, vocabulary, segment):
    """Precommitted deterministic teacher-forced prefix positions.

    Probe sampling NEVER uses accuracy. Report support and OOV together
    before choosing max PROBE_EXAMPLES. No next token is model input.
    """
    items = []
    unknown = 0
    all_position_count = 0
    for row in segment:
        n = int(mask[row].sum())
        check(n >= 12 and bool(mask[row,:n].all()) and
              not bool(mask[row,n:].any()),
              "original BPE row must be contiguous")
        for prefix in range(8, min(n-1, 48), 5):
            all_position_count += 1
            token = int(ids[row, prefix])
            label = label_to_index(vocabulary, token)
            if label is None:
                unknown += 1
            else:
                items.append((row, prefix, label, token))
    check(all_position_count == len(items)+unknown,
          "no unknown test labels may disappear")
    return items, {
        "sampled_next_token_positions":all_position_count,
        "within_TRAIN_lexicon_positions":len(items),
        "unknown_next_token_positions_NOT_EVALUATED":unknown,
        "known_token_coverage":len(items)/all_position_count,
    }


def logits_at_actual_native_bpe(model, h, valid, prefix, frozen_prototypes, *,
                                 contextual):
    """Actual next native BPE class is supervised; no future token lookup.

    Full prior 12-layer hidden capture is sliced BEFORE next position;
    S114 source capture was full-sequence, so true prefix-only original
    GT causal equivalence is still NOT recertified here.
    """
    check(h.shape[0] == 1 and valid.shape == (1,h.shape[2]) and
          2 <= prefix < h.shape[2] and bool(valid[:,:prefix].all()) and
          bool(torch.isfinite(frozen_prototypes).all()) and
          frozen_prototypes.shape[1] == HIDDEN,
          "full prefix and frozen source-trained BPE key bank required")
    graph, relational, plan, projected = model(
        h[:,:,:prefix], valid[:,:prefix], contextual=contextual)
    logits = PROJECTION_SCALE * (
        F.normalize(projected,dim=-1) @ frozen_prototypes.T)
    check(logits.shape == (1, frozen_prototypes.shape[0]) and
          bool(torch.isfinite(logits).all()),
          "must predict original next native tokenizer BPE class")
    return logits, graph, relational, plan


def source_only_objective(model, h, valid, prefix, true_token_label_index,
                          frozen_prototypes, *, contextual):
    logits, graph, relation, plan = logits_at_actual_native_bpe(
        model,h,valid,prefix,frozen_prototypes,contextual=contextual)
    label=torch.tensor([true_token_label_index],dtype=torch.long)
    native_ce=F.cross_entropy(logits,label)
    with torch.no_grad():
        target_event,target_relation,old_mini = teacher_targets(
            model,h,valid,prefix)
    event_ce=-(target_event*F.log_softmax(
        plan.next_event_logits,-1)).sum(-1).mean()
    relational_ce=-(target_relation*F.log_softmax(
        relation,-1)).sum(-1)[valid[:,:prefix]].mean()
    retention=(graph.mini_area_logits[valid[:,:prefix]]-
               old_mini[valid[:,:prefix]]).square().mean()
    total=native_ce+0.10*event_ce+0.02*relational_ce+0.005*retention
    check(bool(torch.isfinite(total)), "real BPE+old NRA3 retention loss finite")
    return total, {
        "native_BPE_candidate_crossentropy":float(native_ce.detach()),
        "native_BPE_candidate_top1":int(logits.argmax(-1).item()==true_token_label_index),
        "source_only_event_soft_consistency":float(event_ce.detach()),
        "source_only_NRA3_relation_consistency":float(relational_ce.detach()),
    }


def frozen_unigram_baseline(vocab,counts,observations):
    """Simple TRAIN-derived prior; use SAME restricted vocabulary and labels."""
    seen=torch.zeros(vocab.numel(),dtype=torch.long)
    for _,_,label,_ in observations:
        seen[label]+=1
    dist=(seen.float()+1.)/(seen.sum().float()+vocab.numel())
    return torch.log(dist)


def evaluate(model,h,mask,examples,prototypes,contextual,baseline_logprobs):
    model.eval()
    ce=0.
    acc=0
    unice=0.
    uniacc=0
    top1=[]
    with torch.inference_mode():
        for idx,prefix,label,token in examples:
            n=int(mask[idx].sum())
            src=h[idx:idx+1,:,:n].float()
            valid=mask[idx:idx+1,:n]
            scores,_,_,_=logits_at_actual_native_bpe(
                model,src,valid,prefix,prototypes,contextual=contextual)
            ce+=float(F.cross_entropy(
                scores,torch.tensor([label])).item())
            pred=int(scores.argmax(-1))
            acc+=int(pred==label)
            unice+=float(-baseline_logprobs[label])
            uniacc+=int(int(baseline_logprobs.argmax())==label)
            top1.append({"row":idx,"prefix_length":prefix,
                         "expected_native_BPE_id":token,
                         "predicted_native_BPE_id":
                            None})  # omit wrong next tokens in public report
    n=len(examples)
    check(n>=1, "at least one true native BPE-covered probe")
    return {
        "restrict_vocab_next_native_BPE_CE":ce/n,
        "restrict_vocab_next_native_BPE_top1":acc,
        "restrict_vocab_next_native_BPE_count":n,
        "train_unigram_CE_reference_same_examples":unice/n,
        "train_unigram_top1_reference_same_examples":uniacc,
        "not_full_vocabulary_generation":True,
    }


def train(model,h,mask,examples,prototypes,*,contextual):
    model.train()
    old={k:v.detach().half().clone()
         for k,v in model.nra3_neural_weights.state_dict().items()}
    init_fuse=model.structured_interface.fuse.weight.detach().clone()
    opt=torch.optim.Adam(model.new_joint_trainables(),lr=.00045)
    gradients=[]
    for epoch in range(EPOCHS):
        for j,(idx,prefix,label,token) in enumerate(examples):
            n=int(mask[idx].sum())
            src=h[idx:idx+1,:,:n].float()
            valid=mask[idx:idx+1,:n]
            opt.zero_grad(set_to_none=True)
            objective,parts=source_only_objective(
                model,src,valid,prefix,label,prototypes,contextual=contextual)
            objective.backward()
            grads={
                "new_structured_interface_fuse":
                    model.structured_interface.fuse.weight.grad,
                "new_rank4_core_up":model.core_growth.up.grad,
                "new_neural_NRA3_relational_gate":
                    model.reasoning_role_gate.grad,
                "new_neural_NPA1_action_selector":
                    model.planning_action_area.action_selector.weight.grad,
                "new_next_native_BPE_candidate_prediction":
                    model.next_state_projector.weight.grad,
            }
            if epoch == EPOCHS-1 and j == 0:
                for name,g in grads.items():
                    check(g is not None and bool(torch.isfinite(g).all())
                          and float(g.abs().sum())>1e-12,
                          "must jointly reach "+name)
                gradients=[{"name":name,"gradient_l1":float(g.abs().sum())}
                            for name,g in grads.items()]
            torch.nn.utils.clip_grad_norm_(model.new_joint_trainables(),1.)
            opt.step()
    check(not torch.equal(init_fuse,model.structured_interface.fuse.weight.detach()),
          "new representation requires real weight updates")
    check(all(torch.equal(v.detach().half(),old[k])
              for k,v in model.nra3_neural_weights.state_dict().items()),
          "all trained prior 53751 NRA3 weights must remain full FP16 exact")
    return gradients


def run(prior:Path,prior_report:Path,source:Path,source_report:Path,out:Path):
    check(not out.exists() and prior.name==PARENT_CHECKPOINT
          and prior_report.name==PARENT_REPORT
          and source.name==PARENT_H and source_report.name==SOURCE_REPORT
          and sha(source)==PARENT_H_SHA,
          "only old S119g unified model and original SHA-pinned S114 source")
    p=json.loads(prior_report.read_text())
    s=json.loads(source_report.read_text())
    check(p.get("stage")==PARENT_STAGE and
          p.get("physical_single_child_SHA256")==sha(prior) and
          p.get("physical_single_child_all_FP16_parameters")==116777 and
          p.get("all_53751_original_NRA2_NRA3_params_preserved_byte_exact") is True
          and p.get("output_gate")==0 and p.get("candidate_accepted") is False
          and s.get("ALL12_CPU_SAFETENSORS_SHA256")==PARENT_H_SHA and
          s.get("old_616_J7_physically_replayed_BEFORE_NRA3_hooks") is True
          and s.get("source_ONLY_no_S113_SYNTAGRUS_UD_GOLD_loaded_in_original_CPU") is True,
          "physical true original12 and old 53751/J7 provenance required")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    base=load_file(str(source),device="cpu")
    hidden=base["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    mask=base["native_BPE_source_valid_mask"]
    ids=base["original_native_BPE_token_ids"]
    original_source_valid(hidden,mask,ids)
    vocabulary,prototypes,counts=build_frozen_train_lexicon(hidden,mask,ids)
    all_training, train_coverage=collect_targets(mask,ids,vocabulary,TRAIN_FROM)
    all_probe, probe_coverage=collect_targets(mask,ids,vocabulary,PROBE_FROM)
    # No selection based on model performance. Fixed prefix and row order.
    train_examples=all_training[:TRAIN_EXAMPLES]
    probe_examples=all_probe[:PROBE_EXAMPLES]
    check(len(train_examples)==TRAIN_EXAMPLES
          and len(probe_examples)>=12
          and set(x[0] for x in train_examples).isdisjoint(
                  set(x[0] for x in probe_examples)),
          "precommitted TRAIN and TRAIN-corpus probe must have enough native BPE coverage")
    weights=load_file(str(prior),device="cpu")
    candidate=UnifiedCalibriWithStructuredInterface()
    candidate.load_exact_s119g(weights)
    local=copy.deepcopy(candidate)
    check(sum(x.numel() for x in candidate.parameters())
          ==sum(x.numel() for x in local.parameters()),
          "two ablations have EXACT same full neural weights")
    # Freeze base shared full 53,751 from S115; compare only one dimension
    # (learned context vs token-local), no extra parameters for contextual.
    uniprior=frozen_unigram_baseline(vocabulary,counts,train_examples)
    contextual_before=evaluate(candidate,hidden,mask,probe_examples,
                               prototypes,True,uniprior)
    local_before=evaluate(local,hidden,mask,probe_examples,
                          prototypes,False,uniprior)
    check(contextual_before==local_before,
          "zero-init new area must make ablations exact before training")
    t0=time.perf_counter()
    contextual_grads=train(candidate,hidden,mask,train_examples,
                           prototypes,contextual=True)
    ctx_train_s=time.perf_counter()-t0
    t0=time.perf_counter()
    local_grads=train(local,hidden,mask,train_examples,
                      prototypes,contextual=False)
    loc_train_s=time.perf_counter()-t0
    contextual_after=evaluate(candidate,hidden,mask,probe_examples,
                              prototypes,True,uniprior)
    local_after=evaluate(local,hidden,mask,probe_examples,
                         prototypes,False,uniprior)
    full_state={PARENT_PREFIX+k:v.detach().half().cpu().contiguous()
                for k,v in candidate.state_dict().items()}
    check(all(torch.equal(full_state[k],w)
              for k,w in weights.items()
              if ".nra3_neural_weights." in k)
          and sum(x.numel() for x in full_state.values())
              ==sum(x.numel() for x in local.state_dict().values()),
          "original learned 53751 EXACT and all same-sized comparators")
    out.mkdir(parents=True)
    f=out/CHILD
    save_file(full_state,str(f))
    reread=load_file(str(f),device="cpu")
    check(set(reread)==set(full_state)
          and all(torch.equal(reread[k],v) for k,v in full_state.items()),
          "physical unified candidate FP16 roundtrip exact")
    proto_file=out/VOCAB
    save_file({"native_original_BPE_class_ids_from_TRAIN_only":
                  vocabulary.to(torch.int32).contiguous(),
               "frozen_original_CB59_GT_layer0_train_prototype":prototypes.half().contiguous(),
               "TRAIN_source_frequency_by_original_token_id":counts.to(torch.int32).contiguous()},
              str(proto_file))
    report={
        "stage":STAGE,
        "source_original_S114_all12_SHA256":PARENT_H_SHA,
        "inherited_full_S119g_fp16_model_SHA256":sha(prior),
        "registered_unified_child_original_trained_NRA3_53751_exact":True,
        "native_original_BPE_target_supervision_NOT_hidden_proxy":True,
        "native_BPE_output_is_RESTRICTED_TRAIN_lexicon_NOT_true_full_decoder":True,
        "vocabulary_constructed_from_only_FIRST128_TRAIN_A_rows_no_probe_used":True,
        "native_BPE_train_prototype_dictionary_size":int(vocabulary.numel()),
        "precommitted_SOURCE_TRAIN_rows_for_prototype":"0..127",
        "precommitted_SOURCE_TRAIN_rows_for_actual_gradient":"0..63",
        "precommitted_SOURCE_TRAIN_rows_for_probe_only":"128..159",
        "token_class_coverage_on_all_TRAIN_probe_positions":probe_coverage,
        "training_class_coverage":train_coverage,
        "number_training_pairs":len(train_examples),
        "number_probe_pairs_conditioned_on_TRAIN_vocab":len(probe_examples),
        "matched_parameter_count_each_child":
            sum(x.numel() for x in full_state.values()),
        "matched_identical_optimizer_updates":EPOCHS*len(train_examples),
        "untrained_contextual_and_local_equal":contextual_before==local_before,
        "pretraining_restricted_vocab_source_probe":contextual_before,
        "after_contextual_restricted_vocab_source_probe":contextual_after,
        "after_equal_parameter_local_restricted_vocab_source_probe":local_after,
        "contextual_MINUS_local_native_BPE_CE":
            contextual_after["restrict_vocab_next_native_BPE_CE"]-
            local_after["restrict_vocab_next_native_BPE_CE"],
        "contextual_WON_narrow_source_native_token_probe":
            contextual_after["restrict_vocab_next_native_BPE_CE"]<
            local_after["restrict_vocab_next_native_BPE_CE"],
        "loss_is_native_BPE_ID_with_genuine_original12_frozen_source":True,
        "two_R_NRA_NPA_core_gradient_traces":
            {"contextual":contextual_grads,"token_local":local_grads},
        "contextual_train_CPU_seconds_not_real_decoder_latency":ctx_train_s,
        "local_train_CPU_seconds_not_real_decoder_latency":loc_train_s,
        "physical_full_new_neural_child_SHA256":sha(f),
        "physical_full_new_neural_child_bytes":f.stat().st_size,
        "physical_frozen_TRAIN_only_lexicon_sha256":sha(proto_file),
        "original_decoder_REAL_CB59_GT_new_forward_NOT_RERUN":True,
        "actual_original_full_vocabulary_head_NOT_USED":True,
        "true_prefix_only_original_GT_causality_NOT_INDEPENDENTLY_RERUN":True,
        "real_free_generated_universal_answers_NOT_TESTED":True,
        "no_new_616_J7_exact_candidate_replay":True,
        "no_measured_matched_whole_model_CPU_throughput":True,
        "no_loader_complete_all_in_BPP":True,
        "SynTagRus_B_GOLD_NEVER_OPENED":True,
        "test_rows_are_same_TRAIN_corpus_not_independent_generalization":True,
        "license_CC_BY_NC_SA_4_0_noncommercial_only":True,
        "output_gate":0,"activation_allowed":False,
        "candidate_accepted":False,
        "decision":"ACTUAL_NATIVE_BPE_RESTRICTED_VOCAB_SOURCE_ONLY_CONTEXTUAL_ABLATION",
    }
    (out/REPORT).write_text(json.dumps(report,sort_keys=True,indent=2,
                                     ensure_ascii=False)+"\n",encoding="utf8")
    print("S119J_REAL_NATIVE_NEXT_BPE_RESTRICTED_LEXICON "+
          json.dumps({
            "candidate_native_BPE_classes":int(vocabulary.numel()),
            "probe_coverage":probe_coverage["known_token_coverage"],
            "probe_examples":len(probe_examples),
            "contextual_before_CE":
                contextual_before["restrict_vocab_next_native_BPE_CE"],
            "contextual_after_CE":
                contextual_after["restrict_vocab_next_native_BPE_CE"],
            "token_local_after_CE":
                local_after["restrict_vocab_next_native_BPE_CE"],
            "contextual_outperformed_local":report[
                "contextual_WON_narrow_source_native_token_probe"],
            "genuine_model_autoregressive_generation":False,
            "original_53751_FP16_exact":True,
            "research_only_accepted":False},sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for n in ("prior","prior_report","source","source_report","out"):
        p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.prior,a.prior_report,a.source,a.source_report,a.out)


if __name__=="__main__":
    main()
