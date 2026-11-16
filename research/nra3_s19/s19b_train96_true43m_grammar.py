"""S19b TRAIN-only REAL frozen Calibri43M+Green Tunnel word grammar training.

Inputs are real, physically SHA-verified S18b original 43M/GT CPU
post-layer8 320-D hidden and unannotated RAW source-text indicators.
Supervision is frozen S18 new96 TRAIN UPOS, UD arc/label and predicate flag.
NEVER read consumed independent EVAL32, EVAL128, S17c, S15c, S12c or S3CI.
Held rows are not optimizer inputs, nor checked for checkpoint selection.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from research.nra3_s0.neural_area import UPOS
from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    raw_surface_grammar, DIM as RAW_DIM,
)
from research.nra3_s18.real43m_train96_physical_source import (
    checked_real_train96, grouped_train96_split_contract, S18B_TRUE_CPU_RUN,
)
from research.nra3_s18.ru96_multiverb_train_source import (
    S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
)
from research.nra3_s19.paid_grammar_parser import (
    CompactWordGrammar, DEP_LABELS, INPUT_DIM, RANK,
    paid_fp16_bytes, load_paid_fp16_bytes, trainable_parameter_count,
)

SEED = 319016
STEPS = 120
FULL_STEPS = 120
LEARNING_RATE = 0.015
OPTIMIZER_WD = 0.001
EXPECTED_ROWS = 96


def require(cond, msg):
    if not bool(cond):
        raise ValueError("S19b true-43M-CPU TRAIN-only word grammar: " + msg)


@dataclass(frozen=True)
class WordGrammarTrainData:
    """Gold labels are optimizer-side only; never forwarded to model."""
    input_338: torch.Tensor      # [96, W, 338] ONLY real hidden and raw text bits
    valid: torch.Tensor          # [96, W] from plain text length
    upos_gold: torch.Tensor      # [96, W], TRAIN-only
    dep_gold: torch.Tensor       # [96, W], TRAIN-only
    head_gold: torch.Tensor      # [96, W], TRAIN-only 0..W
    pred_gold: torch.Tensor      # [96, W], TRAIN-only
    source_sha: str
    hidden_sha: str


def assemble_true43m_train96(samples, report) -> WordGrammarTrainData:
    require(len(samples) == EXPECTED_ROWS and
            report["S18_NEW96_TRAIN_COMPLETE_SOURCE_SHA256"]
                == S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256 and
            report["all_616_native_CPU_generated_token_digests_unchanged"] is True,
            "immutable 96 TRAIN original CPU source / J7 identity")
    sizes = [len(s.row.words) for s in samples]
    max_w = max(sizes)
    X = torch.zeros((EXPECTED_ROWS, max_w, INPUT_DIM), dtype=torch.float32)
    V = torch.zeros((EXPECTED_ROWS, max_w), dtype=torch.bool)
    U = torch.full((EXPECTED_ROWS, max_w), -1, dtype=torch.long)
    D = torch.full((EXPECTED_ROWS, max_w), -1, dtype=torch.long)
    H = torch.full((EXPECTED_ROWS, max_w), -1, dtype=torch.long)
    P = torch.full((EXPECTED_ROWS, max_w), -1, dtype=torch.long)
    for i, item in enumerate(samples):
        row = item.row
        raw = raw_surface_grammar(row.text)
        aligned = item.aligned.word_alignment
        n = len(row.words)
        require(n == len(raw) == len(aligned)
                and [(w.start, w.end) for w in row.words] ==
                    [(w.start, w.end) for w in raw],
                "all words must align to raw Unicode, no TRAIN spans as input")
        require(len(set(w.last_token for w in aligned)) == n
                and all(0 <= w.last_token < item.hidden320.shape[0]
                        for w in aligned),
                "genuine original wordpiece last positions")
        raw_bits = torch.tensor([x.features for x in raw], dtype=torch.float32)
        frozen = torch.stack([item.hidden320[w.last_token]
                              for w in aligned]).float().contiguous()
        require(frozen.shape == (n, 320) and raw_bits.shape == (n, RAW_DIM),
                "original physical 320D hidden or raw18 shape")
        X[i, :n] = torch.cat((frozen, raw_bits), dim=-1)
        V[i, :n] = True
        for j, word in enumerate(row.words):
            require(word.upos in UPOS and word.deprel in DEP_LABELS
                    and 0 <= word.head <= n and word.head != j+1,
                    "unrecognized frozen TRAIN-only UPOS or UD gold")
            U[i, j] = UPOS.index(word.upos)
            D[i, j] = DEP_LABELS.index(word.deprel)
            H[i, j] = word.head
            P[i, j] = int(word.upos == "VERB")
        require(sum(int(w.head == 0) for w in row.words) == 1,
                "each TRAIN96 UD structure needs one root")
    require(X.shape[-1] == INPUT_DIM == 338
            and len(UPOS) == 17 and len(DEP_LABELS) == 14
            and bool((U[V] >= 0).all()) and bool((D[V] >= 0).all())
            and bool((H[V] >= 0).all()) and bool(((P[V] == 0) | (P[V] == 1)).all())
            and int(P[V].sum()) >= 96 and bool(torch.isfinite(X).all()),
            "TRUE data input or human TRAIN gold tensor corrupt")
    require(bool((X[~V] == 0).all())
            and bool((U[~V] == -1).all())
            and bool((H[~V] == -1).all()),
            "padded gold or original inputs leaked")
    return WordGrammarTrainData(X,V,U,D,H,P,
        S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
        report["actual_fp16_feature_sha256"])


def subset(data: WordGrammarTrainData, rowids):
    ind = torch.tensor(tuple(rowids), dtype=torch.long)
    require(ind.ndim == 1 and len(ind) == len(set(ind.tolist()))
            and len(ind) >= 1 and bool(((ind >= 0) & (ind < 96)).all()),
            "TRAIN/dev row ids invalid")
    return WordGrammarTrainData(*(t.index_select(0,ind)
        for t in (data.input_338,data.valid,data.upos_gold,
                  data.dep_gold,data.head_gold,data.pred_gold)),
        data.source_sha,data.hidden_sha)


def word_loss(logits, batch: WordGrammarTrainData):
    mask = batch.valid
    require(logits.upos.shape[:2] == mask.shape and
            logits.heads.shape[:2] == mask.shape,
            "model shape invalid for TRAIN targets")
    u = F.cross_entropy(logits.upos[mask], batch.upos_gold[mask])
    d = F.cross_entropy(logits.dependency[mask], batch.dep_gold[mask])
    h = F.cross_entropy(logits.heads[mask], batch.head_gold[mask])
    p = F.binary_cross_entropy_with_logits(
        logits.predicate[mask], batch.pred_gold[mask].float(),
        pos_weight=torch.tensor(2.,dtype=torch.float32))
    return u + 0.7*d + 1.0*h + 0.4*p


def train_fixed(batch: WordGrammarTrainData, steps=STEPS):
    """No heldout input; fixed optimizer steps; no checkpoint selection."""
    torch.manual_seed(SEED)
    model = CompactWordGrammar().train()
    require(model.output_gate == 0 and model.candidate_accepted is False
            and steps == STEPS,
            "source-only TRAIN fixed schedule / output gate")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=OPTIMIZER_WD)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.input_338, batch.valid)
        loss = word_loss(logits, batch)
        require(bool(torch.isfinite(loss)), "nonfinite training loss")
        loss.backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()), "missing learned grammar gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    physical=paid_fp16_bytes(model)
    frozen=load_paid_fp16_bytes(physical)
    require(physical == paid_fp16_bytes(frozen)
            and not any(p.requires_grad for p in frozen.parameters()),
            "complete FP16 physical grammar weights not inference-only")
    return frozen,physical


def metrics(model: CompactWordGrammar,batch: WordGrammarTrainData):
    model.eval()
    with torch.inference_mode():
        out=model(batch.input_338,batch.valid)
        mask=batch.valid
        pred_upos=out.upos.argmax(-1)
        pred_dep=out.dependency.argmax(-1)
        pred_head=out.heads.argmax(-1)
        pred_verb=(out.predicate >= 0)
        gold_verb=(batch.pred_gold==1)
        tp=int((pred_verb & gold_verb & mask).sum())
        fp=int((pred_verb & ~gold_verb & mask).sum())
        fn=int((~pred_verb & gold_verb & mask).sum())
        correct_u=int(((pred_upos==batch.upos_gold)&mask).sum())
        correct_d=int(((pred_dep==batch.dep_gold)&mask).sum())
        correct_h=int(((pred_head==batch.head_gold)&mask).sum())
        correct_las=int((((pred_head==batch.head_gold)&
                          (pred_dep==batch.dep_gold))&mask).sum())
        n=int(mask.sum())
        return dict(tokens=n,upos_correct=correct_u,dependency_label_correct=correct_d,
                    head_correct=correct_h,las_correct=correct_las,
                    predicate_TP=tp,predicate_FP=fp,predicate_FN=fn,
                    predicate_P=tp/(tp+fp) if tp+fp else 0.,
                    predicate_R=tp/(tp+fn) if tp+fn else 0.,
                    predicate_F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                    upos_accuracy=correct_u/n,head_UAS=correct_h/n,
                    exact_head_and_deprel_LAS=correct_las/n)


def combine_results(records):
    keys=("tokens","upos_correct","dependency_label_correct",
          "head_correct","las_correct","predicate_TP","predicate_FP","predicate_FN")
    sums={key:sum(int(row[key]) for row in records) for key in keys}
    n=sums["tokens"]
    tp,fp,fn=(sums["predicate_"+c] for c in ("TP","FP","FN"))
    return {**sums,
            "upos_accuracy":sums["upos_correct"]/n,
            "dep_label_accuracy":sums["dependency_label_correct"]/n,
            "head_UAS":sums["head_correct"]/n,
            "exact_head_and_deprel_LAS":sums["las_correct"]/n,
            "predicate_micro_F1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--s18b-hidden",required=True,type=Path)
    parser.add_argument("--s18b-report",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    require(not args.out.exists(),"never overwrite original model artifact")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    original,provenance=checked_real_train96(args.s18b_hidden,args.s18b_report)
    data=assemble_true43m_train96(original,provenance)
    folds=grouped_train96_split_contract()
    require(len(folds)==12 and len(data.input_338)==EXPECTED_ROWS
            and S18B_TRUE_CPU_RUN==43910479658,
            "original frozen 12 split TRAIN-only schedule or true CPU provenance")
    records=[]
    for kind,ident,groups,train_ids,held_ids in folds:
        require(set(train_ids).isdisjoint(held_ids)
                and sorted([*train_ids,*held_ids])==list(range(96)),
                "train / internal heldout overlap")
        tr=subset(data,train_ids)
        held=subset(data,held_ids)
        trained,paid=train_fixed(tr)
        m=metrics(trained,held)
        records.append({
            "fold":ident,"kind":kind,"held_groups":list(groups),
            "train_rows":len(train_ids),"internal_dev_rows":len(held_ids),
            "FP16_full_weights_sha256":hashlib.sha256(paid).hexdigest(),
            **m,
        })
        print("TRAIN_INTERNAL_CV_ONLY", ident, "UPOS",m["upos_accuracy"],
              "UAS",m["head_UAS"],"LAS",m["exact_head_and_deprel_LAS"],
              "VERB_F1",m["predicate_F1"],flush=True)
    final,physical=train_fixed(data)
    full=metrics(final,data)
    args.out.mkdir(parents=True,exist_ok=False)
    fn="NRA3_S19B_TRUE43M_GT_TRAIN96_WORD_GRAMMAR_ONLY_FP16.safetensors"
    path=args.out/fn
    path.write_bytes(physical)
    require(path.read_bytes()==paid_fp16_bytes(load_paid_fp16_bytes(physical)),
            "physical final model must have identical FP16 checksum")
    count=trainable_parameter_count(final)
    report={
        "stage":"NRA3_S19B_TRUE_ORIGINAL43M_GT_TRAIN96_GROUPED12_WORD_GRAMMAR_RESEARCH",
        "input_source":"REAL_ORIGINAL43M_GT_LAYER8_320D_TRAIN_ONLY_PLUS_RAW18",
        "S18b_source_run":S18B_TRUE_CPU_RUN,
        "S18_original_96_gold_SHA":data.source_sha,
        "S18_real_original_43M_GT_hidden_FP16_SHA":data.hidden_sha,
        "original_616_J7_CPU_digest_unchanged_parent":True,
        "frozen_original_43M_GT_weights_never_updated":True,
        "raw_word_feature_DIM":RAW_DIM,
        "original_hidden_dimension":320,
        "train_only_source_rows":96,
        "train_only_UD_or_UPOS_NEVER_passed_as_model_feature":True,
        "upos_labels":list(UPOS),
        "dependency_relation_labels":list(DEP_LABELS),
        "full_FP16_parameter_count":count,
        "full_FP16_weight_payload_bytes":len(physical),
        "all_physical_weights_billed":True,
        "full_trained_FP16_weights_SHA256":hashlib.sha256(physical).hexdigest(),
        "full_trained_FP16_file":fn,
        "fixed_optimization_steps_per_fold":STEPS,
        "fixed_optimization_steps_full_train":FULL_STEPS,
        "fixed_12fold_TRAIN_internal_results":records,
        "TRAIN_internal_lexical_aggregate":combine_results([z for z in records if z["kind"]=="lexical"]),
        "TRAIN_internal_syntax_aggregate":combine_results([z for z in records if z["kind"]=="syntax"]),
        "TRAIN_only_full96_resubstitution_metrics":full,
        "no_independent_EVAL32_or_EVAL128_or_prior_spent_EVAL_loaded":True,
        "heldout_not_used_for_gradient_checkpoint_or_selection":True,
        "strict_word_online_Calibri43M_causality_NOT_PROVEN":True,
        "new_candidate_616_J7_EOS_NOT_CERTIFIED":True,
        "inclusive_bpp_le07_NOT_CERTIFIED":True,
        "full_CPU_speed_ge90pct_CB59_NOT_CERTIFIED":True,
        "NRA_event_graph_integration_NOT_TRAINED":True,
        "candidate_accepted":False,
        "activation_allowed":False,
        "output_gate":0,
        "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
    }
    (args.out/"NRA3_S19B_TRUE43M_TRAIN96_GROUPED12_GRAMMAR_FP16_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf-8")
    print("S19B_TRAIN_INTERNAL_CV_LEX",json.dumps(report["TRAIN_internal_lexical_aggregate"],
         ensure_ascii=False,sort_keys=True),flush=True)
    print("S19B_TRAIN_INTERNAL_CV_SYNTAX",json.dumps(report["TRAIN_internal_syntax_aggregate"],
         ensure_ascii=False,sort_keys=True),flush=True)
    print("S19B_SOURCE_ONLY_FULL_TRAIN_FINISHED_NOT_EVAL",report["full_trained_FP16_weights_SHA256"],
         flush=True)


if __name__=="__main__":
    main()
