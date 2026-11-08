"""S25 read-only diagnosis of the S24 FAILURE to bind new nouns.

Precommitted inputs: the S20 and S24 sealed paid 36M CB59 checkpoints;
never select new training checkpoints or adjust a loss with the 96 QA cases.
Both checkpoints have already been scored independently by S24; 0/96
complete+EOS in each, S24's new-noun lexical gold was 0/96 even when
85/96 were well-formed two-word responses.

This examines noun-token causal context on (a) historical S20 TRAIN,
(b) S24 same-color/opposing-noun TRAIN and (c) the existing frozen S24 QA.
Teacher-force the CORRECT first color; measure second (noun) gold rank,
EOS competition, and whether second-token top1 changes with noun target.
Also inspect EOS after forcing both gold words. A correct forced-prefix
token is a DIAGNOSTIC, not a correct generative answer.

No gradients, no codebook mutation, no KL/NLL objective, no QA fitting.
All future claims of heldout improvement require a NEW frozen suite
precommitted before the next distinct experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES, SOURCE_PARAMS, MODEL_SPEC
from calibri2.s20.multiword_curriculum import make_training_pairs
from calibri2.s24.noun_guarded_training import create_train_pairs, COLORS, NOUNS
from calibri2.s24.independent_binding_qa import cases, unseen
from v176.operators import clear
from v186.pairs import tokenize_one
from v174.run import save

START_S20="4b62bcec5e40c2b7b73d3be38e4c365d1cf9a62614420497490795534477e93d"
SEALED_S24="b21619b12ff6014b63129916857a525455960852c8f5a530de233acf10e8f900"


def datasets(tok):
    s20=make_training_pairs(tok)
    s24=create_train_pairs(tok)
    qa=[c for c in cases() if c["domain"]=="new_heldout_color_noun"]
    assert len(s20)==192 and len(s24["noun"])==320
    assert len(qa)==96 and len(unseen())==8
    assert all(c["gold"].split()[0] in COLORS and
               c["gold"].split()[1] in NOUNS for c in qa)
    # Samples are a fixed read-only TRAIN diagnosis, not a holdout or fitting set.
    historical=s20[:16]+s20[-16:]
    noun=s24["noun"][:16]+s24["noun"][-16:]
    rows=[]
    for domain,group in (("historical_S20_TRAIN",historical),
                         ("new_S24_noun_TRAIN",noun)):
        for pair in group:
            for side in ("left","right"):
                p=pair[side]
                rows.append({"domain":domain, "question":p["question"],
                             "gold":p["answer"],"name":f'{pair["pair_id"]}_{side}'})
    for c in qa:
        rows.append({"domain":"existing_S24_frozen_new_96_QA",
                     "question":c["question"],"gold":c["gold"],
                     "name":c["id"]})
    assert len(rows)==224 and len({x["question"] for x in rows})==224
    # No data is used for weight updates, stopping, selecting hyperparameters
    # or modifying an inference adapter. Subsequent VALID must be new.
    return rows


def score_model(tok, cb59, rows, label):
    blob=cb59.read_bytes()
    assert len(blob)==SOURCE_BYTES
    wanted=START_S20 if label=="S20" else SEALED_S24
    sha=hashlib.sha256(blob).hexdigest()
    if sha!=wanted: raise AssertionError(f"{label} archive SHA changed: {sha}")
    archive=CB59(blob)
    assert archive.meta["original_unique_parameter_count"]==SOURCE_PARAMS
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    model.eval()
    results=[]
    with torch.inference_mode():
        for i,case in enumerate(rows):
            gold=case["gold"].split()
            if len(gold)!=2:raise AssertionError("S25 requires exact two-word gold")
            ncolor=len(tok.encode(gold[0],add_special_tokens=False))
            row=tokenize_one(tok,case["question"],case["gold"])
            target=row["targets"]
            if ncolor<1 or ncolor>=len(target)-1:
                raise AssertionError("Unable to isolate second word before final EOS")
            full=tok.encode(case["gold"],add_special_tokens=False)
            if full[:ncolor]!=tok.encode(gold[0],add_special_tokens=False):
                raise AssertionError("Tokenizer gold color prefix is not prefix of two-word answer")
            logits=model(input_ids=row["ids"][None],use_cache=False).logits[0][row["positions"]]
            first=logits[0]
            noun=logits[ncolor]
            eos=logits[-1]
            noun_target=int(target[ncolor])
            color_target=int(target[0])
            eos_target=int(target[-1])
            assert eos_target==tok.eos_token_id
            # Explicitly teacher-forcing correct first COLOR token(s):
            # token logits after gold first word, not after greedy output.
            gold_rank=int((noun>noun[noun_target]).sum())+1
            top_noun=int(noun.argmax())
            result={
                "name":case["name"],"domain":case["domain"],
                "gold":case["gold"],
                "forced_context":"entire correct first color (teacher-forced)",
                "gold_second_word_first_token_id":noun_target,
                "gold_second_word_first_token_rank":gold_rank,
                "gold_second_word_top1":top_noun==noun_target,
                "gold_second_word_top5":gold_rank<=5,
                "second_word_top1_token_id":top_noun,
                "second_word_top1_printable":repr(tok.decode([top_noun])),
                "premature_EOS_top1_after_gold_color":top_noun==tok.eos_token_id,
                "gold_first_color_token_top1":int(first.argmax())==color_target,
                "gold_EOS_top1_after_all_gold_words":
                    int(eos.argmax())==eos_target,
                "second_word_gold_minus_EOS_logit":
                    round(float(noun[noun_target]-noun[tok.eos_token_id]),4),
                "second_word_top1_token_logit":
                    round(float(noun[top_noun]),4)
            }
            results.append(result)
            if (i+1)%56==0:
                print(json.dumps({"event":"S25_READONLY_PROGRESS",
                    "label":label,"checked":i+1,
                    "noun_top1":sum(v["gold_second_word_top1"] for v in results)}),
                    flush=True)
    clear(sources)
    del model,sources
    domains={}
    for domain in sorted({x["domain"] for x in results}):
        selected=[x for x in results if x["domain"]==domain]
        first=Counter(x["second_word_top1_printable"] for x in selected)
        domains[domain]={
            "n":len(selected),
            "first_gold_top1":sum(x["gold_first_color_token_top1"] for x in selected),
            "noun_gold_top1_after_forced_correct_color":
               sum(x["gold_second_word_top1"] for x in selected),
            "noun_gold_top5_after_forced_correct_color":
               sum(x["gold_second_word_top5"] for x in selected),
            "eos_top1_after_forced_correct_color":
               sum(x["premature_EOS_top1_after_gold_color"] for x in selected),
            "eos_top1_after_forced_full_two_words":
               sum(x["gold_EOS_top1_after_all_gold_words"] for x in selected),
            "unique_second_word_top1_token_count":len(first),
            "most_common_second_word_top1_tokens":first.most_common(12),
            "gold_noun_token_rank_min":min(x["gold_second_word_first_token_rank"] for x in selected),
            "gold_noun_token_rank_max":max(x["gold_second_word_first_token_rank"] for x in selected),
            "gold_noun_token_rank_median":sorted(x["gold_second_word_first_token_rank"]
               for x in selected)[len(selected)//2]
        }
    return {"checkpoint":label,"sha256":sha,"phys_bpp":8*len(blob)/SOURCE_PARAMS,
            "per_domain":domains, "every_forced_prefix_outcome":results}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("s20",type=Path)
    p.add_argument("s24",type=Path)
    p.add_argument("out",type=Path)
    opts=p.parse_args()
    opts.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    rows=datasets(tok)
    data=[score_model(tok,opts.s20,rows,"S20"),
          score_model(tok,opts.s24,rows,"S24")]
    comparison={
        "stage":"S25_READONLY_NOUN_SECOND_TOKEN_CAUSAL_DIAGNOSTIC",
        "question_count":len(rows),
        "new_qa_already_seen_before_this_diagnostic":True,
        "existing_QA_never_used_for_training_or_checkpoint_selection":True,
        "no_new_generalization_claim_from_this_qa":True,
        "no_weights_modified":True,
        "no_new_runtime_inference_adapter":True,
        "new_independent_QA_required_before_any_next_training":True,
        "compare_same_gold_correct_color_forced":True,
        "results":data
    }
    save(opts.out/"S25_ACTUAL_PAID_36M_NOUN_FORCED_CONTEXT_DIAGNOSTIC.json",comparison)
    print(json.dumps({"stage":comparison["stage"],
         "n":len(rows),
         "models":[{"label":a["checkpoint"],
                    "sha256":a["sha256"],
                    "physical_bpp":a["phys_bpp"],
                    "per_domain":a["per_domain"]} for a in data],
         "no_training":True,"future_new_QA_required":True}),flush=True)


if __name__=="__main__":
    main()
