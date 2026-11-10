"""S122: deterministic, authored TRAIN-only counterfactual binding curriculum.

Derived ONLY from existing ORIGINAL S82 TRAIN frames, TRAIN names,
historical 24 tokenizer-compatible S82 TRAIN nouns and four original
S30 color tokens. No new QA/TEST queries and no use of the S118
already-inspected holdout as a source of training examples, choices or
comparisons.

Original S82 TRAIN generates same-color left/right labels. S122 adds
synthetic independent color and noun assignments on both owners. Each
pair is bidirectional (the assignment is swapped) and generated from
a strictly precommitted formula; correct two-token phrase + EOS is
checked using the exact original tokenizer. No QA or model output is
used to choose the training rows. No inference-time lookup tables.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Callable


PAIR_COUNT=64
SEED=1220122


def design_pairs(words, colors, names, frames):
    """Pure deterministic 64-pair design, with two INDEPENDENT colors.

    No random search, no change after scoring a model. This is TRAIN.
    """
    if (len(words)<24 or len(set(words))!=len(words)
            or len(colors)!=4 or len(set(colors))!=4
            or len(names)!=4 or len(set(names))!=4
            or len(frames)!=4 or len(set(frames))!=4):
        raise ValueError("Only the frozen S82 lexical/train-frame ABI supported")
    items=[]
    distinct_specs=set()
    for i in range(PAIR_COUNT):
        frame=i%4
        a,b=names[frame]
        color_a=colors[(i//4)%4]
        color_b=colors[((i//4)+(i//16)%3+1)%4]
        # At least two labels with different colors for every pair.
        if color_a==color_b:
            raise AssertionError("S122 training labels must have independent colors")
        noun_a=(i*5+1+(i//48)*3)%24
        noun_b=(i*7+13+(i//48)*5)%24
        if noun_b==noun_a:
            noun_b=(noun_b+9)%24
        label_a=f"{color_a} {words[noun_a]}"
        label_b=f"{color_b} {words[noun_b]}"
        if label_a==label_b:
            raise AssertionError("S122 cannot train identical owner labels")
        # Both counterfactual sides are materialized below. Thus an
        # unordered pair (A,B) on the SAME frame must be unique: (B,A)
        # would duplicate both TRAIN questions even when ordered specs
        # differ. Resolve lexical collisions deterministically without
        # viewing model outputs or any held-out QA.
        ident=(frame,tuple(sorted((label_a,label_b))))
        if ident in distinct_specs:
            for offset in range(1,24):
                candidate=f"{color_b} {words[(noun_b+offset)%24]}"
                proposed=(frame,tuple(sorted((label_a,candidate))))
                if candidate!=label_a and proposed not in distinct_specs:
                    label_b=candidate
                    ident=proposed
                    break
            else:
                raise AssertionError("No deterministic unique original-TRAIN pair available")
        distinct_specs.add(ident)
        items.append({
            "pair_id":122000+i,
            "family":"S122_causal_cross_binding_TRAIN",
            "a":a,"b":b,
            "frame":frames[frame],
            "left_label":label_a,
            "right_label":label_b,
            "left_color":color_a,
            "right_color":color_b,
        })
    if len(items)!=64 or len(distinct_specs)!=64:
        raise AssertionError("S122 expected exactly 64 TRAIN counterfactual pairs")
    if len({(x["left_color"],x["right_color"]) for x in items})!=12:
        raise AssertionError("S122 must exercise all ordered distinct color pairs")
    return items


def materialize_train_only(tok):
    """Materialize two sides of each pair, checking 456 original TRAIN SHA.

    Does not inspect or import any previous QA: uses only s82/s87
    already-authorized original TRAIN plus a deterministic data recipe.
    """
    from binary_geometry_v64 import MODEL,MODEL_SHA
    from calibri2.s30 import matched_dense_fp32_train as s30
    from calibri2.s82 import real_mps_generic_token_context_copy as s82
    from calibri2.s87 import guard_all_456_full_cpu as s87
    from v186.pairs import tokenize_one
    from calibri2.s94.physical_profile_bank import sha

    _original,words=s82.newly_authored_train(tok)
    if len(words)!=24:
        raise AssertionError("Frozen original S82 24-word TRAIN inventory changed")
    ordered_colors=tuple(sorted(s30.TRAIN_ELIGIBLE))
    specs=design_pairs(words,ordered_colors,s82.TRAIN_NAMES,s82.TRAIN_FRAMES)
    historical={
        sha(side["question"].encode())
        for pairs in s87.families(tok).values()
        for pair in pairs for side in (pair["left"],pair["right"])
    }
    if len(historical)!=456:
        raise AssertionError("Historical 456 original TRAIN identities changed")

    candidates=[]
    unique=set()
    for spec in specs:
        frame=spec["frame"]
        a,b=spec["a"],spec["b"]
        left,right=spec["left_label"],spec["right_label"]
        sample_left=tokenize_one(
            tok,frame.format(a=a,b=b,left=left,right=right),left)
        sample_right=tokenize_one(
            tok,frame.format(a=a,b=b,left=right,right=left),right)
        for side,sample in (("left",sample_left),("right",sample_right)):
            question=sample["question"]
            digest=sha(question.encode())
            if digest in unique or digest in historical:
                raise AssertionError("S122 TRAIN duplicate or collision with original 456")
            unique.add(digest)
            answer=sample["answer"]
            expected=tok.encode(answer,add_special_tokens=False)
            if (len(expected)!=2 or tok.eos_token_id in expected
                    or expected[0]!=sample["first_expected_id"]
                    or len(tok.encode(question,add_special_tokens=False))<8):
                raise AssertionError("S122 two-token TRAIN owner binding ABI violation")
            candidates.append({
                "id":f'S122/{spec["pair_id"]}/{side}',
                "pair_id":spec["pair_id"],
                "side":side,
                "question":question,
                "answer":answer,
                "expected":expected+[tok.eos_token_id],
                "source":"new TRAIN ONLY",
            })
    if len(unique)!=128 or len(candidates)!=128:
        raise AssertionError("S122 materialized TRAIN is not 128 new questions")
    return candidates,sha("\n".join(sorted(unique)).encode()),specs


def main():
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    from calibri2.s30 import matched_dense_fp32_train as s30
    s30.TOK=tok
    new,digest,specs=materialize_train_only(tok)
    by_colors=Counter(
        f'{x["left_color"]}/{x["right_color"]}' for x in specs)
    metadata={
        "stage":"S122_S82_ONLY_DETERMINISTIC_CROSS_COLOR_COUNTERFACTUAL_TRAIN",
        "seed":SEED,
        "existing_S82_original_words":24,
        "existing_S82_train_frames":4,
        "existing_S82_train_name_pairs":4,
        "new_TRAIN_counterfactual_pairs":len(specs),
        "new_TRAIN_questions":len(new),
        "new_TRAIN_question_hashset_sha256":digest,
        "historical_456_TRAIN_disjoint":True,
        "different_left_right_color_per_pair":True,
        "all_12_ordered_distinct_color_pairings":True,
        "by_color_pair":dict(sorted(by_colors.items())),
        "same_original_two_token_plus_EOS_target_ABI":True,
        "created_before_any_new_model_training":True,
        "not_QA_or_independent_test":True,
        "not_generated_or_selected_from_inspected_S118":True,
        "no_existing_or_new_QA_inputs_loaded":True,
        "source_unchanged_no_model_training_or_acceptance":True,
    }
    (a.out/"S122_TRAIN_ONLY_CROSS_BINDING_CURRICULUM_REPORT.json").write_text(
        json.dumps(metadata,indent=2,sort_keys=True))
    print(json.dumps({
        "event":"S122_TRAIN_ONLY_CURRICULUM_READY",
        "pairs":metadata["new_TRAIN_counterfactual_pairs"],
        "questions":metadata["new_TRAIN_questions"],
        "all_original_456_excluded":True,
        "distinct_ordered_color_pairings":len(by_colors),
        "train_only_not_QA":True,
    }),flush=True)


if __name__=="__main__":
    main()
