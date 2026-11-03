"""Immutable preregistered independent v86 TEST inputs and source bytes."""
from __future__ import annotations
import hashlib

import repro_qa_calibration_v78 as evaluation
from paid_lowrank_v76 import SOURCE_BYTES
from v85.physical import CB85

ORIGINAL_CB59_SHA="50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"
FROZEN_ADAPTIVE_CB85_SHA="2fe36504a0eaf0159859cff68ee1ac3269c55959912f3b0ea1b17bbe1d9c7f5d"
FROZEN_ADAPTIVE_CB85_BYTES=11240769

# Predeclared disjoint from v82 TEST and from v84/v85 TRAIN/VALID.
TEST_SCIQ=(320,344)
TEST_BOOLQ=(3140,3164)
TEST_WIKI_START=160000
TEST_NEWS=(7520,7600)
WINDOWS=8
WINDOW_SIZE=64
GREEDY_PROMPTS=(
 "The new method succeeded because",
 "A small village near the mountains had",
)
GENERATED_TOKENS_PER_PROMPT=16


def freeze_audit(original:bytes,adapted:bytes)->dict:
    if len(original)!=SOURCE_BYTES or hashlib.sha256(original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise AssertionError("v86 must use exact originally audited full CB59 source")
    if (len(adapted)!=FROZEN_ADAPTIVE_CB85_BYTES or
        hashlib.sha256(adapted).hexdigest()!=FROZEN_ADAPTIVE_CB85_SHA):
        raise AssertionError("v86 candidate differs from frozen PRE-TEST v85 source")
    source=CB85(adapted)
    if source.original.hash()!=ORIGINAL_CB59_SHA:
        raise AssertionError("v86 CB85 original fully packed source differs")
    if len(source.extra)!=3:
        raise AssertionError("v86 source lacks adaptive packed native operators")
    d=source.physical_stats()
    return {"physical_frozen_before_unseen_TEST":True,
            "pre_declared_independent_TEST_SciQ":TEST_SCIQ,
            "pre_declared_independent_TEST_BoolQ":TEST_BOOLQ,
            "pre_declared_independent_TEST_Wiki_start":TEST_WIKI_START,
            "pre_declared_independent_TEST_News_rows":TEST_NEWS,
            "original_unique_source_complete_unchanged":True,
            "new_candidate_architecture_source_complete":d,
            "no_train_or_VALID_reselection_after_TEST":True}


def activate_new_test_slices():
    evaluation.SCI_Q_ROWS=TEST_SCIQ
    evaluation.BOOLQ_ROWS=TEST_BOOLQ
    evaluation.WIKI_TEST_FIRST=TEST_WIKI_START
    evaluation.NEWS_TEST_ROW_RANGE=TEST_NEWS
    evaluation.WINDOW_COUNT=WINDOWS
    evaluation.WINDOW=WINDOW_SIZE
    evaluation.PROMPTS=GREEDY_PROMPTS
    evaluation.NEW_TOKENS_PER_PROMPT=GENERATED_TOKENS_PER_PROMPT
