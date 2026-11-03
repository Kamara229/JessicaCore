"""Pre-test frozen physical sources and disjoint official unseen original TEST."""
from __future__ import annotations
import hashlib
import repro_qa_calibration_v78 as evaluation
from v86.provenance import ORIGINAL_CB59_SHA, FROZEN_ADAPTIVE_CB85_SHA
from v88.physical import unpack, audit, COMPLETE_BYTES

CB88_FROZEN_PRETEST_SHA="d0f547e651c9a3b411fa7b19c0181839ab813c69e9092c77dca2a47d8e5d85ec"
CB88_BYTES=11413683

SCI_Q=(560,608)
BOOL_Q=(3210,3258)
WIKI_START=185000
NEWS_ROWS=(6900,7020)
N_WINDOWS=16
CONTEXT=64
GENERATE=24
PROMPTS=(
    "The chemist checked the measurements and concluded",
    "During the expedition, the old map revealed",
    "A useful method for solving this problem is",
    "The farmer brought the machine into the workshop",
)

def freeze_before_test(original,cb85,cb88):
    if (hashlib.sha256(original).hexdigest()!=ORIGINAL_CB59_SHA or
        hashlib.sha256(cb85).hexdigest()!=FROZEN_ADAPTIVE_CB85_SHA or
        hashlib.sha256(cb88).hexdigest()!=CB88_FROZEN_PRETEST_SHA or
        len(cb88)!=CB88_BYTES or len(cb88)!=COMPLETE_BYTES):
        raise AssertionError("pretested source bytes changed or physically incomplete")
    embedded,*_=unpack(cb88)
    if embedded != original:
        raise AssertionError("CB88 does not embed full native CB59")
    counted=audit(cb88)
    if counted["all_in_bits_per_original_parameter"]>0.7:
        raise AssertionError("model exceeds physically paid bits per parameter")
    return {"frozen_CB59_SHA":ORIGINAL_CB59_SHA,
            "frozen_CB85_SHA":FROZEN_ADAPTIVE_CB85_SHA,
            "frozen_CB88_SHA":CB88_FROZEN_PRETEST_SHA,
            "CB88_full_original_pretrained_native_plus_tensor":counted,
            "original_official_new_SciQ_TEST_range":SCI_Q,
            "original_new_BoolQ_VALID_range":BOOL_Q,
            "new_WikiText_TEST_token_start":WIKI_START,
            "new_AGNews_TEST_row_range":NEWS_ROWS,
            "test_windows":N_WINDOWS,
            "test_window_tokens":CONTEXT,
            "greedy_tokens_per_prompt":GENERATE,
            "all_model_sources_frozen_before_first_official_TEST":True}

def activate_unseen_test():
    evaluation.SCI_Q_ROWS=SCI_Q
    evaluation.BOOLQ_ROWS=BOOL_Q
    evaluation.WIKI_TEST_FIRST=WIKI_START
    evaluation.NEWS_TEST_ROW_RANGE=NEWS_ROWS
    evaluation.WINDOW_COUNT=N_WINDOWS
    evaluation.WINDOW=CONTEXT
    evaluation.PROMPTS=PROMPTS
    evaluation.NEW_TOKENS_PER_PROMPT=GENERATE
