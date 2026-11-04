"""Frozen causal Green Tunnel placement after measured output top-K ceiling.

v165 measured the frozen physically paid 0.6715785-bpp original CB161:
an UNPAID PERFECT teacher top-1 oracle confined to the model's top 128
could recover at most 11.5% Wiki and 9.5% AG News teacher decisions
on the already consumed 4x64 TRAIN FIT. Even top 2048 could recover only
60.7% Wiki / 37.3% AG News. Hence a small topK-only reranker cannot
fully restore original teacher decisions.

Instead test the *counterfactual* upper bounds from replacing ONE
originally compressed hidden stream activation at one of five PRECOMMITTED
boundaries with the true original FP32 teacher tensor for that SAME
sample. This illegally relies on teacher hidden activations and so is
a diagnostic ONLY; a real Green Tunnel must independently infer such
state from packed-model inputs using its own trained, PHYSICALLY PAID
weights, without calling FP32 teacher at runtime.

Use historical already-consumed v76 wiki_fit/news_fit ONLY, no new VALID,
new SciQ/BoolQ or official frozen v89 TEST. No source compression changes.
"""
from v158.protocol import FIT_DOMAINS,WINDOW,ORIGINAL_TRAIN_FIT_WINDOWS_EACH
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v161.protocol import MATRIX_KEYS

PAID_V161_BYTES=11292174
PRIOR_V165_ACTION=41924549705
TOPK_FUTILE_CEILING=128
WINDOWS=ORIGINAL_TRAIN_FIT_WINDOWS_EACH
TOKENS_PER_DOMAIN=WINDOWS*(WINDOW-1)
TEACHER_ONLY_STAGES=(
    "input_embedding",
    "block_00_output",
    "block_09_output",
    "block_28_output",
    "final_norm",
)
ARM_NAMES=("original_CB59","paid_original_block0")
SOURCE_WINDOW_LENGTH=WINDOW
EXPECTED_HIDDEN=576
VOCAB_SIZE=49152
NO_UNPAID_SOURCE_COUNTS_AS_COMPACT=True

def check():
    if (WINDOWS,WINDOW,TOKENS_PER_DOMAIN)!=(4,64,252):
        raise ValueError("changed existing original TRAIN FIT source")
    if (FIT_DOMAINS,TEACHER_ONLY_STAGES)!=(("wiki_fit","news_fit"),
        ("input_embedding","block_00_output","block_09_output",
         "block_28_output","final_norm")):
        raise ValueError("unregistered teacher-oracle boundary")
    if (EXPECTED_HIDDEN,VOCAB_SIZE,PAID_V161_BYTES)!=(576,49152,11292174):
        raise ValueError("source architecture or paid physical budget altered")
    if len(MATRIX_KEYS)!=7:
        raise ValueError("missing original paid transformer first-block operators")
    return True
