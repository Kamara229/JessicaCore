"""Predeclared Green Tunnel feasibility with zero new learned weights.

v163 measured actual very repetitive low-diversity full generations:
CB59 distinct-token ratio 0.0625 and paid block0 only 0.0859375 on
four 32-token prompts. This is NOT semantic recovery. Before building
a separate trainable Green Tunnel, ask if a *reranker restricted to
the existing packed student top-K vocabulary* could even recover the
original teacher next-token top-1 decision. The upper bound is exact,
optimistic tie-breaking, and REQUIRES an oracle knowing the teacher;
it is never an actual deployable module or model-quality gain.

Only original historical v76 wiki_fit and news_fit TRAIN FIT token
windows, 4x64 each, and original FP32 teacher evaluation-only. No
SciQ/BoolQ, new VALID, frozen v89 TEST, training or compression changes.
"""
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v158.protocol import FIT_DOMAINS,WINDOW,ORIGINAL_TRAIN_FIT_WINDOWS_EACH
from v161.protocol import MATRIX_KEYS

TOPK=(1,8,32,128,512,2048)
ARMS=("original_native_CB59","fully_paid_layer0_CB161")
VOCAB=49152
WINDOWS=ORIGINAL_TRAIN_FIT_WINDOWS_EACH
TOKENS_PER_DOMAIN=WINDOWS*(WINDOW-1)
FULL_PARENT_ARCHIVE_BYTES=11292174
V163_ACTION=41922383305

def check():
    if (TOPK,ARMS,VOCAB)!=(
        (1,8,32,128,512,2048),
        ("original_native_CB59","fully_paid_layer0_CB161"),49152):
        raise ValueError("precommitted Green Tunnel support experiment changed")
    if (FIT_DOMAINS,WINDOW,WINDOWS,TOKENS_PER_DOMAIN)!=(
        ("wiki_fit","news_fit"),64,4,252):
        raise ValueError("not previous original TRAIN FIT contexts")
    if len(MATRIX_KEYS)!=7:
        raise ValueError("original 7 paid source operators changed")
    return True
