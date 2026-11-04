"""PREREGISTER fully independent 32x64 source robustness after v161.

v161 improved NLL/KL on 8 new source VALID windows per domain under the
actual physically paid 11,292,174-byte original CB59+layer0 5+1bit model.
However v161's News gold NLL margin was ONLY -0.00496, and teacher next
token top1 match remained 1/504 Wiki and 0/504 News. Those source
VALID checks cannot certify generalization or language quality.

No refit, source bit changes, oracle FP32 layer borrowing, or v89 TEST.
"""
from v117.protocol import ORIGINAL_CB59_SHA, TOTAL_UNIQUE, MAX_SOURCE_BYTES
from v147.protocol import WIKI_REVISION, NEWS_REVISION

PARENT_V161_RUN_ID=41920207168
PARENT_MODEL_BYTES=11292174
ORIGINAL_CODEBOOK_SHA=ORIGINAL_CB59_SHA
LAYER=0
NATIVE_ARMS=("CB59_native","CB161_paid_original_block0")
DOMAINS=("wikitext_fifth_new_block0","ag_news_fifth_new_block0")
WINDOW=64
WINDOWS=32
SCORED_TOKENS_EACH=WINDOWS*(WINDOW-1)
SOURCE_TOKENS=WINDOW*WINDOWS
WIKI_START=81920
NEWS_START=25000
NEWS_END=25500
MIN_BETTER_WINDOWS=20
DROP_SINGLE_MOST_FAVORABLE_WINDOW=True


def check():
    if (WIKI_START,NEWS_START,NEWS_END)!=(81920,25000,25500):
        raise ValueError("preregistered independent source range changed")
    if (WINDOWS,WINDOW,SCORED_TOKENS_EACH,SOURCE_TOKENS)!=(32,64,2016,2048):
        raise ValueError("missing complete 32 window validation")
    if WIKI_START<73728+512 or NEWS_START<22200:
        raise ValueError("overlap with v161 original TRAIN source VALID")
    if (MIN_BETTER_WINDOWS,DROP_SINGLE_MOST_FAVORABLE_WINDOW)!=(20,True):
        raise ValueError("no retrospective rate-distortion threshold selection")
    if NATIVE_ARMS!=("CB59_native","CB161_paid_original_block0") or len(DOMAINS)!=2:
        raise ValueError("missing genuine physical source model controls")
    return True
