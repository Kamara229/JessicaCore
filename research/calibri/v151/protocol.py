"""Register immutable post-v150 experiment and brand-new TRAIN-only VALID."""
from v149.protocol import (WIKI_REVISION,NEWS_REVISION,TRAIN_HIDDEN,VOCAB,
    FIT_DOMAINS,WINDOW,FIT_WINDOWS_EACH,TOTAL_UNIQUE)
from v117.protocol import MAX_SOURCE_BYTES

PINNED_CB149_SHA="28d7a34f004a99f0162319e4333cd7db49746bd3ea5d9ff3a203c775119c2c22"
RANK2_PAID_BIT_WIDTH=3
RANK2_PAID_INT_MIN=-3
RANK2_PAID_INT_MAX=3
RANK2_POWER_ITERS=8
RANK2_RIDGE=0.2
RANK2_OUTLIER_QUANTILE=.997
NEW_VALID_DOMAINS=("wiki_second_rank_valid","news_second_rank_valid")
WIKI_NEW_START=35328
NEWS_NEW_START=11400
NEWS_NEW_END=11500
NEW_VALID_WINDOWS=8
NEW_VALID_TOKENS=512
FULL_SOURCE_SHA_CHECK=True

def check_protocol():
    if not (WIKI_NEW_START>30720+512 and NEWS_NEW_START>=10400):
        raise ValueError("v151 source VALID overlaps earlier confirmed data")
    if RANK2_PAID_BIT_WIDTH!=3 or RANK2_PAID_INT_MIN!=-3 or RANK2_PAID_INT_MAX!=3:
        raise ValueError("successive paid rank changed")
    if (RANK2_POWER_ITERS,RANK2_RIDGE,RANK2_OUTLIER_QUANTILE)!=(8,.2,.997):
        raise ValueError("posthoc rank2 fit change prohibited")
    if (NEW_VALID_WINDOWS,NEW_VALID_TOKENS)!=(8,512):
        raise ValueError("source holdout changed")
    if len(PINNED_CB149_SHA)!=64:return False
    return True
