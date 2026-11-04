"""Freeze source FIT and previously untouched TRAIN-split text VALID.

A rank-one context-conditioned decoder-head correction consumes 4 bits for
each of the 49152 output tokens, 576 FP16 input projection coefficients,
a FP16 intercept and FP32 scale. No Green or uncounted sidecar is allowed.
"""
from v147.protocol import WIKI_REVISION,NEWS_REVISION
from v117.protocol import TOTAL_UNIQUE,MAX_SOURCE_BYTES

CB147_SHA="639be185c259260c614823831275113ffa2d83231eb54805ebfc67be30da5bbd"
FIT_DOMAINS=("wiki_fit","news_fit")
VALID_DOMAINS=("wiki_conditional_valid","news_conditional_valid")
WINDOW=64
FIT_WINDOWS_EACH=4
VALID_WINDOWS_EACH=8
TRAIN_HIDDEN=576
VOCAB=49152
RANK=1
POWER_ITERS=8
RIDGE_FRACTION=0.2
OUTLIER_QUANTILE=0.997
WIKI_VALID_START=30720
NEWS_VALID_START=10300
NEWS_VALID_END=10400
VALID_TOKENS=WINDOW*VALID_WINDOWS_EACH
ORIGINAL_PARAMS=TOTAL_UNIQUE

assert WIKI_VALID_START >= 26624+512
assert NEWS_VALID_START>=9600
assert VALID_TOKENS==512
assert len(CB147_SHA)==64
