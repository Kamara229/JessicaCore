"""Freeze input/output interventions BEFORE a fresh disjoint source check.

v159 found the original 5-bit tied embedding has teacher-relative RMSE
~0.72 before block0 and that block0/9/28 accumulate drift. This
experiment asks whether the *already physically paid* extra tied CB140
one-bit source reduces the ACTUAL input embedding and block0 divergence.
Do not modify base weights or add an unaccounted green adapter.

Four pre-registered gain arms, one immutable CB140 physical 0.6942bpp
archive for all: baseline input/head 0/0, input-only 1/0,
head-only 0/1, both 1/1. The gain choices are runtime controls, NOT
serialized changes; a future release must pay explicitly for gain tags.
Original independent v89 TEST never opened.
"""
from v141.run import ORIGINAL_V140_SHA
from v159.protocol import CB153_SHA,FIT_DOMAINS
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v147.protocol import WIKI_REVISION,NEWS_REVISION

ARMS=("native_CB59_00","paid_input_only_10","paid_head_only_01","paid_tied_full_11")
GAINS=((0.,0.),(1.,0.),(0.,1.),(1.,1.))
DOMAINS=("wiki_fresh_embedding_causal","news_fresh_embedding_causal")
WINDOW=64
WINDOWS=8
VALID_TOKENS=WINDOW*WINDOWS
WIKI_START=65536
NEWS_START=19000
NEWS_END=19200
ORIGINAL_PHY_CB140_ACTION=41898410897
OLD_DRIFT_ACTION=41917353381

def check():
    if (WIKI_START,NEWS_START,NEWS_END,VALID_TOKENS)!=(65536,19000,19200,512):
        raise ValueError("predeclared v160 new source changed")
    if not (WIKI_START>=57344+2048 and NEWS_START>=17400):
        raise ValueError("v160 overlaps v157 untouched source")
    if (len(ARMS),len(GAINS),len(DOMAINS))!=(4,4,2):
        raise ValueError("incomplete causal tied-input/head controls")
    if (ARMS,GAINS)!=(("native_CB59_00","paid_input_only_10",
                      "paid_head_only_01","paid_tied_full_11"),
                     ((0.,0.),(1.,0.),(0.,1.),(1.,1.))):
        raise ValueError("posthoc gain selection not allowed")
    return True
