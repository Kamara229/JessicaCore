"""A precommitted 4x larger, two-domain, contamination-free robustness gate.

v156 mean NLL/KL for paid original layer6 v_proj on its third disjoint
source window improved, but WikiText improved on ONLY 3 of 8 windows;
one unusually large negative NLL delta was influential. Therefore v156
does not imply a robust result. This test is planned BEFORE opening a
NEW fourth source TRAIN slice; no retuning weights or bitstream.
"""
from v154.protocol import CB153_SHA
from v117.protocol import ORIGINAL_CB59_SHA
from v147.protocol import WIKI_REVISION,NEWS_REVISION

LAST_ACTION=41914405749
SELECTED="model.layers.6.self_attn.v_proj.weight"
CONTROL="model.layers.6.self_attn.o_proj.weight"
ARMS=("native_CB59","paid_layer6_v_only","paid_layer6_o_only","paid_layer6_all7")
DOMAINS=("wiki_fourth_robustness_VALID","news_fourth_robustness_VALID")
WINDOW=64
WINDOWS=32
TOKENS=WINDOW*WINDOWS
SCORED_TOKENS_PER_DOMAIN=WINDOWS*(WINDOW-1)
WIKI_START=57344
NEWS_START=17000
NEWS_END=17400
MIN_IMPROVED_WINDOWS=20
REQUIRE_LEAVE_LARGEST_GAIN_OUT=True

def check():
    if (WINDOW,WINDOWS,TOKENS,SCORED_TOKENS_PER_DOMAIN)!=(64,32,2048,2016):
        raise ValueError("new v157 source sample size changed")
    if (WIKI_START,NEWS_START,NEWS_END)!=(57344,17000,17400):
        raise ValueError("new fourth disjoint source changed")
    if not (WIKI_START>=49152+512 and NEWS_START>=14400):
        raise ValueError("overlap with the source used for v156 model selection")
    if (MIN_IMPROVED_WINDOWS,REQUIRE_LEAVE_LARGEST_GAIN_OUT)!=(20,True):
        raise ValueError("robust preregistered check changed")
    return True
