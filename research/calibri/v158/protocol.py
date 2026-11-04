"""Predeclare ORIGINAL TRAIN FIT-only operator error decomposition.

No source VALID after v156/v157 is touched: exactly the already-used
original v76 WikiText/AG News TRAIN FIT (4x64 windows/domain).
This is NOT quality evaluation or new candidate selection.

The paired original FP32 / native CB59 inputs help separate two factors:
  (a) original-source weight quantization error using SAME inputs;
  (b) upstream hidden-state drift into source layer6 v_proj;
  (c) combined error and effect of physically paid 5+1bit residual.
"""
from v154.protocol import CB153_SHA
from v117.protocol import ORIGINAL_CB59_SHA
from v147.protocol import WIKI_REVISION,NEWS_REVISION
LAYER=6
OPERATOR="model.layers.6.self_attn.v_proj"
KEY=OPERATOR+".weight"
FIT_DOMAINS=("wiki_fit","news_fit")
ORIGINAL_TRAIN_FIT_WINDOWS_EACH=4
WINDOW=64
TOKENS_EACH=ORIGINAL_TRAIN_FIT_WINDOWS_EACH*WINDOW
SOURCE_ARMS=("FP32_teacher","native_CB59","paid_onebit_vproj")
DIAGNOSTIC_ONLY=True


def check():
    if (LAYER,OPERATOR)!=(6,"model.layers.6.self_attn.v_proj"):
        raise ValueError("source operator changed after seeing v156/v157")
    if (FIT_DOMAINS,ORIGINAL_TRAIN_FIT_WINDOWS_EACH,WINDOW)!=(
            ("wiki_fit","news_fit"),4,64):
        raise ValueError("only historical original source FIT is allowed")
    if len(SOURCE_ARMS)!=3:return False
    return True
