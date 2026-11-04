"""Freeze v155 exploratory outcome BEFORE accessing third disjoint source VALID.

v155 predeclared "worst" operator based on only 8 reused v154 diagnostic
prompts. Contrary to that ranking, original paid layer6 v_proj lowered both
gold NLL and FP32 teacher KL on v155 fresh WikiText/AG News source ranges.
Promoting it on v155 is a model selection, so NOT validation: only entirely
new v156 TRAIN-source text can confirm. Original CB153 7 patch full physical
11,292,174-byte archive remains counted even if runtime enables one patch.
"""
from v154.protocol import MATRIX_KEYS,CB153_SHA
from v117.protocol import ORIGINAL_CB59_SHA
from v147.protocol import WIKI_REVISION,NEWS_REVISION

SELECTED_KEY="model.layers.6.self_attn.v_proj.weight"
SECONDARY_KEY="model.layers.6.self_attn.o_proj.weight"
ARMS=("native_CB59","paid_v_proj_only","paid_o_proj_only","paid_all7")
DOMAINS=("wiki_THIRD_unseen_VALID","news_THIRD_unseen_VALID")
WIKI_START=49152
NEWS_START=14300
NEWS_END=14400
WINDOW=64
WINDOWS=8
TOKENS=WINDOW*WINDOWS
PARENT_RUN=41913322415
TRAIN_SOURCE_ONLY=True


def check():
    if (SELECTED_KEY,SECONDARY_KEY)!=(MATRIX_KEYS[2],MATRIX_KEYS[3]):
        raise ValueError("original source operator selection was silently altered")
    if (WIKI_START,NEWS_START,NEWS_END,TOKENS)!=(49152,14300,14400,512):
        raise ValueError("predeclared new holdout changed after v155")
    if not (WIKI_START>=45056+512 and NEWS_START>=13300):
        raise ValueError("new source overlaps previous selection VALID")
    if ARMS!=("native_CB59","paid_v_proj_only","paid_o_proj_only","paid_all7"):
        raise ValueError("predeclared comparator missing")
    return True
