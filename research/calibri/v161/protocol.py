"""Precommit layer-0 original-weight repair from completed v159/v160 facts.

v159 shows the largest early teacher-relative activation error jump at
original block 0 on BOTH source FIT corpora, ~+0.23. v160 then finds that
physically paying 1 more bit for the INPUT embedding improves local
embeddings but worsens true NLL/KL on fresh WikiText and AG News source.
Instead change original seven 2D source operator approximations in the
first transformer layer, while keeping all tied input/output embedding
weights and the other 29 blocks ORIGINAL packed CB59.
"""
from v159.protocol import CB153_SHA
from v141.run import ORIGINAL_V140_SHA
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v147.protocol import WIKI_REVISION,NEWS_REVISION

LAYER=0
MATRIX_KEYS=tuple(f"model.layers.0.{name}.weight" for name in (
    "self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
    "mlp.gate_proj","mlp.up_proj","mlp.down_proj"))
ARMS=("original_CB59_native","physically_paid_block0_onebit",
      "UNPAID_FP32_block0_oracle")
PHYSICAL_ARMS=ARMS[:2]
ORACLE_NAME=ARMS[2]
DOMAINS=("wiki_block0_new_valid","news_block0_new_valid")
WINDOW=64
WINDOWS=8
TOKENS=WINDOW*WINDOWS
WIKI_START=73728
NEWS_START=22000
NEWS_END=22200
SOURCE_ONLY_FIT=True
PREVIOUS_V160_RUN_ID=41919065068
PREVIOUS_V159_RUN_ID=41917353381


def check():
    if (LAYER,WIKI_START,NEWS_START,NEWS_END,TOKENS)!=(0,73728,22000,22200,512):
        raise ValueError("predeclared block0 source validation changed")
    if WIKI_START<65536+512 or NEWS_START<19200:
        raise ValueError("source VALID overlaps original v160")
    if len(MATRIX_KEYS)!=7 or any(not s.startswith("model.layers.0.") for s in MATRIX_KEYS):
        raise ValueError("not the seven actual original first block operators")
    if ARMS!=(ARMS[0],ARMS[1],"UNPAID_FP32_block0_oracle"):
        raise ValueError("a comparison arm was changed")
    return True
