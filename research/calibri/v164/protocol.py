"""Preregister 16 causal paid original operator arms, old TRAIN FIT only.

v162 physically paid block0 passed WikiText per-domain robustness but AG
News FAILED 32-window predeclared 20/32 and leave-best-out gate. Before
another expensive source rate allocation, measure which actual original
first-block q,k,v,o,gate,up,down operators help or harm, including
interactions, without opening any new VALID/TEST or touching weights.
All 16 arms reference SAME frozen original full 11,292,174-byte
0.6715785-bpp CB161 archive, including unused paid source patches.
Diagnostic RAM toggles are not separately packed candidate releases.
"""
from v161.protocol import MATRIX_KEYS
from v117.protocol import ORIGINAL_CB59_SHA
from v158.protocol import FIT_DOMAINS,WINDOW,ORIGINAL_TRAIN_FIT_WINDOWS_EACH
from v162.protocol import PARENT_MODEL_BYTES

PARENT_RUN_161=41920207168
PARENT_RUN_162=41921242142
LAYER=0
WINDOWS=ORIGINAL_TRAIN_FIT_WINDOWS_EACH
TOKENS_EACH=WINDOWS*WINDOW
SHORT=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
ARMS=("original_CB59","paid_original_all7")+tuple(
    f"paid_ONLY_{x}" for x in SHORT)+tuple(
    f"paid_EXCEPT_{x}" for x in SHORT)
EXPECTED_ARMS=16


def check():
    if len(MATRIX_KEYS)!=7 or (
            tuple(x.rsplit(".",2)[-2] for x in MATRIX_KEYS)!=SHORT):
        raise ValueError("original layer0 operator list or order changed")
    if ARMS!=("original_CB59","paid_original_all7")+tuple(
            f"paid_ONLY_{x}" for x in SHORT)+tuple(
            f"paid_EXCEPT_{x}" for x in SHORT):
        raise ValueError("source operator ablation changed after inspection")
    if (WINDOWS,WINDOW,TOKENS_EACH)!=(4,64,256):
        raise ValueError("only previously used 4x64 TRAIN FIT source is permitted")
    return True
