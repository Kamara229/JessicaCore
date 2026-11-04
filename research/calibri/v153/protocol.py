"""Fixed follow-up to v152 all-30 FP32 oracle ranking, no official TEST.

Choose exactly one original transformer block by pre-existing v152
diagnostic ranking. Never optimize based on new v153 source VALID data.
The source-only bit refinement for EACH 2D operator in that block is
paid in the complete original CB59 physical model (not CB149/CB151).
"""
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v147.protocol import WIKI_REVISION,NEWS_REVISION
import json
from pathlib import Path

FROZEN_V152_ACTION=41910659729
PARENT_CB147_SHA="639be185c259260c614823831275113ffa2d83231eb54805ebfc67be30da5bbd"
PARENT_CB59_SHA=ORIGINAL_CB59_SHA
MODEL_LAYERS=30
VECTOR=8
CODEBOOK_K=32
RESIDUAL_K=2
FIT_GROUP_SAMPLE_MAX=8192
FIT_STEPS=8
NEXT_VALID=("wiki_paid_block_valid","news_paid_block_valid")
WIKI_NEXT_START=40960
NEWS_NEXT_START=12200
NEWS_NEXT_END=12300
NEXT_WINDOWS=8
WINDOW=64
NEXT_TOKENS=NEXT_WINDOWS*WINDOW
EXPECTED_BLOCK_MATRICES=7

def select_layer(path):
    report=json.loads(Path(path).read_text())
    if report.get("stage")!="v152_NONDEPLOYABLE_FP32_30BLOCK_CAUSAL_LOCALIZATION":
        raise ValueError("must use v152 full causal oracle report, not guesses")
    if report.get("physically_packed_CB59_sha256")!=PARENT_CB59_SHA:
        raise ValueError("v152 source must equal CB59")
    ranking=report["summary"]["all_30_single_layer_FP32_oracle_ablation_ranking"]
    if sorted(x["layer"] for x in ranking)!=list(range(30)):
        raise ValueError("v152 must run all 30 original layer oracles")
    winner=ranking[0]
    if winner!=min(ranking,key=lambda r:r["mean_KL_delta_vs_original_CB59"]):
        raise ValueError("v152 sensitivity ranking changed")
    return int(winner["layer"]),{
        "selected_layer_by_v152_diagnostic_only":int(winner["layer"]),
        "v152_unpaid_FP32_KL_difference":float(winner["mean_KL_delta_vs_original_CB59"]),
        "v152_unpaid_FP32_oracle_was_helpful":
            winner["mean_KL_delta_vs_original_CB59"]<0,
        "not_a_paid_result_or_valid_metric":True,
    }

def check():
    if not (WIKI_NEXT_START>=35840 and NEWS_NEXT_START>=11500):
        raise ValueError("new heldout data overlaps prior experiments")
    if (NEXT_WINDOWS,WINDOW,NEXT_TOKENS)!=(8,64,512):
        raise ValueError("registered VALID protocol changed")
    if (CODEBOOK_K,RESIDUAL_K,VECTOR,FIT_STEPS)!=(32,2,8,8):
        raise ValueError("registered source-weight fit changed")
    return True
