"""Select best and worst paid original operator by REUSED diagnostics only.

The v154 eight diagnostic prompts are not new validation. This candidate is
not fitted on v155's new TRAIN-split WikiText/AG News source rows.
No bitstream changes; 0.67158-bpp complete CB153 including seven stored
patches is counted even when only one patch is enabled.
"""
import json
from pathlib import Path
from v154.protocol import MATRIX_KEYS,CB153_SHA
from v147.protocol import WIKI_REVISION,NEWS_REVISION
from v117.protocol import ORIGINAL_CB59_SHA

ARM_NAMES=("CB59_original","CB153_all_seven","v154_best_single","v154_worst_single")
DOMAINS=("wiki_new_single_op_valid","news_new_single_op_valid")
WINDOW=64
WINDOWS=8
TOKENS=WINDOW*WINDOWS
WIKI_START=45056
NEWS_START=13200
NEWS_END=13300
V154_PRIOR_RUN_ID=41912617369

def precommit_selection(path):
    result=json.loads(Path(path).read_text())
    if result.get("stage")!="v154_ORIGINAL_SOURCE_LAYER6_SEVEN_OPERATOR_ABLATIONS":
        raise ValueError("not original v154 diagnostic report")
    if result.get("fully_serialized_CB153_SHA256")!=CB153_SHA:
        raise ValueError("posthoc model source changed")
    if result.get("complete_original_CB59_SHA256")!=ORIGINAL_CB59_SHA:
        raise ValueError("not pinned pretrained 5bit codebook")
    table=result["diagnostic"]["per_original_operator_diagnostic"]
    if sorted(row["original_operator"] for row in table)!=sorted(MATRIX_KEYS):
        raise ValueError("not complete 7-operator diagnostic")
    best=min(table,key=lambda row:(row["single_operator_KL_vs_CB59"],row["original_operator"]))
    worst=max(table,key=lambda row:(row["single_operator_KL_vs_CB59"],row["original_operator"]))
    # Selection is explicit from already-used v152 8-prompt diagnostic;
    # even if every candidate worsens KL it is tested and reported FAIL.
    return {
       "best":best["original_operator"],
       "worst":worst["original_operator"],
       "best_old_diagnostic_KL_delta":best["single_operator_KL_vs_CB59"],
       "worst_old_diagnostic_KL_delta":worst["single_operator_KL_vs_CB59"],
       "all_diagnostic_operators_in_selection":len(table),
       "no_valid_teacher_logits_enter_selection":True,
       "no_training_or_requantization":True,
    }

def check():
    if (WIKI_START,NEWS_START,NEWS_END,TOKENS)!=(45056,13200,13300,512):
        raise ValueError("v155 source VALID was changed after selection")
    if not(WIKI_START>=40960+512 and NEWS_START>=12300):
        raise ValueError("overlaps previous v153 VALID source")
    if len(ARM_NAMES)!=4 or len(DOMAINS)!=2:
        raise ValueError("not preregistered original source controls")
    return True
