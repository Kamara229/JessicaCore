"""Ground GT feasibility in actually completed low-diversity v163 outputs."""
import json
from pathlib import Path
from .protocol import FULL_PARENT_ARCHIVE_BYTES,check


def verify_bad_real_generation(physical,v163_report):
    check()
    prior=json.loads(Path(v163_report).read_text())
    if prior.get("stage")!="v163_REAL_PAID_ORIGINAL_BLOCK0_NEW_QA_AND_GREEDY_GENERATION":
        raise ValueError("v163 real QA and greedy experiment required")
    if prior.get("immutable_physical_archive_SHA256")!=physical.sha256:
        raise ValueError("different physically paid source in v163")
    if physical.size!=FULL_PARENT_ARCHIVE_BYTES or physical.layer!=0:
        raise ValueError("not complete physically paid original block0 model")
    gen=prior["genuine_autoregressive_text_and_speed"][
        "genuine_full_model_autoregressive_measures"]
    d0=gen["original_CB59"]["distinct_token_ratio_over_all_four_responses"]
    d1=gen["fully_paid_CB161_block0"]["distinct_token_ratio_over_all_four_responses"]
    if abs(d0-0.0625)>1e-10 or abs(d1-0.0859375)>1e-10:
        raise ValueError("do not rewrite actual four greedy source outcomes")
    if (gen["original_CB59"]["genuine_full_model_greedy_prompts"]!=4 or
            gen["fully_paid_CB161_block0"]["genuine_full_model_greedy_prompts"]!=4):
        raise ValueError("not four genuine full-model greedy generations")
    return {
        "real_v163_four_greedy_CB59_distinct_token_ratio":d0,
        "real_v163_four_greedy_paid_block0_distinct_token_ratio":d1,
        "real_language_repetition_not_recovered":True,
        "physical_archive_SHA256":physical.sha256,
        "theoretical_oracle_does_not_solve_or_justify_GT_as_a_model":True,
    }
