"""Require *actual* completed v157 FAIL and v158 upstream-drift evidence.

Do not silently replace a FAIL with a hypothesis. These reports are
existing source data; v159 traces exclusively original already-consumed
TRAIN FIT and never opens protected v89 TEST or new VALID.
"""
import json
from pathlib import Path
from .protocol import CB153_SHA,FIT_DOMAINS


def verify_prior(v157,v158):
    out157=json.loads(Path(v157).read_text())
    out158=json.loads(Path(v158).read_text())
    if out157.get("stage")!="v157_FOURTH_UNSEEN_SOURCE_32WINDOW_VPROJ_OUTLIER_ROBUSTNESS":
        raise ValueError("missing actual v157 32-window robustness run")
    if out158.get("stage")!="v158_ORIGINAL_LAYER6_VPROJ_WEIGHT_ERROR_VS_UPSTREAM_DRIFT":
        raise ValueError("missing actual v158 matched operator input/error diagnostic")
    if out157.get("immutable_complete_CB153_SHA256")!=CB153_SHA:
        raise ValueError("v157 source changed")
    if out158.get("exact_frozen_paid_CB153_sha256")!=CB153_SHA:
        raise ValueError("v158 source changed")
    v157_gate=out157["window_level_outlier_and_strong_two_domain_gate"]
    if v157_gate["v_proj_STRONG_two_domain_robustness_pass"]:
        raise ValueError("previous v_proj 32-window gate was not FAIL")
    d157=v157_gate["honest_fourth_NEW_source_results"]
    wiki=d157["wiki_fourth_robustness_VALID"]
    news=d157["news_fourth_robustness_VALID"]
    if not (
       wiki["v_proj_minus_CB59_gold_NLL"]>0 and
       wiki["v_proj_minus_CB59_teacher_KL"]>0 and
       not wiki["precommitted_STRONG_pass"] and
       not news["precommitted_STRONG_pass"]):
        raise ValueError("v157 previously measured negative evidence changed")
    v158_by_domain=out158["findings"]["original_TRAlN_FIT_only_domain_findings"]
    if tuple(v158_by_domain)!=FIT_DOMAINS:
        raise ValueError("v158 missing original source domains")
    if not all(v158_by_domain[d]["upstream_drift_dominates_original_weight_only_error"]
               and not v158_by_domain[d]["paid_1bit_reduces_combined_actual_original_operator_error"]
               for d in FIT_DOMAINS):
        raise ValueError("v158 earlier result is not the documented upstream drift")
    return {
       "v157_real_32window_two_domain_strong_gate":"FAIL",
       "v157_WikiText_TRUE_NLL_delta":wiki["v_proj_minus_CB59_gold_NLL"],
       "v157_WikiText_teacher_KL_delta":wiki["v_proj_minus_CB59_teacher_KL"],
       "v157_News_precommitted_gate":"FAIL",
       "v158_original_fit_upstream_drift_dominance_ratios":{
           d:v158_by_domain[d]["ratio_upstream_drift_vs_original_5bit_weight_only_error"]
           for d in FIT_DOMAINS},
       "v158_paid_local_operator_combined_error_improved":False,
       "original_paid_source_sha256":CB153_SHA,
       "no_SciQ_BoolQ_or_official_independent_TEST_claim":True,
    }
