"""Verify real v162 FAIL on an untouched 32-window source before diagnosis."""
import json
from pathlib import Path
from .protocol import MATRIX_KEYS,PARENT_MODEL_BYTES,ORIGINAL_CB59_SHA,check


def verify_v162_fail(physical,report_path):
    check()
    data=json.loads(Path(report_path).read_text())
    if data.get("stage")!="v162_REAL_FULLY_PAID_BLOCK0_32WINDOW_TWO_DOMAIN_ROBUSTNESS":
        raise ValueError("not actual complete v162 source-robustness result")
    if data.get("source_parent_frozen_v161_SHA256")!=physical.sha256:
        raise ValueError("changed physically paid original block0 source")
    if (physical.layer!=0 or
            set(physical.patches)!=set(MATRIX_KEYS) or
            physical.size!=PARENT_MODEL_BYTES):
        raise ValueError("not complete original 7-matrix paid physical source")
    if data.get("original_complete_CB59_source_SHA256")!=ORIGINAL_CB59_SHA:
        raise ValueError("native original CB59 codewords changed")
    audit=data["true_NLL_PPL_teacher_KL_early_teacher_argmax_results"]
    if audit["both_domain_precommitted_STRONG_NLL_KL_robustness_pass"]:
        raise ValueError("v162 did not fail precommitted crossdomain gate")
    domains=audit["per_domain_robustness"]
    wiki=domains["wikitext_fifth_new_block0"]
    news=domains["ag_news_fifth_new_block0"]
    if (not wiki["per_domain_robust_gate"] or
            news["per_domain_robust_gate"] or
            news["paired_TRUE_NLL"]["improved_windows_out_of_32"]!=18 or
            news["paired_teacher_KL"]["improved_windows_out_of_32"]!=17):
        raise ValueError("v162 actual negative source-window evidence overwritten")
    return {
        "source_sha256":physical.sha256,
        "whole_physically_paid_source_bytes":physical.size,
        "v162_both_domains_STRONG_gate":"FAIL",
        "v162_WikiText_gate":"PASS",
        "v162_AG_News_gate":"FAIL",
        "v162_AG_News_NLL_improved_windows_out_of_32":18,
        "v162_AG_News_KL_improved_windows_out_of_32":17,
        "no_unseen_VALID_or_TEST_used_for_source_operator_attribution":True,
    }
