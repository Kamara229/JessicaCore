"""Validate finished v156 exploratory selection (not its independent TEST)."""
import json
from pathlib import Path
from .protocol import SELECTED,CONTROL,DOMAINS,CB153_SHA,check


def verify_prior_report(path):
    check()
    data=json.loads(Path(path).read_text())
    if data.get("stage")!="v156_POST_SELECTION_THIRD_DISJOINT_SOURCE_VALID_VPROJ":
        raise ValueError("not completed v156 original-source report")
    if data.get("immutable_whole_CB153_sha")!=CB153_SHA:
        raise ValueError("changed original physical CB153")
    results=data["paired_window_sign_and_holdout_audit"]
    if not results["v_proj_improves_NLL_and_FP32_teacher_KL_on_both_third_holdout_domains"]:
        raise ValueError("v156 hypothesis not an actual mean pass")
    previous=results["new_domain_deltas_and_pairing"]
    if sorted(previous)!=sorted(("wiki_THIRD_unseen_VALID","news_THIRD_unseen_VALID")):
        raise ValueError("v156 corpus replaced")
    for x in previous.values():
        if not(x["v_proj_minus_CB59_NLL"]<0 and x["v_proj_minus_CB59_KL"]<0):
            raise ValueError("v156 did not show paired mean gains")
    wik=previous["wiki_THIRD_unseen_VALID"]["paired_eight_windows"]
    if len(wik["paired_window_mean_NLL_deltas"])!=8:
        raise ValueError("v156 paired window evidence missing")
    if wik["NLL_better_windows_of_8"]!=3 or wik["KL_better_windows_of_8"]!=2:
        raise ValueError("v156 observed fragile WikiText win was erased")
    return {
        "prior_source_SHA256":CB153_SHA,
        "frozen_candidate":SELECTED,
        "negative_control":CONTROL,
        "original_v156_WikiText_nll_better_windows":3,
        "original_v156_WikiText_kl_better_windows":2,
        "v156_means_improved_but_windows_were_inconsistent":True,
        "v156_is_model_selection_evidence_NOT_final":True,
        "v157_new_training_source_NOT_opened":True,
    }
