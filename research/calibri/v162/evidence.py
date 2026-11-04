"""Seal SHA of completed physically paid CB161 *before* opening fresh data."""
import json
from pathlib import Path
from v161.protocol import MATRIX_KEYS
from .protocol import (check,ORIGINAL_CODEBOOK_SHA,
                       PARENT_MODEL_BYTES,MAX_SOURCE_BYTES,TOTAL_UNIQUE)


def frozen_v161(physical,latest_report,sealed_source_report):
    check()
    old=json.loads(Path(latest_report).read_text())
    seal=json.loads(Path(sealed_source_report).read_text())
    if old.get("stage")!="v161_GENUINE_PAID_ORIGINAL_BLOCK0_REPAIR_VS_UNPAID_FP32_ORACLE":
        raise ValueError("not real completed v161 narrow source result")
    if seal.get("stage")!="v161_PHYSICAL_SOURCE_LOCK":
        raise ValueError("not v161 full archive pre-validation source SHA seal")
    if physical.original is None or physical.layer!=0 or (
            set(physical.patches)!=set(MATRIX_KEYS)):
        raise ValueError("actual original layer0 seven source operators not preserved")
    if (old["unchanged_original_CB59_SHA256"]!=ORIGINAL_CODEBOOK_SHA or
            seal["pinned_original_CB59_SHA256"]!=ORIGINAL_CODEBOOK_SHA):
        raise ValueError("original 5bit transformer source changed")
    if (seal["physically_serialized_complete_CB161_SHA256"]!=physical.sha256 or
            old["complete_source_SHA256"]!=physical.sha256):
        raise ValueError("complete paid archive SHA doesn't match source or previous result")
    report=old["physically_paid_CB161_0p7bpp_source"]
    if physical.size!=PARENT_MODEL_BYTES or (
            report["complete_CB153_archive_bytes"]!=physical.size or
            abs(report["all_in_bpp_unique_original"]-8*physical.size/TOTAL_UNIQUE)>1e-12):
        raise ValueError("physical original rate does not match original v161 evidence")
    if physical.size>MAX_SOURCE_BYTES:
        raise ValueError("physical original source exceeds 0.7 bpp")
    observed=old["per_arm_full_original_model_true_NLL_PPL_teacher_KL_top1"]
    if not observed["physically_paid_block0_improves_true_NLL_and_teacher_KL_in_both_domains"]:
        raise ValueError("v161 did not actually meet narrow source VALID gate")
    changes=observed["new_VALID_frozen_effects"]
    if tuple(changes)!=("wiki_block0_new_valid","news_block0_new_valid"):
        raise ValueError("v161 source provenance changed")
    return {
        "complete_v161_model_SHA256":physical.sha256,
        "source_byte_length":physical.size,
        "source_all_in_bpp":8*physical.size/TOTAL_UNIQUE,
        "preexisting_v161_NLL_KL_changes":changes,
        "narrow_8x64_PER_DOMAIN_v161_VALID_not_independent_TEST":True,
        "all_SEVEN_source_ops_physically_paid":True,
        "new_v162_validation_has_not_been_opened_at_seal":True,
    }
