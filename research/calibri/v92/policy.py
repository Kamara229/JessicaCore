"""Safe autonomic operation: independently gate research candidate, forbid live mutation."""
from __future__ import annotations
from binary_geometry_v64 import EXPECTED_UNIQUE

MAX_BPP=.7
MIN_SPEED_RATIO=.90
MIN_GREEDY_DISTINCT=.25

class ApprovalRequired(RuntimeError):pass

def require_research_only(action):
    permitted={"inspect","diagnose","propose","sandbox_rewrite","validate","stage_candidate"}
    if action not in permitted:
        raise ApprovalRequired("no live deploy/merge/TEST-informed optimization: "+str(action))
    return True

def physical_budget(total_bytes):
    bpp=8*total_bytes/EXPECTED_UNIQUE
    return {"total_bytes":total_bytes,"bpp":bpp,"PASS":bpp<=MAX_BPP}

def decide_candidate(*,source_bytes,valid_teacher_KL_gain,valid_QA_regression,
                     valid_greedy_ratio,valid_speed_ratio,full_model_speed_proven=False,
                     true_language_NLL_proven=False):
    budget=physical_budget(source_bytes)
    checks={"physical_budget":budget["PASS"],
            "teacher_KL_gain_strict_positive":valid_teacher_KL_gain>0,
            "valid_QA_no_regression":valid_QA_regression>=0,
            "greedy_diversity":valid_greedy_ratio>=MIN_GREEDY_DISTINCT,
            "native_speed":valid_speed_ratio>=MIN_SPEED_RATIO}
    return {"checks":checks,"fully_paid":budget,
        "status":"STAGE_FOR_REVIEW" if all(checks.values()) else "REJECT",
        "automatic_merge_or_production_write":False,
        "new_independent_TEST_not_opened":True}
