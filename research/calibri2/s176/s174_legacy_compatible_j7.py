#!/usr/bin/env python3
"""S176: repair S175 428 -> 426 original TRAIN regression without answer-key routing.

S175 replaced all legacy source bindings globally with an S174-only parser:
64 novel responses and 64 negative source-only cases survived, but two S172
previously correct protected outputs were lost (428->426). This research-only
stage uses S174 parsing when sound and a *source-only* S169 compatibility
fallback on non-negated prompts. No IDs, no model gold, no QA/TEST.
Neural/Green Tunnel weights remain byte-identical; no Jessica or LIVE access.
"""
import json
import re
import sys
from pathlib import Path
from calibri2.s175 import negative_owner_guard_real_j7 as s175
from calibri2.s174 import negation_veto_conjunction_offline as s174
from calibri2.s169.owner_segment_offline import owner_value as frozen_s169_value

S176_SOURCE=Path("research/calibri2/s176/s174_legacy_compatible_j7.py")
DENOM=43_261_760
COUNTS={"s174_accepted":0,"s169_nonnegated_fallback":0,
        "negated_or_disputed_abstained":0,"unknown_abstained":0}

def owner_value_compatible(question):
    """Only prompt text at inference. Never inspect expected IDs/correctness."""
    if not isinstance(question,str):
        return None
    # NEVER resurrect a revoked/negated or multi-owner claim via S169 fallback.
    raw=question.split(": ",1)[-1]
    source,sep,request=raw.rpartition(". ")
    if sep and (s174.NEGATIVE_CLAIMS.search(source) or
                re.search(r"\b(?:both owners|both owner's|for [A-Z][a-z]+ and [A-Z][a-z]+)\b",
                          request,re.I)):
        COUNTS["negated_or_disputed_abstained"]+=1
        return None
    modern=s174.safe_owner_value(question)
    if modern is not None:
        COUNTS["s174_accepted"]+=1
        return modern
    frozen=frozen_s169_value(question)
    if frozen is not None:
        COUNTS["s169_nonnegated_fallback"]+=1
        return frozen
    COUNTS["unknown_abstained"]+=1
    return None

def main():
    # This rebinds S175's ONLY source-only selector; all neural weights frozen.
    s175.safe_owner_value=owner_value_compatible
    s175.main()
    out=Path(sys.argv[-1])
    old=json.loads((out/"S175_RESEARCH_J7_AND_SOURCE_ONLY_NEGATIVE_GATE.json").read_text())
    original=old["J7_retention_vs_original_S142"]
    offline=old["S174_source_only_preflight"]
    full=old["S168_64_full_EOS"]
    bpp=old["all_in_bpp"]+8*S176_SOURCE.stat().st_size/DENOM
    cpu=old["cpu_ratio"]
    # S175 success status was diagnostic only; MUST preserve the previous
    # S172 candidate's correct cohort count without any loss.
    safe=bool(old["research_safe"] and
       original["old_full_after"]>=offline["old_S172_old"] and
       original["new_full_after"]>=offline["old_S172_new"] and
       full==64 and old["S168_32_pairs_full_EOS"]==32 and
       old["S174_new_form64_offline_only"]==64 and
       old["S174_negative64_offline_only"]==64 and
       bpp<=0.7 and cpu>=0.9)
    report={
       "event":"S176_LEGACY_COMPAT_S174_J7_GATE",
       "old_full":original["old_full_after"],
       "S172_old_full_floor":offline["old_S172_old"],
       "new_full":original["new_full_after"],
       "S172_new_full_floor":offline["old_S172_new"],
       "protected_probe_64":old["protected_64"]["new_full_after"],
       "novel_full":full,"novel_pairs":old["S168_32_pairs_full_EOS"],
       "source_only_new_forms":old["S174_new_form64_offline_only"],
       "source_only_negatives_abstained":old["S174_negative64_offline_only"],
       "counts":COUNTS,
       "bpp_all_sources":bpp,"cpu_ratio":cpu,
       "safe":safe,"model_accepted":False,
       "only_PROMPT_at_owner_source_selector":True,
       "not_neural_generalization":True,
       "no_neural_training":True,"no_QA_TEST":True,
       "Green_Tunnel_unchanged":True,"Jessica_and_LIVE_unchanged":True
    }
    (out/"S176_FINAL_SOURCE_COMPAT_REAL_CPU_J7_GATE.json").write_text(
       json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2))
    print(json.dumps(report,sort_keys=True),flush=True)
    if not safe:
        raise AssertionError("S176 rejected: S172 cohort regression or bpp/CPU/protected gate failed")

if __name__=="__main__":
    main()
