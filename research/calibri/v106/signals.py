"""Character-level repetition signals for preexisting, SHA-locked responses.

Designed as DIAGNOSTIC flags, not automatic semantic truth or candidate
selection. Threshold is an exploratory choice after inspecting v96 output:
never report its apparent accuracy as unseen generalization.
"""
from __future__ import annotations
from collections import Counter
import re
import zlib

FIVEGRAM_N=5
DUPLICATE_COVERAGE_THRESHOLD=0.50
COMPRESSED_SIZE_RATIO_THRESHOLD=0.60
MIN_ALNUM_CHARS=32
FAKE_ROLE=re.compile(r"(?im)^[ \t]*(?:system|user|assistant)[ \t]*$")

def repeated_character_ngram_coverage(s,n=FIVEGRAM_N):
    clean=re.sub(r"[^a-z0-9]", "", (s or "").lower())
    if len(clean)<n:
        return 0.,len(clean)
    windows=(clean[i:i+n] for i in range(len(clean)-n+1))
    counts=Counter(windows)
    duplicates=sum(count for count in counts.values() if count>1)
    return duplicates/max(1,len(clean)-n+1),len(clean)

def generation_signals(text,token_ids=None):
    if not isinstance(text,str):
        raise TypeError("original generated response must be text")
    raw=text.encode("utf-8",errors="replace")
    duplicated,alnum_length=repeated_character_ngram_coverage(text)
    ratio=len(zlib.compress(raw))/max(1,len(raw))
    detector=(alnum_length>=MIN_ALNUM_CHARS and
         duplicated>DUPLICATE_COVERAGE_THRESHOLD and
         ratio<COMPRESSED_SIZE_RATIO_THRESHOLD)
    unique_token_fraction=None
    if token_ids is not None:
        if not token_ids:
            raise AssertionError("empty original trained-model continuation")
        unique_token_fraction=len(set(token_ids))/len(token_ids)
    return {
       "five_character_ngram_duplicated_window_fraction":round(duplicated,6),
       "compressed_UTF8_size_ratio_zlib_default":round(ratio,6),
       "alnum_ASCII_chars":alnum_length,
       "simulated_role_line_present":bool(FAKE_ROLE.search(text)),
       "unique_token_fraction_observed":unique_token_fraction,
       "exploratory_fivegram_AND_zlib_anomaly_alarm":bool(detector),
       "not_a_semantic_truth_or_accuracy_criterion":True,
       "thresholds_selected_after_historical_v96_review_NOT_independent_VALID":True,
    }
