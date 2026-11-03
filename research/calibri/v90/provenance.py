"""CALIBRI v90: diagnostic-only, physically frozen real sources; never official TEST."""
from __future__ import annotations
import hashlib
from v86.provenance import ORIGINAL_CB59_SHA
from v89.provenance import CB88_FROZEN_PRETEST_SHA
from v88.physical import unpack, audit

BOOLQ_DATASET = "google/boolq"
# Both slices are exclusively from the official *train* split.
SPLITS = {"TRAIN": (800, 832), "TRAIN_VALID": (6000, 6032)}
DIAGNOSTIC_PROMPTS = (
    "A student studied the instructions and then said",
    "Several engineers discussed the design and decided",
    "At the museum, the guide showed an unusual object",
)
TOKENS_PER_PROMPT = 32

def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()

def check_slices(splits=SPLITS):
    assert set(splits) == {"TRAIN", "TRAIN_VALID"}
    a, b = splits["TRAIN"], splits["TRAIN_VALID"]
    assert all(isinstance(x, int) for span in (a, b) for x in span)
    assert 0 <= a[0] < a[1] <= 9427 and 0 <= b[0] < b[1] <= 9427
    assert a[1] <= b[0] or b[1] <= a[0], "diagnostic slices overlap"
    return True

def verify_sources(original: bytes, cb88: bytes):
    check_slices()
    if sha(original) != ORIGINAL_CB59_SHA or sha(cb88) != CB88_FROZEN_PRETEST_SHA:
        raise AssertionError("expected physically frozen original CB59/CB88 sources")
    embedded, *_ = unpack(cb88)
    if original != embedded:
        raise AssertionError("CB88 does not embed exact CB59 pretrained native source")
    info = audit(cb88)
    if info["all_in_bits_per_original_parameter"] > 0.700:
        raise AssertionError("fully paid physical bit budget exceeded")
    return {"CB59_SHA": sha(original), "CB88_SHA": sha(cb88), "physical": info,
            "diagnostic_dataset": BOOLQ_DATASET, "train_only_slices": SPLITS,
            "official_TEST_not_loaded_or_used_for_selection": True}
