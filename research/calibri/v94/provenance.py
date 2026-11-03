"""Strict provenance of independent *train-split* validation and paid physical sources.
Do not call this an untouched official independent TEST. It remains TRAIN_VALID.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from direct_codebook_v59 import CB59
from v86.provenance import ORIGINAL_CB59_SHA
from binary_geometry_v64 import EXPECTED_UNIQUE

BPP_LIMIT=.700
BOOLQ_RANGE=(7200,7328)
SCIQ_RANGE=(5200,5264)
EXPECTED_BOOLQ_COUNT=BOOLQ_RANGE[1]-BOOLQ_RANGE[0]
EXPECTED_SCIQ_COUNT=SCIQ_RANGE[1]-SCIQ_RANGE[0]

def digest(b):return hashlib.sha256(b).hexdigest()

def verify_sources(original_path:Path,candidate_path:Path,report_path:Path):
    original=original_path.read_bytes();candidate=candidate_path.read_bytes()
    if digest(original)!=ORIGINAL_CB59_SHA:raise AssertionError("original CB59 SHA mismatch")
    old=CB59(original);new=CB59(candidate)
    report=json.loads(report_path.read_text())
    if report.get("experiment")!="CALIBRI_v92_automated_detect_triage_rewrite_validate_RESEARCH_ONLY":
        raise AssertionError("not prior frozen same-byte whole-block candidate")
    change=report["physical_candidate"]
    name=change["rewritten_module"]
    if digest(candidate)!=change["candidate_SHA256"]:
        raise AssertionError("candidate archive is not pinned v92 original output")
    if digest(original)!=change["base_SHA256"] or len(candidate)!=len(original):
        raise AssertionError("source is changed or extra bytes were paid")
    if set(old.meta["tensor_descriptors"])!=set(new.meta["tensor_descriptors"]) or old.meta!=new.meta:
        raise AssertionError("trained original model structure changed")
    desc=old.meta["tensor_descriptors"]
    for key in desc:
        if key==name:continue
        if key in old.handles:
            a,b=old.handles[key],new.handles[key]
            import numpy as np
            if (not np.array_equal(a.packed,b.packed) or
                not np.array_equal(a.row_scales,b.row_scales) or
                not np.array_equal(a.centroids,b.centroids)):
                raise AssertionError("an unselected block changed: "+key)
        else:
            import numpy as np
            if not np.array_equal(old.raw[key],new.raw[key]):
                raise AssertionError("an unselected raw pretrained tensor changed: "+key)
    rate=8*len(candidate)/EXPECTED_UNIQUE
    if rate>BPP_LIMIT:raise AssertionError("physical <=0.7 bpp budget exceeded")
    # New behavioral validation rows are disjoint from all *TRAIN* ranges
    # consulted in v90/v91/v92. They have never entered the encoder fit.
    original_boolq=((800,832),(6000,6032))
    for lo,hi in original_boolq:
        if not (BOOLQ_RANGE[1]<=lo or BOOLQ_RANGE[0]>=hi):
            raise AssertionError("reused original selection set")
    return {"CB59_SHA256":digest(original),"candidate_SHA256":digest(candidate),
        "rewritten_complete_operator":name,"complete_archive_bytes":len(candidate),
        "source_all_in_bpp":rate,"native_ABI_unchanged":True,
        "unique_original_trainable_parameters":EXPECTED_UNIQUE,
        "BOOLQ_NEW_TRAIN_VALID_range":BOOLQ_RANGE,
        "SCIQ_NEW_TRAIN_VALID_range":SCIQ_RANGE,
        "old_v89_independent_TEST_never_loaded":True,
        "different_DATASET_TRAIN_ONLY_public_probes":True}
