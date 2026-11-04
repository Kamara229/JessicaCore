"""Eight contiguous vocabulary banks, source-only deterministic K32 V8.

Existing CB59 head stores one 8x32 centroid bank, so CB104 keeps it
for parser ABI, and adds exactly 8 explicitly paid FP16 banks in wrapper.
The native BANKED decoder ALWAYS uses the wrapper bank centroids for
head+embedding, no special-case or unpriced original weight materialization.
"""
from __future__ import annotations
import hashlib
import numpy as np
from direct_codebook_v59 import CB59,row_scales,_centroids
from v102.recode import recode_rows,swap_whole_tied_head_blob
from .spec import BANKS,VOCAB,DIM,SAMPLES_PER_BANK,ROW_PART
from .physical import pack,unpack

def recode_original(frozen:bytes,teacher_weight):
    old=CB59(frozen)
    h=old.handles["lm_head.weight"]
    if h.shape!=(VOCAB,DIM):
        raise AssertionError("wrong exact original trained source head shape")
    w=np.asarray(teacher_weight.detach().cpu().numpy(),dtype=np.float32)
    if w.shape!=(VOCAB,DIM) or not np.isfinite(w).all():
        raise AssertionError("pinned teacher tied output original trained weights required")
    scales=row_scales(w).astype(np.float32)
    bank_centroids=[];packed_by_bank=[];source_scales=[];error_numerator=0.;energy=0.
    for bank in range(BANKS):
        start=bank*ROW_PART
        end=(bank+1)*ROW_PART
        rows=w[start:end]
        parts=rows.reshape(-1,8)
        keys=np.linspace(0,len(parts)-1,SAMPLES_PER_BANK,dtype=np.int64)
        normalized=parts[keys].astype(np.float32)/scales[start+keys//(DIM//8),None]
        c=_centroids(normalized).astype("<f2")
        if c.shape!=(32,8):raise AssertionError("wrong bank centroid shape")
        segment,scale_bytes,error=recode_rows(rows,c.astype(np.float32),
                                              scales[start:end],row_ls=True)
        bank_centroids.append(c)
        packed_by_bank.append(segment)
        source_scales.append(scale_bytes)
        y=rows.astype(np.float64)
        energy+=float((y*y).sum())
        error_numerator+=float(error*error*(y*y).sum())
    centroids=np.asarray(bank_centroids,dtype="<f2")
    parts={"packed":b"".join(packed_by_bank),
        "row_scales":b"".join(source_scales),
        "centroids":centroids[0].tobytes()}
    base=swap_whole_tied_head_blob(frozen,parts)
    file=pack(base,centroids)
    decoded,cent=unpack(file)
    if decoded!=base or not np.array_equal(cent,centroids):
        raise AssertionError("physical source bank codes unable to round-trip")
    return file,{
        "baseline_original_source_SHA256":hashlib.sha256(frozen).hexdigest(),
        "source_only_true_pretrained_tied_head_RMSE":float(np.sqrt(error_numerator/energy)),
        "all_original_other_trained_parameters_byte_identical":True,
        "original_CB59_native_nonhead_codebooks_unchanged":True,
        "original_tied_head_single_canonical_physical_matrix":True,
        "banked_dictionary_partition":"contiguous 6144 token vocabulary rows per bank",
        "codebook_fitting_uses_only_original_trained_weights":True,
        "no_validation_label_or_QA_used_to_select_codebook":True,
        "new_actual_repacked_5bit_source_bytes":len(base)}
