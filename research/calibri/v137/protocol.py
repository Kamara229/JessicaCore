"""Fixed source-weight vector sample and honest rate-accounting anchors."""
import numpy as np
from v117.protocol import TOTAL_UNIQUE,MAX_SOURCE_BYTES
from direct_codebook_v59 import K,VECTOR

SEED=137
FIT_GROUPS=98_304
VALID_GROUPS=16_384
ITERATIONS=8
assert (VECTOR,K)==(8,32)


def select(total):
    if FIT_GROUPS+VALID_GROUPS>total: raise ValueError("source tensor too small")
    rng=np.random.default_rng(SEED)
    indices=rng.choice(total,size=FIT_GROUPS+VALID_GROUPS,replace=False)
    fit=indices[:FIT_GROUPS]
    valid=indices[FIT_GROUPS:]
    if len(np.intersect1d(fit,valid)):
        raise AssertionError("source fit/valid overlap")
    return fit,valid


def groups(weight,scales,indices):
    rows,cols=weight.shape
    if cols%VECTOR or len(scales)!=rows:
        raise ValueError("unexpected embedding source geometry")
    per_row=cols//VECTOR
    row=np.asarray(indices,dtype=np.int64)//per_row
    block=np.asarray(indices,dtype=np.int64)%per_row
    if np.any(row>=rows) or np.any(row<0):
        raise ValueError("out-of-bounds source-group index")
    actual=weight[row[:,None],block[:,None]*VECTOR+np.arange(VECTOR)].astype(np.float32)
    scale=np.asarray(scales[row],dtype=np.float32)
    normalized=(actual/scale[:,None]).astype("<f2").astype(np.float32)
    return actual,normalized,scale


def physical_budget(source_bytes, matrix):
    m,n=matrix
    # Source archive keeps the same number of fixed-size groups and centroids.
    paid_index_bytes=((m*n//VECTOR)*5+7)//8
    additional_bytes=MAX_SOURCE_BYTES-source_bytes
    if additional_bytes<0: raise ValueError("source above budget")
    return {
        "original_CB59_bytes":source_bytes,
        "original_unique_parameters":TOTAL_UNIQUE,
        "room_to_0_7bpp_bytes":additional_bytes,
        "embedding_packed_codes_fixed_bytes":paid_index_bytes,
        "same_codebook_entries":K,
        "same_vector_dimension":VECTOR,
        "same_centroid_size_fp16_bytes":K*VECTOR*2,
        "incremental_centroid_training_bytes_IF_repacked":0,
    }
