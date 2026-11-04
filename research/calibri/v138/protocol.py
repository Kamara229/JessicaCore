"""Predeclared source-only centroid variants from completed v137 VALID pilot."""
from v137.protocol import select,groups,ITERATIONS
from v137.fit import lloyd

WEIGHTED_SHA="06f34f1f4c3b258ad1d12113d912c91384ad006faf0b5538bbdd5c95fa2fa6c9"
UNIFORM_SHA="e8a184eeee887b2fc793cbadbf9422723010976741fc7d78fb7a689c8d8507ae"
CB59_BASE_SHA="50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"


def choose_source_centers(weight,original):
    import hashlib
    import numpy as np
    total=weight.shape[0]*(weight.shape[1]//8)
    fit_indices,_=select(total)
    _,normalized,scale=groups(weight,original.row_scales,fit_indices)
    initial=original.centroids
    output={
        "uniform_recode_CB59":lloyd(normalized,initial,passes=ITERATIONS),
        "weighted_recode_CB59":lloyd(normalized,initial,
                                     energy=np.square(scale.astype(np.float64)),
                                     passes=ITERATIONS),
    }
    for key,expected in (("uniform_recode_CB59",UNIFORM_SHA),
                         ("weighted_recode_CB59",WEIGHTED_SHA)):
        actual=hashlib.sha256(output[key].astype("<f2").tobytes()).hexdigest()
        if actual!=expected:
            raise ValueError(f"v137 source-only codebook changed for {key}: {actual}")
    return output
