"""A hierarchical K32×K2 vector quantizer, original 5 bits + 1 bit/group.

Train two residual centroids (8D) inside each frozen base codeword; the extra
bit selects which of those two centroids to add. The first 5-bit index need not
change. The 1-bit code is a new physical stream and must be counted in budget.
"""
import numpy as np


def assign(residual, pair):
    r=np.asarray(residual,dtype=np.float32)
    c=np.asarray(pair,dtype=np.float32)
    if r.ndim!=2 or c.shape!=(2,r.shape[1]):
        raise ValueError("two residual codeword geometry mismatch")
    d=((r[:,None,:]-c[None,:,:]).astype(np.float64)**2).sum(axis=2)
    return np.argmin(d,axis=1).astype(np.uint8)


def train(normalized,indices,base_centers,*,energy=None,steps=8):
    x=np.asarray(normalized,dtype=np.float32)
    idx=np.asarray(indices,dtype=np.int64)
    base=np.asarray(base_centers,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=8 or base.shape!=(32,8) or idx.shape!=(len(x),):
        raise ValueError("not trained with frozen 5-bit base codes")
    if not np.isfinite(x).all() or np.any(idx<0) or np.any(idx>=32):
        raise ValueError("bad original-weight source samples")
    weights=np.ones(len(x),dtype=np.float64) if energy is None else np.asarray(
        energy,dtype=np.float64)
    if weights.shape!=(len(x),) or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError("invalid energy weighting")
    if not 1<=steps<=16:
        raise ValueError("invalid training pass count")
    residual=x-base[idx]
    output=np.zeros((32,2,8),dtype="<f2")
    for code in range(32):
        mask=idx==code
        group=residual[mask].astype(np.float32)
        w=weights[mask]
        if len(group)<2:
            if len(group):
                output[code,0]=group[0].astype("<f2")
                output[code,1]=group[0].astype("<f2")
            continue
        # Deterministic PCA principal axis, with stable sign and fallbacks.
        mu=np.average(group,axis=0,weights=w)
        centered=(group-mu).astype(np.float64)
        gram=(centered*w[:,None]).T@centered
        vals,vecs=np.linalg.eigh(gram)
        direction=vecs[:,-1]
        if direction[np.argmax(abs(direction))]<0:
            direction=-direction
        projected=centered@direction
        low=projected<=np.median(projected)
        if np.all(low) or not np.any(low):
            low=np.arange(len(group))%2==0
        pair=np.stack([np.average(group[low],axis=0,weights=w[low]),
                       np.average(group[~low],axis=0,weights=w[~low])]).astype("<f2").astype(np.float32)
        for _ in range(steps):
            b=assign(group,pair)
            next_pair=pair.copy()
            for j in (0,1):
                sub=b==j
                if np.any(sub):
                    next_pair[j]=np.average(group[sub],axis=0,weights=w[sub]).astype("<f2").astype(np.float32)
            pair=next_pair
        output[code]=pair.astype("<f2")
    if not np.isfinite(output).all():
        raise ValueError("nonfinite physical residual FP16 centroids")
    return output.astype(np.float32)


def score(actual,normalized,scales,base_codes,base_centers,residual_centers=None):
    src=np.asarray(actual,dtype=np.float64)
    x=np.asarray(normalized,dtype=np.float32)
    scale=np.asarray(scales,dtype=np.float64)
    idx=np.asarray(base_codes,dtype=np.int64)
    centers=np.asarray(base_centers,dtype=np.float32)
    if src.shape!=x.shape or x.ndim!=2 or x.shape[1]!=8:
        raise ValueError("invalid normalized original weight")
    if idx.shape!=(len(x),) or scale.shape!=(len(x),):
        raise ValueError("invalid base coding")
    if np.any(idx<0) or np.any(idx>=32) or centers.shape!=(32,8):
        raise ValueError("bad K32 main codebook")
    base=centers[idx].copy()
    bit=None
    if residual_centers is not None:
        r=np.asarray(residual_centers,dtype=np.float32)
        if r.shape!=(32,2,8):
            raise ValueError("unexpected residual bitplane codebook geometry")
        values=x-base
        bit=np.zeros(len(x),dtype=np.uint8)
        for code in range(32):
            chosen=np.where(idx==code)[0]
            if len(chosen):
                bit[chosen]=assign(values[chosen],r[code])
                base[chosen]+=r[code,bit[chosen]]
    reconstructed=base.astype(np.float64)*scale[:,None]
    delta=src-reconstructed
    rmse=float(np.sqrt((delta*delta).sum()/max(float((src*src).sum()),1e-14)))
    return {
        "relative_weight_RMSE":rmse,
        "max_absolute_error":float(np.max(np.abs(delta))),
        "mean_normalized_8d_squared_error":float(np.mean(((x-base)**2).sum(axis=1))),
        "refinement_onebit_codes_USED":(None if bit is None else [int((bit==j).sum()) for j in (0,1)]),
    }
