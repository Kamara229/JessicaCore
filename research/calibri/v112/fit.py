"""Only pretrained source weights enter shared codebook fitting, never VALID."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from .roles import bank_for

GROUP = 8
K = 32
BANKS = 4
SEED = 112
MAX_SAMPLES = 22000

def normalize(matrix):
    w = np.asarray(matrix, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % GROUP:
        raise ValueError("bad matrix")
    scale = np.sqrt(np.maximum(np.mean(w*w, axis=1), 1e-20))
    return np.ascontiguousarray((w / scale[:, None]).reshape(-1, GROUP), dtype=np.float32)

def learn_banks(source, mode):
    if mode not in ("global", "role", "layer_control"):
        raise ValueError(mode)
    groups = {k: [] for k in range(BANKS)}
    for name, w in sorted(source.items()):
        key = bank_for(mode, name)
        z = normalize(w)
        # Evenly sample all matrices to avoid domination by one larger role.
        if len(z) > 35000:
            z = z[np.linspace(0, len(z)-1, 35000).astype(int)]
        groups[key].append(z)
    rng = np.random.default_rng(SEED)
    centers = np.zeros((BANKS, K, GROUP), dtype=np.float32)
    for k in (range(1) if mode == "global" else range(BANKS)):
        if not groups[k]:
            raise ValueError("empty source bank")
        raw = np.concatenate(groups[k])
        if len(raw) > MAX_SAMPLES:
            raw = raw[rng.choice(len(raw), MAX_SAMPLES, replace=False)]
        if len(raw) < K:
            raise ValueError("fewer samples than centroids")
        fit = KMeans(n_clusters=K, n_init=1, max_iter=25, random_state=SEED+k, algorithm="lloyd")
        fit.fit(raw)
        centers[k] = fit.cluster_centers_.astype(np.float16).astype(np.float32)
    if mode == "global":
        centers[:] = centers[0]
    return np.ascontiguousarray(centers, dtype=np.float16)

def nearest(vectors, centers):
    z = np.asarray(vectors, dtype=np.float32)
    book = np.asarray(centers, dtype=np.float32)
    result = np.empty(len(z), dtype=np.uint8)
    c2 = np.sum(book*book, axis=1)
    for start in range(0, len(z), 4096):
        slab = z[start:start+4096]
        d = c2[None,:] - 2*slab @ book.T + np.sum(slab*slab, axis=1)[:,None]
        result[start:start+len(slab)] = np.argmin(d, axis=1).astype(np.uint8)
    return result

def encode_matrix(w, book):
    w = np.asarray(w, dtype=np.float32)
    z = normalize(w)
    indices = nearest(z, book)
    predicted = np.asarray(book, dtype=np.float32)[indices].reshape(w.shape)
    denom = np.sum(predicted*predicted,axis=1)
    scales = np.sum(w*predicted,axis=1) / np.maximum(denom, 1e-16)
    scales = scales.astype("<f2")
    if not np.isfinite(scales).all():
        raise FloatingPointError("nonfinite scale")
    return indices, scales
