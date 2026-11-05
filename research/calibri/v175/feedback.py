"""Groupwise constrained quadratic compensation, independently implemented.

For H^-1 = U.T @ U, a fixed group error e is compensated in remaining
weights by -e @ inv(U_BB) @ U_BR. The conditional metric is ||e inv(U_BB)||².
This alters assignments only: one existing K32 code per eight coefficients.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from .protocol import DAMPING, TIED_EMBEDDING_PENALTY


def metric_and_factor(second_moment, *, tied=False):
    h = np.array(second_moment, dtype=np.float64, copy=True)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] % 8:
        raise ValueError("Invalid covariance geometry")
    if not np.isfinite(h).all() or not np.allclose(h, h.T, atol=1e-10, rtol=1e-7):
        raise ValueError("Nonfinite or nonsymmetric covariance")
    scale = float(np.mean(np.diag(h)))
    if scale <= 0:
        raise ValueError("No source activation energy")
    h = (h + h.T) * (0.5 / scale)
    penalty = DAMPING + (TIED_EMBEDDING_PENALTY if tied else 0.0)
    h.flat[::len(h) + 1] += penalty
    inv = cho_solve(cho_factor(h, lower=True), np.eye(len(h)))
    inv = (inv + inv.T) * 0.5
    upper = np.linalg.cholesky(inv).T
    return h, upper


def conditional_codes(normalized_weights, centers, upper):
    work = np.array(normalized_weights, dtype=np.float64, copy=True)
    c = np.asarray(centers, dtype=np.float64)
    u = np.asarray(upper, dtype=np.float64)
    if work.ndim != 2 or work.shape[1] % 8 or c.shape != (32, 8):
        raise ValueError("Wrong K32/V8 geometry")
    rows, cols = work.shape
    if u.shape != (cols, cols) or not np.allclose(u, np.triu(u)) or np.any(np.diag(u) <= 0):
        raise ValueError("Invalid upper covariance factor")
    if not all(np.isfinite(x).all() for x in (work, c, u)):
        raise ValueError("Nonfinite source")
    codes = np.empty((rows, cols // 8), dtype=np.uint8)
    for first in range(0, cols, 8):
        stop = first + 8
        inv_block = solve_triangular(u[first:stop, first:stop], np.eye(8))
        transformed_centers = c @ inv_block
        transformed = work[:, first:stop] @ inv_block
        # Row norm is constant across candidate centroids and can be omitted.
        costs = np.sum(transformed_centers ** 2, axis=1)[None, :] - 2 * transformed @ transformed_centers.T
        chosen = np.argmin(costs, axis=1).astype(np.uint8)
        codes[:, first // 8] = chosen
        error = (work[:, first:stop] - c[chosen]) @ inv_block
        work[:, stop:] -= error @ u[first:stop, stop:]
    return codes


def quadratic_error(weights, reconstructed, metric):
    total = 0.0
    for first in range(0, len(weights), 256):
        e = (np.asarray(weights[first:first+256], dtype=np.float64)
             - np.asarray(reconstructed[first:first+256], dtype=np.float64))
        total += float(np.sum((e @ metric) * e))
    return total


def fit_indices(weights, handle, old_codes, second_moment, *, tied=False):
    w = np.asarray(weights, dtype=np.float32)
    scales = np.asarray(handle.row_scales, dtype=np.float32)
    centers = np.asarray(handle.centroids, dtype=np.float32)
    old = np.asarray(old_codes)
    if w.shape != handle.shape or old.shape != (len(w), w.shape[1] // 8):
        raise ValueError("Source shape mismatch")
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("Invalid stored row scales")
    h, upper = metric_and_factor(second_moment, tied=tied)
    proposed = conditional_codes(w.astype(np.float64) / scales[:, None], centers, upper)
    old_dense = centers[old].reshape(w.shape) * scales[:, None]
    new_dense = centers[proposed].reshape(w.shape) * scales[:, None]
    before = quadratic_error(w, old_dense, h)
    after = quadratic_error(w, new_dense, h)
    # Calibration fallback is not a model-quality gate. No QA is used here.
    accepted = bool(np.isfinite(after) and after <= before)
    chosen = proposed if accepted else old.copy()
    return chosen, {"fit_quadratic_before": before, "fit_quadratic_proposed": after,
                    "fit_quadratic_selected": after if accepted else before,
                    "accepted_on_calibration_only": accepted,
                    "changed_groups": int(np.count_nonzero(chosen != old)),
                    "tied_source_penalty": TIED_EMBEDDING_PENALTY if tied else 0.0}
