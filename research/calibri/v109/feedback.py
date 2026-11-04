"""One encoder-side coordinate sweep. The native CB59 decoder is unchanged."""
import numpy as np


def principal_directions(inputs, rank=4):
    """Uncentered directions: a recurrent mean input is also worth protecting."""
    x = np.asarray(inputs, dtype=np.float64)
    _, singular, vectors = np.linalg.svd(x, full_matrices=False)
    if len(singular) < rank or singular[rank - 1] <= 0:
        raise ValueError("insufficient calibration rank")
    return np.ascontiguousarray(vectors[:rank].T, dtype=np.float32)


def recode_rows(target, book, original, basis):
    """Minimize ||e||² + (width/rank)||e U||² for each normalized weight row.

    Keep the residual of the *whole row* while choosing a centroid for each
    eight-weight group. An independent group metric would drop the cross terms.
    """
    ids = original.copy()
    rows, groups = ids.shape
    width, rank = basis.shape
    if width != groups * book.shape[1] or target.shape != (rows, width):
        raise ValueError("incompatible code, target or calibration dimensions")
    strength = width / rank
    group_weights = target.reshape(rows, groups, -1)
    error = target - book[ids].reshape(rows, width)
    residual = error @ basis

    def loss(e, projected):
        return float(np.square(e).sum(dtype=np.float64)
                     + strength * np.square(projected).sum(dtype=np.float64))

    before = loss(error, residual)
    center_norm = np.einsum("ij,ij->i", book, book)
    row_ids = np.arange(rows)
    for group, axes in enumerate(np.split(basis, groups)):
        projected_book = book @ axes
        old = ids[:, group]
        available = residual + projected_book[old]
        projected_norm = np.einsum("ij,ij->i", projected_book, projected_book)
        score = (center_norm - 2 * (group_weights[:, group] @ book.T)
                 + strength * (projected_norm - 2 * (available @ projected_book.T)))
        chosen = score.argmin(axis=1)
        # An exact tie must retain the incumbent to avoid gratuitous edits.
        chosen = np.where(score[row_ids, chosen] < score[row_ids, old], chosen, old)
        residual = available - projected_book[chosen]
        ids[:, group] = chosen

    error = target - book[ids].reshape(rows, width)
    after = loss(error, error @ basis)
    if not np.isfinite([before, after]).all() or after > before * (1 + 1e-5) + 1e-8:
        raise ArithmeticError("encoder increased its declared calibration objective")
    return ids, before, after
