"""Intruder mining with an exact 2-D KD-tree: O(n log n) instead of the O(n^2) chunked distance scan."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def mine_intruders_kdtree(Y, exclude_idx, m, n_jobs=8):
    """For every point, its m nearest embedded points that are not in exclude_idx (and not itself).

    exclude_idx: (n, K) int array (the K high-dimensional neighbors that must not count as intruders).
    Queries K + m + 1 neighbors; the rare rows with fewer than m survivors are re-queried with a larger radius.
    """
    n, K = exclude_idx.shape
    q = K + m + 1
    nn = NearestNeighbors(n_neighbors=min(q, n), algorithm="kd_tree", n_jobs=n_jobs).fit(Y)
    cand = nn.kneighbors(Y, return_distance=False)
    out = np.empty((n, m), dtype=np.int64)
    short = []
    excl_sets = None
    rows = np.arange(n)
    # vectorized filter: mark candidates that are self or excluded
    bad = cand == rows[:, None]
    for j in range(K):
        bad |= cand == exclude_idx[:, j][:, None]
    for i in range(n):
        keep = cand[i][~bad[i]]
        if len(keep) >= m:
            out[i] = keep[:m]
        else:
            short.append(i)
    if short:
        q2 = min(4 * q, n)
        cand2 = nn.kneighbors(Y[short], n_neighbors=q2, return_distance=False)
        for r, i in enumerate(short):
            ex = set(exclude_idx[i].tolist()) | {i}
            keep = [c for c in cand2[r] if c not in ex]
            out[i] = np.array((keep + keep)[:m])
    return out
