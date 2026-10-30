import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr, norm, rankdata

def ranks(D):
    """rank[i, j] in 0..n-2 = position of j in i's neighbor order (self gets n-1)."""
    n = D.shape[0]; D = D.copy(); np.fill_diagonal(D, np.inf)
    O = np.argsort(D, axis=1, kind="stable")
    R = np.empty((n, n), dtype=np.int32); R[np.arange(n)[:, None], O] = np.arange(n, dtype=np.int32)[None, :]
    return R

def profile(RX, RY):
    """Q_NX(K) = DLP_K and R_NX(K) = nu_K for all K = 1..n-2, via the co-ranking matrix."""
    n = RX.shape[0]; m = n - 1
    mask = ~np.eye(n, dtype=bool)
    idx = RX[mask].astype(np.int64) * m + RY[mask]
    Q = np.bincount(idx, minlength=m * m).reshape(m, m)
    S = Q.cumsum(0).cumsum(1)
    K = np.arange(1, m)                       # 1..n-2
    QNX = S[K - 1, K - 1] / (K * n)
    RNX = ((n - 1) * QNX - K) / (n - 1 - K)
    return K, QNX, RNX

def per_point_overlap(RX, RY, k):
    """fraction of each point's k-NN kept, per point (for bootstrap)."""
    return ((RX < k) & (RY < k)).sum(1) / k

def scores(K, RNX, n, bands=None):
    bands = bands or {"G": (0.3, 0.5), "G_20_50": (0.2, 0.5), "G_40_50": (0.4, 0.5), "M": (0.1, 0.2)}
    f = K / n; out = {"L": float(RNX[K == 10][0]), "nu_half": float(RNX[K == (n - 1) // 2][0])}
    for b, (lo, hi) in bands.items():
        sel = (f >= lo) & (f <= hi); out[b] = float(RNX[sel].mean())
    w = 1.0 / K; out["AUC_lnK"] = float((RNX * w).sum() / w.sum())   # Lee et al. 2015 summary
    return out

def triplet_acc(X, Y, n_trip=20000, seed=0):
    rng = np.random.default_rng(seed); n = len(X)
    i, j, k = rng.integers(0, n, (3, n_trip)); ok = (i != j) & (i != k) & (j != k); i, j, k = i[ok], j[ok], k[ok]
    dx = np.linalg.norm(X[i] - X[j], axis=1) - np.linalg.norm(X[i] - X[k], axis=1)
    dy = np.linalg.norm(Y[i] - Y[j], axis=1) - np.linalg.norm(Y[i] - Y[k], axis=1)
    return float(np.mean(np.sign(dx) == np.sign(dy)))

def triplet_acc_D(DX, Y, n_trip=20000, seed=0):
    rng = np.random.default_rng(seed); n = len(Y)
    i, j, k = rng.integers(0, n, (3, n_trip)); ok = (i != j) & (i != k) & (j != k); i, j, k = i[ok], j[ok], k[ok]
    dx = DX[i, j] - DX[i, k]
    dy = np.linalg.norm(Y[i] - Y[j], axis=1) - np.linalg.norm(Y[i] - Y[k], axis=1)
    return float(np.mean(np.sign(dx) == np.sign(dy)))

def sheppard_prediction(DX, DY):
    """Predicted nu at k=(n-1)/2 from per-point normal-score correlations via Sheppard's formula."""
    n = DX.shape[0]; preds = []
    for i in range(n):
        a = np.delete(DX[i], i); b = np.delete(DY[i], i)
        za = norm.ppf((rankdata(a) - 0.5) / len(a)); zb = norm.ppf((rankdata(b) - 0.5) / len(b))
        r = np.corrcoef(za, zb)[0, 1]; preds.append(2 / np.pi * np.arcsin(r))
    return float(np.mean(preds))

def bootstrap_ci(v, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(v)
    bs = [v[rng.integers(0, n, n)].mean() for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
