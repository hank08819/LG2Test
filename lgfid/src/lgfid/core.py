"""Local and global fidelity of a two-dimensional map (L, G) and the LG reversal test.

nu_k = ((n-1) DLP_k - k) / (n-1-k), DLP_k = mean fraction of each point's k nearest neighbors in the data that are also among
its k nearest in the map. L = nu_10. G = mean of nu_k over 0.3 <= k/n <= 0.5.
Exact for n <= max_n (co-ranking). Above max_n: L exact over all points (k-d tree / ball tree), G and the profile from
n_sub uniform subsamples of m points.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t as tdist

BAND = (0.3, 0.5)


def _dist(A, metric):
    A = np.asarray(A, np.float64)
    return cdist(A, A, metric=metric)


def ranks(D):
    """rank[i, j] in 0..n-2 = position of j in i's neighbor order; self gets n-1."""
    n = D.shape[0]; D = D.copy(); np.fill_diagonal(D, np.inf)
    O = np.argsort(D, axis=1, kind="stable")
    R = np.empty((n, n), dtype=np.int32); R[np.arange(n)[:, None], O] = np.arange(n, dtype=np.int32)[None, :]
    return R


def _profile_from_ranks(RX, RY):
    n = RX.shape[0]; m = n - 1
    mask = ~np.eye(n, dtype=bool)
    Q = np.bincount(RX[mask].astype(np.int64) * m + RY[mask], minlength=m * m).reshape(m, m)
    S = Q.cumsum(0).cumsum(1); K = np.arange(1, m)
    dlp = S[K - 1, K - 1] / (K * n)
    return K, dlp, ((n - 1) * dlp - K) / (n - 1 - K)


def _band_mean(K, nu, n, band):
    f = K / n; sel = (f >= band[0]) & (f <= band[1]); return float(nu[sel].mean())


def _knn(A, k, metric):
    from sklearn.neighbors import NearestNeighbors
    A = np.asarray(A, np.float64)
    J = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(A).kneighbors(A, return_distance=False)
    return np.stack([r[r != i][:k] for i, r in enumerate(J)])


def _check(X, Y):
    X = np.asarray(X); Y = np.asarray(Y)
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
        raise ValueError(f"X and Y must be 2-D arrays with the same number of rows; got {X.shape} and {Y.shape}")
    if len(X) < 30:
        raise ValueError("need at least 30 points")
    return X, Y


def lg_scores(X, Y, k=10, band=BAND, metric="euclidean", map_metric="euclidean", max_n=5000, m=2000, n_sub=10, seed=0,
              return_profile=False):
    """L, G (and O-DLP@k, the share of the k nearest neighbors returned at their exact rank) of the map Y of the data X.

    X: (n, p) data, or an (n, n) distance matrix with metric="precomputed". Y: (n, d) map.
    Returns a dict with L, G, odlp, n, mode ('exact' or 'subsample'), G_sd (subsample mode), and optionally the profile."""
    X, Y = _check(X, Y); n = len(Y)
    if n <= max_n:
        DX = np.asarray(X, np.float64) if metric == "precomputed" else _dist(X, metric)
        RX, RY = ranks(DX), ranks(_dist(Y, map_metric)); K, dlp, nu = _profile_from_ranks(RX, RY)
        HX = np.argsort(RX, 1)[:, :k]; HY = np.argsort(RY, 1)[:, :k]
        out = dict(L=float(nu[K == k][0]), G=float(_band_mean(K, nu, n, band)), odlp=float((HX == HY).mean()), n=n, mode="exact")
        if return_profile: out["profile"] = dict(k_over_n=(K / n).tolist(), nu=nu.tolist(), dlp=dlp.tolist())
        return out
    if metric == "precomputed":
        raise ValueError("a precomputed distance matrix is supported up to max_n points")
    IX, IY = _knn(X, k, metric), _knn(Y, k, map_metric)
    ov = np.mean([np.intersect1d(a, b).size for a, b in zip(IX, IY)]) / k; L = ((n - 1) * ov - k) / (n - 1 - k)
    rng = np.random.default_rng(seed); Gs, prof = [], []
    for _ in range(n_sub):
        s = np.sort(rng.choice(n, m, replace=False))
        K, _, nu = _profile_from_ranks(ranks(_dist(X[s], metric)), ranks(_dist(Y[s], map_metric)))
        Gs.append(_band_mean(K, nu, m, band)); prof.append(nu)
    out = dict(L=float(L), G=float(np.mean(Gs)), G_sd=float(np.std(Gs, ddof=1)), odlp=float((IX == IY).mean()), n=n, mode="subsample", m=m, n_sub=n_sub)
    if return_profile: out["profile"] = dict(k_over_n=(np.arange(1, m - 1) / m).tolist(), nu=np.mean(prof, 0).tolist(), note=f"mean over {n_sub} subsamples of {m}")
    return out


def _pp_profile(RX, RY):
    """per-point profile nu_i(k), k = 1..n-2 (its mean over i is nu_k)."""
    n = len(RX); M = np.maximum(RX, RY).astype(np.int64); np.fill_diagonal(M, n)
    H = np.bincount((np.arange(n)[:, None] * (n + 1) + M).ravel(), minlength=n * (n + 1)).reshape(n, n + 1)[:, :n - 1]
    C = H.cumsum(1)[:, :n - 2].astype(np.float32); K = np.arange(1, n - 1, dtype=np.float32)
    return ((n - 1) * C / K - K) / (n - 1 - K)


def _jk(stat, blocks, nb):
    th = np.array([stat(blocks != b) for b in range(nb)]); full = stat(np.ones_like(blocks, bool))
    return full, float(np.sqrt((nb - 1) / nb * ((th - th.mean()) ** 2).sum()))


def _calibrate(d, blocks, nb, B, alpha, seed):
    rng = np.random.default_rng(seed); d0 = d - d.mean(); idx = [np.where(blocks == b)[0] for b in range(nb)]; q = tdist.ppf(1 - alpha, nb - 1); rej = 0
    for _ in range(B):
        pick = rng.integers(0, nb, nb); dd = np.concatenate([d0[idx[b]] for b in pick]); bb = np.concatenate([np.full(len(idx[b]), j) for j, b in enumerate(pick)])
        full, se = _jk(lambda w: dd[w].mean(), bb, nb); rej += full / se > q
    return rej / B


def _sx(dmean, n):
    w = np.where(dmean <= 0)[0]; return float((w[0] + 1) / n) if len(w) else None


def reversal_test(X, YA, YB, k=10, band=BAND, metric="euclidean", map_metric="euclidean", n_blocks=20, alpha=0.05,
                  max_n=5000, calibrate=False, n_cal=2000, seed=0):
    """LG reversal test: is map A more local (L_A > L_B) and map B more global (G_B > G_A)?

    Two one-sided tests on per-point scores with delete-a-block jackknife errors over n_blocks k-means blocks of the data
    (t with n_blocks - 1 df), joined by the intersection-union principle: the reversal is declared at level alpha when
    max(pL, pG) < alpha. Also returns the crossover scale s_x (smallest k/n at which B reaches A) with a jackknife interval.
    Above max_n points the test runs on a uniform subsample of max_n points and says so."""
    from sklearn.cluster import KMeans
    X, YA = _check(X, YA); _, YB = _check(X, YB); n0 = len(X); note = None; kk = k
    if n0 > max_n:
        s = np.sort(np.random.default_rng(seed).choice(n0, max_n, replace=False)); X, YA, YB = np.asarray(X)[s], np.asarray(YA)[s], np.asarray(YB)[s]
        note = f"test on a uniform subsample of {max_n} of {n0} points"
    n = len(X); RX = ranks(_dist(X, metric)); f = np.arange(1, n - 1) / n; sel = (f >= band[0]) & (f <= band[1])
    blocks = KMeans(n_blocks, n_init=4, random_state=seed).fit_predict(np.asarray(X, np.float64))
    NA, NB = _pp_profile(RX, ranks(_dist(YA, map_metric))), _pp_profile(RX, ranks(_dist(YB, map_metric)))
    LA, LB, GA, GB = NA[:, kk - 1], NB[:, kk - 1], NA[:, sel].mean(1), NB[:, sel].mean(1)
    dL, seL = _jk(lambda w: (LA[w] - LB[w]).mean(), blocks, n_blocks); dG, seG = _jk(lambda w: (GB[w] - GA[w]).mean(), blocks, n_blocks)
    pL, pG = float(tdist.sf(dL / seL, n_blocks - 1)), float(tdist.sf(dG / seG, n_blocks - 1))
    Dd = NA - NB; s0 = _sx(Dd.mean(0), n); sb = [_sx(Dd[blocks != b].mean(0), n) for b in range(n_blocks)]; ci = None
    if s0 is not None and all(v is not None for v in sb):
        sb = np.array(sb); se = np.sqrt((n_blocks - 1) / n_blocks * ((sb - sb.mean()) ** 2).sum()); q = tdist.ppf(0.975, n_blocks - 1)
        ci = [float(max(0.0, s0 - q * se)), float(min(1.0, s0 + q * se))]
    out = dict(L_A=float(LA.mean()), L_B=float(LB.mean()), G_A=float(GA.mean()), G_B=float(GB.mean()),
               dL=float(dL), seL=seL, pL=pL, dG=float(dG), seG=seG, pG=pG, p_reversal=max(pL, pG), reversal=bool(max(pL, pG) < alpha),
               alpha=alpha, crossover=s0, crossover_ci=ci, n=n, n_blocks=n_blocks)
    if note: out["note"] = note
    if calibrate:
        out["calibration"] = dict(L=_calibrate(LA - LB, blocks, n_blocks, n_cal, alpha, seed), G=_calibrate(GB - GA, blocks, n_blocks, n_cal, alpha, seed + 1), n_boot=n_cal)
    return out


def plot_profile(results, ax=None, labels=None, band=BAND, k=10):
    """Draw nu_k against k/n for one or more lg_scores(..., return_profile=True) results, with the L line and the G band."""
    import matplotlib.pyplot as plt
    if isinstance(results, dict): results = [results]
    labels = labels or [f"map {i + 1}" for i in range(len(results))]
    ax = ax or plt.subplots(figsize=(4, 3))[1]
    for r, lab in zip(results, labels):
        p = r["profile"]; ax.plot(p["k_over_n"], p["nu"], lw=1.2, label=f"{lab}: L={r['L']:.3f}, G={r['G']:.3f}")
    n = results[0]["n"] if results[0]["mode"] == "exact" else results[0]["m"]
    ax.axvline(k / n, color="0.5", ls=":", lw=0.8); ax.axvspan(*band, color="0.9", zorder=0)
    ax.set_xscale("log"); ax.set_xlabel("k/n"); ax.set_ylabel(r"$\nu_k$"); ax.legend(frameon=False, fontsize=7)
    return ax


def _pca_axes(X, d):
    Xc = X - X.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:d], s ** 2 / (len(X) - 1)


def pca_stability(X, d=2, frac=0.5, n_rep=10, min_cos=0.9, seed=0):
    """Is PCA a stable reference map for X? Refit the d-dimensional PCA plane on n_rep random subsets of frac*n points and
    return the cosine of the largest principal angle between each refit and the full-data plane. When the d-th and
    (d+1)-th eigenvalues are close, the plane can turn between subsets (Davis and Kahan 1970) and G of a PCA map
    fitted on a subset changes with it; compare maps against the returned full-data map in that case."""
    X = np.asarray(X, np.float64); n = len(X); rng = np.random.default_rng(seed)
    V, ev = _pca_axes(X, d + 1); V = V[:d]
    cos = []
    for _ in range(n_rep):
        Vs, _ = _pca_axes(X[rng.choice(n, max(d + 2, int(frac * n)), replace=False)], d)
        cos.append(float(np.linalg.svd(Vs @ V.T, compute_uv=False).min()))
    cos = np.array(cos)
    return {"eig_ratio": float(ev[d - 1] / ev[d]), "cos": cos.tolist(), "min_cos": float(cos.min()), "stable": bool(cos.min() >= min_cos),
            "map": (X - X.mean(0)) @ V.T}
