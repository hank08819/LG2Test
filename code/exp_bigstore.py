"""Large stores (60,000 to 1.4 million points): the six maps of the MNIST-60k experiment, scored at full size.
L and O-DLP@10 exact over all points: the 10 nearest neighbors of every point in the data (exact, chunked matrix products on GPU 0)
and in the map (k-d tree). G from R = 10 uniform subsamples of m = 2,000 points (the same draws for every map); Spearman rho and
triplet accuracy on the same subsamples; a convergence check of G over m = 1,000 to 8,000 for PCA and t-SNE. The LG reversal test
(lgfid) of t-SNE, SR-tSNE2 and UMAP against PCA on a uniform subsample of 5,000 points.
Rows of X are compared by Euclidean distance (unit rows: the cosine order).
Maps: PCA; NPCA (alpha 0.1, 3,000 steps); t-SNE (openTSNE, FFT, perplexity 30, PCA init); t-SNE-200 (same, perplexity 200); SR-tSNE2 (from that t-SNE map, FFT, lambda
from $SR_LAM (default 50, the rule of the SR-tSNE study for n >= 50,000), learning rate $SR_LR, $SR_IT iterations, m = 10); UMAP (low_memory); PaCMAP; all seed 0. A map already saved as <out>/map_<name>.npy, or given
as PRE_<name>=<path.npy> in the environment, is read instead of refit.
usage: python exp_bigstore.py <store npz with X [, label]> <out dir> [methods, comma-separated]"""
import sys, os, json, time, resource, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lgcore import ranks, profile, triplet_acc_D
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
SRC, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
METHODS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["PCA", "NPCA", "t-SNE", "SR-tSNE2", "UMAP", "PaCMAP"]
NJ = int(os.environ.get("NJOBS", 8)); LAM = float(os.environ.get("SR_LAM", 50)); SLR = float(os.environ.get("SR_LR", 10)); SIT = int(os.environ.get("SR_IT", 1000))
z = np.load(SRC, allow_pickle=True); X = np.ascontiguousarray(z["X"], dtype=np.float32); n = len(X); N = n - 1
print("store", SRC, X.shape, flush=True)

def knn_data(k=10):
    c = f"{OUT}/knnX_k{k}.npy"
    if os.path.exists(c): return np.load(c)
    import torch
    dev = torch.device("cuda:0"); Xt = torch.from_numpy(X).to(dev); sq = (Xt * Xt).sum(1); I = np.empty((n, k), np.int64); B = max(256, min(4096, int(2e9 // n)))
    for s in range(0, n, B):
        q = Xt[s:s + B]; d = q @ Xt.T; d.mul_(-2).add_(sq[None, :]).add_(sq[s:s + B, None])
        d[torch.arange(len(q), device=dev), torch.arange(s, s + len(q), device=dev)] = float("inf"); del q
        I[s:s + B] = torch.topk(d, k, dim=1, largest=False).indices.cpu().numpy()
    np.save(c, I); del Xt; torch.cuda.empty_cache(); return I
t0 = time.time(); IX = knn_data(); print("data kNN", IX.shape, round(time.time() - t0), "s", flush=True)
DRAWS = [np.sort(np.random.default_rng(1000 + r).choice(n, 2000, replace=False)) for r in range(10)]
DX = [cdist(X[s].astype(np.float64), X[s].astype(np.float64)) for s in DRAWS]; RXs = [ranks(d) for d in DX]

def gband(RX, RY, m):
    K, _, nu = profile(RX, RY); f = K / m; return float(nu[(f >= 0.3) & (f <= 0.5)].mean()), nu

def score(Y):
    Y = np.asarray(Y, np.float64)
    J = NearestNeighbors(n_neighbors=11, n_jobs=NJ).fit(Y).kneighbors(Y, return_distance=False)
    IY = np.stack([r[r != i][:10] for i, r in enumerate(J)])
    ov = np.mean([np.intersect1d(a, b).size for a, b in zip(IX, IY)]) / 10
    G, rho, trip, nus = [], [], [], []
    for s, dx, rx in zip(DRAWS, DX, RXs):
        dy = cdist(Y[s], Y[s]); g, nu = gband(rx, ranks(dy), len(s)); G.append(g); nus.append(nu)
        iu = np.triu_indices(len(s), 1); rho.append(spearmanr(dx[iu], dy[iu]).correlation); trip.append(triplet_acc_D(dx, Y[s]))
    return dict(L=float((N * ov - 10) / (N - 10)), G=float(np.mean(G)), G_sd=float(np.std(G, ddof=1)), odlp=float((IY == IX).mean()),
                rho=float(np.mean(rho)), triplet=float(np.mean(trip))), np.mean(nus, 0)

def cpu(): r = resource.getrusage(resource.RUSAGE_SELF); c = resource.getrusage(resource.RUSAGE_CHILDREN); return r.ru_utime + r.ru_stime + c.ru_utime + c.ru_stime

def fit(name, maps):
    if name == "PCA":
        from sklearn.decomposition import PCA; return PCA(2, random_state=0).fit_transform(X)
    if name == "NPCA":
        from npca import NPCA
        Xd = X.astype(np.float64); mu = Xd.mean(0); C = (Xd - mu).T @ (Xd - mu); m = NPCA(n_components=2, alpha=0.1, n_iter=3000, device="cpu")
        m.mean_ = mu; m._fit_covariance(C); np.save(f"{OUT}/npca_components.npy", m.components_); return (Xd - mu) @ m.components_
    if name == "t-SNE":
        from openTSNE import TSNE
        return np.asarray(TSNE(perplexity=30, initialization="pca", random_state=0, n_jobs=NJ, negative_gradient_method="fft", verbose=False).fit(X))
    if name == "t-SNE-200":
        from openTSNE import TSNE
        return np.asarray(TSNE(perplexity=200, initialization="pca", random_state=0, n_jobs=NJ, negative_gradient_method="fft", verbose=False).fit(X))
    if name == "SR-tSNE2":
        from openTSNE.affinity import PerplexityBasedNN
        from srtsne2.refine_fast import refine_fast
        P = PerplexityBasedNN(X, perplexity=30, n_jobs=NJ, random_state=0, verbose=False).P
        return refine_fast(maps["t-SNE"].astype(np.float64), P, IX, lam=LAM, m=10, n_iter=SIT, lr=SLR, method="fft", n_jobs=NJ, device="cpu", seed=0)
    if name == "UMAP":
        import umap; return umap.UMAP(random_state=0, low_memory=True).fit_transform(X)
    if name == "PaCMAP":
        import pacmap; return pacmap.PaCMAP(n_components=2, random_state=0).fit_transform(X)

res = json.load(open(f"{OUT}/scores.json")) if os.path.exists(f"{OUT}/scores.json") else {}; maps = {}
for nm in METHODS:
    f = f"{OUT}/map_{nm}.npy"; pre = os.environ.get("PRE_" + nm.replace("-", "_"))
    if os.path.exists(f): Y = np.load(f); c = res.get(nm, {}).get("cpu_s")
    elif pre: Y = np.load(pre)[:, :2]; c = None; np.save(f, Y.astype(np.float32))
    else:
        c0 = cpu(); t = time.time(); Y = np.asarray(fit(nm, maps), np.float32); c = cpu() - c0
        if nm == "SR-tSNE2" and res.get("t-SNE", {}).get("cpu_s"): c += res["t-SNE"]["cpu_s"]
        np.save(f, Y); print(nm, "fit", round(time.time() - t), "s wall", flush=True)
    assert np.all(np.isfinite(Y)) and len(Y) == n; maps[nm] = Y
    if nm not in res or "L" not in res[nm]:
        r, nu = score(Y); r["cpu_s"] = c; r["J"] = None if c is None else 15.0 * c; r["source"] = "given" if pre else "fit"; res[nm] = r
        np.save(f"{OUT}/nu_{nm}.npy", nu.astype(np.float32)); json.dump(res, open(f"{OUT}/scores.json", "w"), indent=1)
    print(nm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[nm].items()}, flush=True)

if "PCA" in maps and "t-SNE" in maps and not os.path.exists(f"{OUT}/conv.json"):
    conv = {}
    for m in (1000, 2000, 4000, 8000):
        for r_ in range(5):
            s = np.sort(np.random.default_rng(7000 + 13 * m + r_).choice(n, m, replace=False)); rx = ranks(cdist(X[s].astype(np.float64), X[s].astype(np.float64)))
            for nm in ("PCA", "t-SNE"):
                g, _ = gband(rx, ranks(cdist(maps[nm][s].astype(np.float64), maps[nm][s].astype(np.float64))), m); conv.setdefault(nm, {}).setdefault(str(m), []).append(g)
        print("conv", m, {k: round(float(np.mean(v[str(m)])), 4) for k, v in conv.items()}, flush=True)
    json.dump(conv, open(f"{OUT}/conv.json", "w"))

if "PCA" in maps:
    from lgfid import reversal_test
    tests = json.load(open(f"{OUT}/lgtest.json")) if os.path.exists(f"{OUT}/lgtest.json") else {}
    for A in ("t-SNE", "t-SNE-200", "SR-tSNE2", "UMAP"):
        if A in maps and A not in tests:
            tests[A] = reversal_test(X, maps[A], maps["PCA"], max_n=5000, calibrate=True, n_cal=500, seed=0); json.dump(tests, open(f"{OUT}/lgtest.json", "w"), indent=1)
            print("test", A, "vs PCA", {k: tests[A][k] for k in ("dL", "dG", "pL", "pG", "reversal", "crossover")}, flush=True)
print("done", flush=True)
