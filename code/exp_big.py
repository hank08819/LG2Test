"""MNIST-60k (the 60,000 training digits, pixels / 255, Euclidean reference): maps of six methods and the four parameter paths.
L is exact: the 10 nearest neighbors of every point among all 60,000, in the data (faiss, flat L2) and in the map (k-d tree).
G is the estimator of Section 7: the profile on R = 10 uniform subsamples of m = 2,000 points (the same draws for every map), averaged
over 0.3 <= k/m <= 0.5; distance rho and triplet accuracy are computed on the same subsamples.
usage: python exp_big.py <out dir> maps                 -> PCA, NPCA, t-SNE, SR-tSNE2, UMAP, PaCMAP (seed 0), coordinates saved
       python exp_big.py <out dir> sweep <path> <value> <seed>
t-SNE at this size is openTSNE with FFT gradients (perplexity 30, PCA initialization unless random is stated); energy = CPU s x 15 W."""
import sys, os, json, time, resource, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lgcore import ranks, profile, triplet_acc_D
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
OUT = sys.argv[1]; mode = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
NJ = int(os.environ.get("NJOBS", 8))

def data():
    cache = f"{OUT}/mnist60k.npz"
    if os.path.exists(cache): z = np.load(cache); return z["X"], z["y"]
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    X = (X[:60000] / 255.0).astype(np.float32); y = y[:60000].astype(int); np.savez_compressed(cache, X=X, y=y); return X, y
X, y = data(); n = len(X); N = n - 1
def knnX():
    c = f"{OUT}/knnX_k10.npy"
    if os.path.exists(c): return np.load(c)
    import faiss
    faiss.omp_set_num_threads(NJ); ix = faiss.IndexFlatL2(X.shape[1]); ix.add(X); _, J = ix.search(X, 11)
    I = np.stack([r[r != i][:10] for i, r in enumerate(J)]); np.save(c, I); return I
IX = knnX()
if mode == "prep": print("prep done", X.shape, IX.shape); sys.exit()
DRAWS = [np.sort(np.random.default_rng(1000 + r).choice(n, 2000, replace=False)) for r in range(10)]

def score(Y):
    Y = np.asarray(Y, np.float64)
    J = NearestNeighbors(n_neighbors=11, n_jobs=NJ).fit(Y).kneighbors(Y, return_distance=False)
    IY = np.stack([r[r != i][:10] for i, r in enumerate(J)])
    ov = np.mean([np.intersect1d(a, b).size for a, b in zip(IX, IY)]) / 10; L = (N * ov - 10) / (N - 10)
    G, rho, trip = [], [], []
    for s in DRAWS:
        DX = cdist(X[s].astype(np.float64), X[s].astype(np.float64)); DY = cdist(Y[s], Y[s])
        K, _, nu = profile(ranks(DX), ranks(DY)); f = K / len(s); G.append(nu[(f >= 0.3) & (f <= 0.5)].mean())
        iu = np.triu_indices(len(s), 1); rho.append(spearmanr(DX[iu], DY[iu]).correlation); trip.append(triplet_acc_D(DX, Y[s]))
    return dict(L=float(L), G=float(np.mean(G)), G_sd=float(np.std(G, ddof=1)), rho=float(np.mean(rho)), triplet=float(np.mean(trip)))

def cpu(): r = resource.getrusage(resource.RUSAGE_SELF); c = resource.getrusage(resource.RUSAGE_CHILDREN); return r.ru_utime + r.ru_stime + c.ru_utime + c.ru_stime

def tsne(Xin, perplexity=30, exaggeration=None, init="pca", seed=0):
    from openTSNE import TSNE
    return np.asarray(TSNE(perplexity=perplexity, exaggeration=exaggeration, initialization=init, random_state=seed, n_jobs=NJ,
                           negative_gradient_method="fft", verbose=False).fit(Xin))

def run(name, f):
    c0 = cpu(); t0 = time.time(); Y = f(); c = cpu() - c0
    assert np.all(np.isfinite(Y)); r = score(Y); r.update(cpu_s=c, wall_s=time.time() - t0, J=15.0 * c); return Y, r

if mode == "maps":
    from sklearn.decomposition import PCA
    from npca import NPCA
    import umap, pacmap
    res = {}
    def npca2():
        Xd = X.astype(np.float64); C = np.cov(Xd, rowvar=False, bias=True) * n; m = NPCA(n_components=2, alpha=0.1, n_iter=3000, device="cpu")
        m.mean_ = Xd.mean(0); m._fit_covariance(C); return (Xd - Xd.mean(0)) @ m.components_
    def sr():
        from openTSNE.affinity import PerplexityBasedNN
        from srtsne2.refine_fast import refine_fast
        P = PerplexityBasedNN(X, perplexity=30, n_jobs=NJ, random_state=0, verbose=False).P
        return refine_fast(res_Y["t-SNE"].astype(np.float64), P, IX, lam=50.0, m=10, n_iter=1000, method="fft", n_jobs=NJ, device="cpu", seed=0)
    res_Y = {}
    for nm, f in [("PCA", lambda: PCA(2).fit_transform(X)), ("NPCA", npca2), ("t-SNE", lambda: tsne(X)), ("SR-tSNE2", sr),
                  ("UMAP", lambda: umap.UMAP(random_state=0, low_memory=True).fit_transform(X)), ("PaCMAP", lambda: pacmap.PaCMAP(n_components=2, random_state=0).fit_transform(X))]:
        Y, r = run(nm, f)
        if nm == "SR-tSNE2": r["cpu_s"] += res["t-SNE"]["cpu_s"]; r["J"] = 15.0 * r["cpu_s"]          # cumulative, the t-SNE start included
        res_Y[nm] = Y; res[nm] = r; print(nm, {k: round(v, 4) for k, v in r.items()}, flush=True)
        json.dump(res, open(f"{OUT}/maps.json", "w")); np.savez_compressed(f"{OUT}/maps.npz", y=y, **{k.replace("-", "_"): v.astype(np.float32) for k, v in res_Y.items()})
    # G from subsamples of growing size against the largest, for the t-SNE map (the estimator at this n)
    conv = {}
    for m in (250, 500, 1000, 2000, 4000):
        g = []
        for r_ in range(10):
            s = np.sort(np.random.default_rng(5000 + 17 * m + r_).choice(n, m, replace=False))
            for nm in ("PCA", "t-SNE", "UMAP"):
                K, _, nu = profile(ranks(cdist(X[s].astype(np.float64), X[s].astype(np.float64))), ranks(cdist(res_Y[nm][s].astype(np.float64), res_Y[nm][s].astype(np.float64))))
                f = K / m; conv.setdefault(nm, {}).setdefault(str(m), []).append(float(nu[(f >= 0.3) & (f <= 0.5)].mean()))
        print("conv", m, {k: round(np.mean(v[str(m)]), 4) for k, v in conv.items()}, flush=True)
        json.dump(conv, open(f"{OUT}/conv.json", "w"))
else:
    path, val, seed = sys.argv[3], float(sys.argv[4]), int(sys.argv[5]); v = int(val) if val == int(val) else val
    init = "pca" if seed == 0 else "random"
    if path == "perplexity": f = lambda: tsne(X, perplexity=v, init=init, seed=seed)
    elif path == "exaggeration": f = lambda: tsne(X, exaggeration=v, init=init, seed=seed)
    elif path == "pcak":
        from sklearn.decomposition import PCA
        f = lambda: tsne(PCA(int(v), random_state=0).fit_transform(X).astype(np.float32), init=init, seed=seed)
    elif path == "umap":
        import umap
        f = lambda: umap.UMAP(n_neighbors=int(v), random_state=seed, low_memory=True).fit_transform(X)
    Y, r = run(path, f); r.update(path=path, value=v, seed=seed)
    json.dump(r, open(f"{OUT}/sweep_{path}_{v}_s{seed}.json", "w")); print(path, v, seed, {k: round(x, 4) for k, x in r.items() if isinstance(x, float)}, flush=True)
