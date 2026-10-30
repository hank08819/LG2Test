"""SR-tSNE2 rank weight lambda on the grid of the SR-tSNE paper, eight datasets, seed 0, same t-SNE start for every lambda.
Scores: L, G (this paper), and the native criteria of SR-tSNE against the reference neighbors in rank order:
O-DLP@10 (exact-rank hit rate), nDCG@10.  lambda = 0 is the equal-budget control (KL only).
usage: python tune_sr.py <out json>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_own_lib import load
from lgcore import ranks, profile, scores
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
LAMS = [0, 5, 20, 50, 100, 200]
DS = ["Iris", "Wine", "Breast cancer", "Digits", "MNIST-5k", "Swiss roll", "S-curve", "Tree"]

def native(Y, H, k=10):
    low = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)
    low = np.stack([r[r != i][:k] for i, r in enumerate(low)]); high = H[:, :k]
    od = float((low == high).mean())
    match = low[:, :, None] == high[:, None, :]
    rel = np.where(match.any(2), k - np.argmax(match, 2), 0.0); disc = 1 / np.log2(np.arange(k) + 2.0)
    nd = float(((rel * disc).sum(1) / ((k - np.arange(k)) * disc).sum()).mean())
    return od, nd

if __name__ == "__main__":
    out = {}
    for nm in DS:
        X, y, Dref = load(nm); X = X.astype(np.float64); n = len(X); RX = ranks(Dref)
        D2 = Dref.copy(); np.fill_diagonal(D2, np.inf); H = np.argsort(D2, 1, kind="stable")[:, :10]
        from openTSNE.affinity import PerplexityBasedNN
        from srtsne2.refine_fast import refine_fast
        Y0 = TSNE(perplexity=30, init="pca", random_state=0).fit_transform(X).astype(np.float64)
        P = PerplexityBasedNN(X, perplexity=30, n_jobs=4, random_state=0, verbose=False).P
        nnX = NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X, return_distance=False); nnX = np.stack([r[r != i][:10] for i, r in enumerate(nnX)])
        def sc(Y):
            K, _, nu = profile(RX, ranks(squareform(pdist(Y)))); r = scores(K, nu, n); od, nd = native(Y, H)
            return dict(L=float(r["L"]), G=float(r["G"]), odlp=od, ndcg=nd)
        out[nm] = {"start": sc(Y0)}; print(nm, "start", out[nm]["start"], flush=True)
        for lam in LAMS:
            Y = np.asarray(refine_fast(Y0.copy(), P, nnX, lam=float(lam), m=10, n_iter=1000, method="bh", n_jobs=4, device="cpu", seed=0))
            out[nm][str(lam)] = sc(Y); print(nm, lam, {k: round(v, 4) for k, v in out[nm][str(lam)].items()}, flush=True)
            json.dump(out, open(sys.argv[1], "w"))
