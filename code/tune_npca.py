"""NPCA orthonormality penalty alpha on a grid up to 0.1 (the largest value used), on the eight datasets, scored as exp_own.py scores NPCA.
usage: python tune_npca.py <out json>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_own_lib import load
from lgcore import ranks, profile, scores
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from npca import NPCA
ALPHAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
DS = ["Iris", "Wine", "Breast cancer", "Digits", "MNIST-5k", "Swiss roll", "S-curve", "Tree"]
out = {}
for nm in DS:
    X, y, Dref = load(nm); X = X.astype(np.float64); n = len(X); RX = ranks(Dref); iu = np.triu_indices(n, 1)
    C = np.cov(X, rowvar=False, bias=True) * n
    for a in ALPHAS:
        m = NPCA(n_components=2, alpha=a, n_iter=3000, device="cpu"); m.mean_ = X.mean(0); m._fit_covariance(C)
        Y = (X - X.mean(0)) @ m.components_; DY = squareform(pdist(Y))
        K, _, nu = profile(RX, ranks(DY)); r = scores(K, nu, n)
        r = dict(L=float(r["L"]), G=float(r["G"]), rho=float(spearmanr(DY[iu], Dref[iu]).correlation),
                 ortho=m.orthogonality_, nonzero=m.nonzero_fraction_)
        out.setdefault(nm, {})[str(a)] = r; print(nm, a, {k: round(v, 4) for k, v in r.items()}, flush=True)
    json.dump(out, open(sys.argv[1], "w"))
