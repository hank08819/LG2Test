"""DREAMS (t-SNE regularized toward the scaled first two principal components; berenslab/DREAMS fork of openTSNE)
on the eight datasets, recipe of berenslab/dreams-experiments run_dreams_pca.py as used in the SR-tSNE paper:
reference = initialization = PCA-2 scaled to std 1e-4 on the first axis, reg_lambda = 0.15, reg_scaling norm, one dim.
Seeds 0-2, scored as exp_own.py: L, G, rho, triplet, plus O-DLP@10 and nDCG@10 against the reference neighbors.
usage: python exp_dreams.py <out json>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_own_lib import load
from lgcore import ranks, profile, scores, triplet_acc_D
from tune_sr import native
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from openTSNE import TSNE
DS = ["Iris", "Wine", "Breast cancer", "Digits", "MNIST-5k", "Swiss roll", "S-curve", "Tree"]
out = {}
for nm in DS:
    X, y, Dref = load(nm); X = X.astype(np.float32); n = len(X); RX = ranks(Dref); iu = np.triu_indices(n, 1)
    D2 = Dref.copy(); np.fill_diagonal(D2, np.inf); H = np.argsort(D2, 1, kind="stable")[:, :10]
    p2 = PCA(n_components=2, random_state=0).fit_transform(X); init = (p2 / p2[:, 0].std() * 1e-4).astype(np.float32)
    out[nm] = []
    for s in range(3):
        Y = np.asarray(TSNE(initialization=init, regularization=True, reg_lambda=0.15, reg_embedding=init, reg_scaling="norm",
                            reg_scaling_dims="one", random_state=s, n_jobs=6, verbose=False).fit(X), np.float64)
        DY = squareform(pdist(Y)); K, _, nu = profile(RX, ranks(DY)); r = scores(K, nu, n); od, nd = native(Y, H)
        r = dict(L=float(r["L"]), G=float(r["G"]), rho=float(spearmanr(DY[iu], Dref[iu]).correlation), triplet=float(triplet_acc_D(Dref, Y)), odlp=od, ndcg=nd)
        out[nm].append(r); print(nm, s, {k: round(v, 4) for k, v in r.items()}, flush=True)
        if s == 0: np.save(f"{os.path.dirname(sys.argv[1])}/nu_{nm.replace(' ', '_')}_DREAMS.npy", nu.astype(np.float32))
        json.dump(out, open(sys.argv[1], "w"))
