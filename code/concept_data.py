"""Data for the concept figure (Fig. 1): the digits as in the paper (standardized, Euclidean reference), PCA and t-SNE (openTSNE,
PCA init, perplexity 30, seed 0) maps, and their full profiles, checked against the paper's L and G for these two methods."""
import numpy as np, json
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from openTSNE import TSNE
from scipy.spatial.distance import cdist
from lgcore import ranks, profile
d = load_digits(); X = StandardScaler().fit_transform(d.data); X = X[:, X.std(0) > 0]; y = d.target
Yp = PCA(2, random_state=0).fit_transform(X)
Yt = np.asarray(TSNE(perplexity=30, initialization="pca", random_state=0, n_jobs=4).fit(X))
RX = ranks(cdist(X, X, "sqeuclidean")); out = dict(X=X, y=y, Yp=Yp, Yt=Yt)
for nm, Y in (("pca", Yp), ("tsne", Yt)):
    K, _, nu = profile(RX, ranks(cdist(Y, Y, "sqeuclidean"))); f = K / len(X)
    out["K"] = K; out["nu_" + nm] = nu
    print(nm, "L %.3f G %.3f" % (nu[K == 10][0], nu[(f >= 0.3) & (f <= 0.5)].mean()))
out["RX_i"] = RX; np.savez_compressed("concept_digits.npz", **{k: v for k, v in out.items() if k != "RX_i"})
