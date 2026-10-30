"""Maps for Fig. 2 (rows 3-4): PCA, NPCA, t-SNE (PCA init), SR-tSNE2 and UMAP, seed 0, on MNIST-5k and the Swiss roll, with the same
settings as exp_paper.py and exp_own.py; saves the 2-d coordinates, labels or the roll parameter, and L, G of each map.
usage: python emb_maps.py <out dir> <dataset>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_own_lib import load, npca2, srtsne2
from lgcore import ranks, profile, scores
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets
import umap
OUT, nm = sys.argv[1], sys.argv[2]
X, y, Dref = load(nm); X = X.astype(np.float64); n = len(X); RX = ranks(Dref)
if y is None: _, y = datasets.make_swiss_roll(1500, random_state=0) if nm == "Swiss roll" else datasets.make_s_curve(1500, random_state=0)
Y = {"PCA": PCA(2).fit_transform(X), "NPCA": npca2(X)}
Y["t-SNE"] = TSNE(perplexity=30, init="pca", random_state=0).fit_transform(X)
Y["SR-tSNE2"] = np.asarray(srtsne2(X, Y["t-SNE"].astype(np.float64), 0))
Y["UMAP"] = umap.UMAP(random_state=0).fit_transform(X)
LG = {}
for k, v in Y.items():
    K, _, nu = profile(RX, ranks(squareform(pdist(v)))); r = scores(K, nu, n); LG[k] = (float(r["L"]), float(r["G"])); print(nm, k, "L %.3f G %.3f" % LG[k], flush=True)
np.savez_compressed(f"{OUT}/maps_{nm.replace(' ', '_')}.npz", y=np.asarray(y), **{k.replace("-", "_"): v for k, v in Y.items()})
json.dump(LG, open(f"{OUT}/maps_{nm.replace(' ', '_')}.json", "w"))
