"""Maps for the seed figure: t-SNE with random initialization and UMAP, seeds 0-4, on the digits and the tree, scored on the full
profile as in exp_paper.py. usage: python exp_seeds.py <out dir>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_own_lib import load
from lgcore import ranks, profile, scores
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
import umap
OUT = sys.argv[1]; res = {}; Ys = {}
for nm in ("Digits", "Tree"):
    X, y, Dref = load(nm); X = X.astype(np.float64); RX = ranks(Dref); n = len(X)
    for meth in ("t-SNE (random init)", "UMAP"):
        for s in range(5):
            Y = TSNE(perplexity=30, init="random", random_state=s).fit_transform(X) if meth != "UMAP" else umap.UMAP(random_state=s).fit_transform(X)
            K, _, nu = profile(RX, ranks(squareform(pdist(Y)))); r = scores(K, nu, n)
            res[f"{nm}|{meth}|{s}"] = (float(r["L"]), float(r["G"])); Ys[f"{nm}|{meth}|{s}"] = Y.astype(np.float32)
            print(nm, meth, s, "L %.3f G %.3f" % res[f"{nm}|{meth}|{s}"], flush=True)
    Ys[f"{nm}|labels"] = np.asarray(y)
json.dump(res, open(f"{OUT}/seeds.json", "w")); np.savez_compressed(f"{OUT}/seeds.npz", **{k.replace("|", "__").replace(" ", "_").replace("(", "").replace(")", ""): v for k, v in Ys.items()})
