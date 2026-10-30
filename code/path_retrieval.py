"""Retrieval of similar tissue on the pathology maps, over every tile: for each tile, how many of its 10 nearest neighbors in the
Phikon space (cosine) are among its 10 nearest on the map. Reported per tissue class and for tumor tiles, mean with a 95% bootstrap interval.
usage: python3 path_retrieval.py <path dir> <out json>"""
import sys, json, numpy as np
from scipy.spatial import cKDTree
D, out = sys.argv[1], sys.argv[2]
z = np.load(f"{D}/path.npz", allow_pickle=True); X = z["X"].astype(np.float64); X /= np.linalg.norm(X, axis=1, keepdims=True); lab = z["label"].astype(str)
M = np.load(f"{D}/path_maps.npz")
def knn(Z):
    J = cKDTree(Z).query(Z, 11)[1]; return np.stack([r[r != i][:10] for i, r in enumerate(J)])
TX = knn(X)    # unit vectors: Euclidean order = cosine order
rng = np.random.default_rng(0); res = {}
for m in M.files:
    TY = knn(M[m].astype(np.float64)); hit = np.array([np.intersect1d(a, b).size for a, b in zip(TX, TY)])
    r = {"all": float(hit.mean())}
    for c in np.unique(lab): r[c] = float(hit[lab == c].mean())
    t = hit[lab == "TUM"]; bs = [rng.choice(t, t.size).mean() for _ in range(2000)]
    r["TUM_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    r["TUM_ge5"] = float((t >= 5).mean()); res[m] = r
    print(m, "all %.2f  tumor %.2f [%.2f, %.2f]  tumor tiles with >=5 of 10: %.0f%%" % (r["all"], r["TUM"], *r["TUM_ci"], 100 * r["TUM_ge5"]))
json.dump(res, open(out, "w"), indent=1)
