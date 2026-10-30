"""One-sided local test on all points: is L of map A above L of map B, with L exact as in Table 5?
Per-point L(i) = ((n-1) ov_i/10 - 10)/(n-11), ov_i the overlap of the exact 10 nearest neighbors in the data (knnX_k10.npy)
and in the map (k-d tree); its mean over i is the exact L. Delete-a-block jackknife over 20 k-means blocks of the data
(fitted on 50,000 points, all points assigned), t with 19 df, as in lgfid.reversal_test; calibration by the same
centred block resampling with 500 draws.
usage: python local_full.py <large dir> <marco npz> <out json> A B"""
import sys, os, json, numpy as np
from scipy.stats import t as tdist
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from lgfid.core import _jk, _calibrate
LD, MARCO, OUT, A, B = sys.argv[1:6]
STORES = ["path100k", "seaad1m", "instacart", "marco1m", "mnist", "cifar10", "fmnist", "speech", "renal", "brain", "covertype", "agnews"]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def knnY(Y):
    J = NearestNeighbors(n_neighbors=11, n_jobs=8).fit(Y).kneighbors(Y, return_distance=False)
    return np.stack([r[r != i][:10] for i, r in enumerate(J)])
for st in STORES:
    key = f"{st}|{A}|{B}"
    if key in res: continue
    X = np.load(MARCO if st == "marco1m" else f"{LD}/{st}.npz", allow_pickle=True)["X"]; n = len(X)
    IX = np.load(f"{LD}/{st}/knnX_k10.npy")
    s = np.sort(np.random.default_rng(0).choice(n, min(n, 50000), replace=False))
    blocks = KMeans(20, n_init=4, random_state=0).fit(np.asarray(X[s], np.float64)).predict(np.asarray(X, np.float64))
    L = {}
    for m in (A, B):
        IY = knnY(np.asarray(np.load(f"{LD}/{st}/map_{m}.npy"), np.float64))
        ov = (IX[:, :, None] == IY[:, None, :]).any(2).sum(1)
        L[m] = ((n - 1) * ov / 10 - 10) / (n - 11)
    d = L[A] - L[B]; dL, seL = _jk(lambda w: d[w].mean(), blocks, 20); pL = float(tdist.sf(dL / seL, 19))
    res[key] = dict(L_A=float(L[A].mean()), L_B=float(L[B].mean()), dL=float(dL), seL=float(seL), pL=pL, n=n, n_blocks=20,
                    calibration_L=float(_calibrate(d, blocks, 20, 500, 0.05, 0)))
    json.dump(res, open(OUT, "w"), indent=1); print(key, res[key], flush=True)
print("done", flush=True)
