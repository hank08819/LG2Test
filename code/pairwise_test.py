"""LG reversal test between two neighbor embeddings (no PCA): A = UMAP or t-SNE (claimed more local), B = PaCMAP (claimed more global),
on every large store; same settings as the tests against PCA in exp_bigstore.py (uniform subsample of 5,000 points, 20 k-means blocks,
block-resampling calibration with 500 draws, seed 0).
usage: python pairwise_test.py <large dir> <marco npz> <out json>"""
import sys, os, json, numpy as np
from lgfid import reversal_test
LD, MARCO, OUT = sys.argv[1:4]
STORES = ["path100k", "seaad1m", "instacart", "marco1m", "mnist", "cifar10", "fmnist", "speech", "renal", "brain", "covertype", "agnews"]
PAIRS = [("UMAP", "PaCMAP"), ("t-SNE", "PaCMAP")]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for st in STORES:
    src = MARCO if st == "marco1m" else f"{LD}/{st}.npz"
    X = None
    for A, B in PAIRS:
        key = f"{st}|{A}|{B}"
        if key in res: continue
        if X is None: X = np.ascontiguousarray(np.load(src, allow_pickle=True)["X"], dtype=np.float32)
        YA, YB = np.load(f"{LD}/{st}/map_{A}.npy"), np.load(f"{LD}/{st}/map_{B}.npy")
        res[key] = reversal_test(X, YA, YB, max_n=5000, calibrate=True, n_cal=500, seed=0)
        json.dump(res, open(OUT, "w"), indent=1, default=float)
        r = res[key]; print(key, {k: r[k] for k in ("dL", "dG", "pL", "pG", "p_reversal", "reversal", "crossover")}, flush=True)
print("done", flush=True)
