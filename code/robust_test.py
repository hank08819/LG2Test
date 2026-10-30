"""Two robustness runs of the LG reversal test on the twelve large stores, same settings as exp_bigstore.py otherwise
(uniform subsample of 5,000 points, block-resampling calibration with 500 draws, seed 0):
  srvt   SR-tSNE2 (A) against the t-SNE map it refines (B): the local test p_L of L^SR > L^t-SNE
  blocks t-SNE (A) against PCA (B) with C = 10 and 50 k-means blocks (C = 20 is in lgtest.json)
usage: python robust_test.py <srvt|blocks> <large dir> <marco npz> <out json>"""
import sys, os, json, numpy as np
from lgfid import reversal_test
MODE, LD, MARCO, OUT = sys.argv[1:5]
STORES = ["path100k", "seaad1m", "instacart", "marco1m", "mnist", "cifar10", "fmnist", "speech", "renal", "brain", "covertype", "agnews"]
RUNS = [("SR-tSNE2", "t-SNE", 20)] if MODE == "srvt" else [("t-SNE", "PCA", 10), ("t-SNE", "PCA", 50)]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for st in STORES:
    X = None
    for A, B, C in RUNS:
        key = f"{st}|{A}|{B}|C{C}"
        if key in res: continue
        if X is None: X = np.ascontiguousarray(np.load(MARCO if st == "marco1m" else f"{LD}/{st}.npz", allow_pickle=True)["X"], dtype=np.float32)
        YA, YB = np.load(f"{LD}/{st}/map_{A}.npy"), np.load(f"{LD}/{st}/map_{B}.npy")
        res[key] = reversal_test(X, YA, YB, n_blocks=C, max_n=5000, calibrate=True, n_cal=500, seed=0)
        json.dump(res, open(OUT, "w"), indent=1, default=float)
        r = res[key]; print(key, {k: r[k] for k in ("dL", "pL", "dG", "pG", "reversal", "calibration")}, flush=True)
print("done", flush=True)
