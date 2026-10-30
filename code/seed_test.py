"""Does the declaration survive a refit? t-SNE (perplexity 30, openTSNE FFT, random initialization) is refitted on a large
store with seeds 1..5, and each map is tested against the PCA map with the settings of exp_bigstore.py (subsample of 5,000,
20 blocks, 500 calibration draws, seed 0). Maps are saved as map_t-SNE_seed<s>.npy.
usage: python seed_test.py <large dir> <store> <out json>"""
import sys, os, json, numpy as np
from lgfid import reversal_test
from openTSNE import TSNE
LD, st, OUT = sys.argv[1:4]
X = np.ascontiguousarray(np.load(f"{LD}/{st}.npz", allow_pickle=True)["X"], dtype=np.float32); P = np.load(f"{LD}/{st}/map_PCA.npy")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for s in range(1, 6):
    key = f"{st}|t-SNE_seed{s}|PCA"
    if key in res: continue
    f = f"{LD}/{st}/map_t-SNE_seed{s}.npy"
    if not os.path.exists(f):
        np.save(f, np.asarray(TSNE(perplexity=30, initialization="random", random_state=s, n_jobs=8, negative_gradient_method="fft", verbose=False).fit(X)))
    res[key] = reversal_test(X, np.load(f), P, max_n=5000, calibrate=True, n_cal=500, seed=0)
    json.dump(res, open(OUT, "w"), indent=1, default=float); r = res[key]
    print(key, {k: r[k] for k in ("dL", "dG", "pL", "pG", "p_reversal", "reversal", "crossover")}, flush=True)
print("done", flush=True)
