"""SR-tSNE2 rank weight on MNIST-60k from the same t-SNE start (maps.npz of exp_big.py), scored as exp_big.score (exact L, G on 10 subsamples).
env SR_LR (default 10), SR_IT (default 1000).
usage: python tune_sr_big.py <big dir> <lambda,...>"""
import sys, os, json, time, numpy as np
sys.argv = [sys.argv[0], sys.argv[1], "prep_only", *sys.argv[2:]]
OUT, LAMS = sys.argv[1], [float(v) for v in sys.argv[3].split(",")]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_big.py")).read().split('if mode == "maps":')[0]
g = {"__file__": os.path.abspath(__file__)}; exec(compile(src, "exp_big_head", "exec"), g)   # data, IX, score, cpu, NJ
from openTSNE.affinity import PerplexityBasedNN
from srtsne2.refine_fast import refine_fast
X, IX, score, NJ = g["X"], g["IX"], g["score"], g["NJ"]
Y0 = np.load(f"{OUT}/maps.npz")["t_SNE"].astype(np.float64)
P = PerplexityBasedNN(X, perplexity=30, n_jobs=NJ, random_state=0, verbose=False).P
from sklearn.neighbors import NearestNeighbors
def odlp(Y):
    J = NearestNeighbors(n_neighbors=11, n_jobs=NJ).fit(Y).kneighbors(Y, return_distance=False); IY = np.stack([r[r != i][:10] for i, r in enumerate(J)])
    return float((IY == IX).mean())
out = f"{OUT}/tune_sr_big.json"; res = json.load(open(out)) if os.path.exists(out) else {}
if "start" not in res: res["start"] = dict(odlp=odlp(Y0))
LR = float(os.environ.get("SR_LR", 10)); IT = int(os.environ.get("SR_IT", 1000))
for lam in LAMS:
    t0 = time.process_time(); Y = refine_fast(Y0.copy(), P, IX, lam=lam, m=10, n_iter=IT, lr=LR, method="fft", n_jobs=NJ, device="cpu", seed=0)
    r = score(np.asarray(Y)); r["odlp"] = odlp(np.asarray(Y)); r["cpu_s"] = time.process_time() - t0; res[f"lam{lam:g}" + ("" if (LR, IT) == (10.0, 1000) else f"_lr{LR:g}_it{IT}")] = r
    print(lam, {k: round(v, 4) for k, v in r.items()}, flush=True); json.dump(res, open(out, "w"), indent=1)
print("done", flush=True)
