"""Does the local-global reversal grow with the size of the data? For n in a ladder of uniform subsets of one store (nested, numpy
seed env SUBSEED, default 0; the same seed drives t-SNE and the test subsample), fit PCA and t-SNE (openTSNE, FFT, perplexity env PP, default 30, PCA init, seed 0) on the n points, and score: L exact over the n points,
G from 10 subsamples of min(n, 2000) points, and the LG reversal test of t-SNE against PCA (lgfid) on min(n, 5000) points.
usage: python scale_reversal.py <store npz> <out json> <n,n,...>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lgfid import lg_scores, reversal_test
from sklearn.decomposition import PCA
from openTSNE import TSNE
SRC, OUT = sys.argv[1], sys.argv[2]; NS = [int(v) for v in sys.argv[3].split(",")]; NJ = int(os.environ.get("NJOBS", 4)); PP = float(os.environ.get("PP", 30)); SD = int(os.environ.get("SUBSEED", 0))
X0 = np.asarray(np.load(SRC)["X"], np.float32); perm = np.random.default_rng(SD).permutation(len(X0))
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for n in NS:
    if str(n) in res: continue
    X = X0[np.sort(perm[:n])]
    Yp = PCA(2, random_state=0).fit_transform(X); Yt = np.asarray(TSNE(perplexity=PP, initialization="pca", random_state=SD, n_jobs=NJ, negative_gradient_method="fft", verbose=False).fit(X))
    r = dict(n=n)
    for nm, Y in (("PCA", Yp), ("t-SNE", Yt)):
        s = lg_scores(X, Y, max_n=5000, m=2000, n_sub=10); r[nm] = {k: s[k] for k in ("L", "G", "odlp")}
    t = reversal_test(X, Yt, Yp, max_n=5000, seed=SD); r["test"] = {k: t[k] for k in ("dL", "seL", "pL", "dG", "seG", "pG", "p_reversal", "reversal", "crossover", "crossover_ci")}
    res[str(n)] = r; json.dump(res, open(OUT, "w"), indent=1)
    print(n, "PCA L %.3f G %.3f | t-SNE L %.3f G %.3f | dG %.3f pG %.3g reversal %s crossover %s" % (r["PCA"]["L"], r["PCA"]["G"], r["t-SNE"]["L"], r["t-SNE"]["G"], t["dG"], t["pG"], t["reversal"], t["crossover"]), flush=True)
print("done", flush=True)
