"""LG reversal test: is method A more local and method B more global, with a confidence interval for the crossover scale?
Per point i the profile nu_i(k) = ((n-1) O_i(k)/k - k)/(n-1-k), O_i(k) = #{j: r^X_ij < k and r^Y_ij < k}; its mean over i is nu_k,
so L = mean_i nu_i(10) and G = mean_i of nu_i averaged over 0.3 <= k/n <= 0.5.
  H1 local : L_A > L_B          H1 global : G_B > G_A          reversal: both one-sided tests reject (intersection-union test, level alpha).
Standard errors: delete-a-block jackknife over 20 k-means blocks of the data (df 19); one-sided t p-values.
Crossover s^x = smallest k/n with nu^B_k >= nu^A_k (Eq. crossover); jackknife 95% interval; 'none' if B never reaches A.
Datasets: all eight paper datasets plus the supply-chain and 8-K stores. A in {t-SNE (PCA init), SR-tSNE2, UMAP}, B = PCA, seed 0.
Calibration: for each comparison the per-point differences are centered (a true null of zero mean difference with the data's own
block dependence), blocks are resampled with replacement 2,000 times, and the share of one-sided rejections at 0.05 is recorded.
Per-point L_i, G_i and the blocks are saved for every dataset (lgpts_<dataset>.npz).
usage: python lgtest.py <out json> <retail dir> <8k dir> <pbmc dir>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_own_lib import load
from lgcore import ranks
from scipy.spatial.distance import pdist, squareform
from scipy.stats import t as tdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
OUT, RD, KD, PD = sys.argv[1:5]; NB = 20; NCAL = 2000

def pp_profile(RX, RY):
    n = len(RX); M = np.maximum(RX, RY).astype(np.int64); np.fill_diagonal(M, n)
    H = np.bincount((np.arange(n)[:, None] * (n + 1) + M).ravel(), minlength=n * (n + 1)).reshape(n, n + 1)[:, :n - 1]
    C = H.cumsum(1)[:, :n - 2].astype(np.float32); K = np.arange(1, n - 1, dtype=np.float32)
    return ((n - 1) * C / K - K) / (n - 1 - K)            # n x (n-2): nu_i(k), k = 1..n-2

def jk(stat, blocks):
    th = np.array([stat(blocks != b) for b in range(NB)]); full = stat(np.ones_like(blocks, bool))
    return full, np.sqrt((NB - 1) / NB * ((th - th.mean()) ** 2).sum()), th

def tstat(d, blocks):
    full, se, _ = jk(lambda w: d[w].mean(), blocks); return full / se

def calibrate(d, blocks, seed=0):
    rng = np.random.default_rng(seed); d0 = d - d.mean(); idx = [np.where(blocks == b)[0] for b in range(NB)]; q = tdist.ppf(0.95, NB - 1); rej = 0
    for _ in range(NCAL):
        pick = rng.integers(0, NB, NB); dd = np.concatenate([d0[idx[b]] for b in pick]); bb = np.concatenate([np.full(len(idx[b]), j) for j, b in enumerate(pick)])
        rej += tstat(dd, bb) > q
    return rej / NCAL

def sx(dmean, n):                                          # dmean(k) = nu^A - nu^B; first k where B reaches A
    w = np.where(dmean <= 0)[0]; return float((w[0] + 1) / n) if len(w) else None

def embed(X, name, seed=0):
    if name == "PCA": return PCA(2).fit_transform(X)
    if name == "t-SNE": return TSNE(perplexity=30, init="pca", random_state=seed).fit_transform(X)
    if name == "UMAP":
        import umap; return umap.UMAP(random_state=seed).fit_transform(X)

def srt(X, Y0, seed=0):
    from openTSNE.affinity import PerplexityBasedNN
    from srtsne2.refine_fast import refine_fast
    P = PerplexityBasedNN(X, perplexity=30, n_jobs=6, random_state=seed, verbose=False).P
    nn = NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X, return_distance=False); nn = np.stack([r[r != i][:10] for i, r in enumerate(nn)])
    lam = 5.0 if len(X) < 50000 else 50.0
    return refine_fast(Y0.astype(np.float64), P, nn, lam=lam, m=10, n_iter=1000, method="bh", n_jobs=6, device="cpu", seed=seed)

DS = ["Iris", "Wine", "Breast cancer", "Digits", "MNIST-5k", "Swiss roll", "S-curve", "Tree", "PBMC 3k", "Pathology", "Supply chain", "8-K memory"]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for nm in DS:
    if nm in res: continue
    if nm in ("Supply chain", "8-K memory", "PBMC 3k", "Pathology"):
        D, st = {"Supply chain": (RD, "retail"), "8-K memory": (KD, "8k"), "PBMC 3k": (PD, "pbmc"), "Pathology": (os.path.join(os.path.dirname(PD), "path"), "path")}[nm]
        X = np.load(f"{D}/{st}.npz", allow_pickle=True)["X"].astype(np.float64); Dref = squareform(pdist(X))
        z = np.load(f"{D}/{st}_maps.npz"); Ys = {"PCA": z["PCA"], "t-SNE": z["t_SNE"], "SR-tSNE2": z["SR_tSNE2"], "UMAP": z["UMAP"]}
    else:
        X, y, Dref = load(nm); X = X.astype(np.float64)
        Ys = {m: embed(X, m) for m in ("PCA", "t-SNE", "UMAP")}; Ys["SR-tSNE2"] = srt(X, Ys["t-SNE"])
    n = len(X); RX = ranks(Dref); f = np.arange(1, n - 1) / n; band = (f >= 0.3) & (f <= 0.5)
    blocks = KMeans(NB, n_init=4, random_state=0).fit_predict(X)
    NU = {m: pp_profile(RX, ranks(squareform(pdist(np.asarray(Y, np.float64))))) for m, Y in Ys.items()}
    B = NU["PCA"]; Li_B, Gi_B = B[:, 9], B[:, band].mean(1); res[nm] = dict(n=n); pts = dict(blocks=blocks, PCA_L=Li_B, PCA_G=Gi_B)
    for A in ("t-SNE", "SR-tSNE2", "UMAP"):
        Li, Gi = NU[A][:, 9], NU[A][:, band].mean(1); Dd = NU[A] - B
        dL, seL, _ = jk(lambda w: (Li[w] - Li_B[w]).mean(), blocks)
        dG, seG, _ = jk(lambda w: (Gi_B[w] - Gi[w]).mean(), blocks)
        pL, pG = float(tdist.sf(dL / seL, NB - 1)), float(tdist.sf(dG / seG, NB - 1))
        s0 = sx(Dd.mean(0), n); sb = [sx(Dd[blocks != b].mean(0), n) for b in range(NB)]
        if s0 is not None and all(v is not None for v in sb):
            sb = np.array(sb); se = np.sqrt((NB - 1) / NB * ((sb - sb.mean()) ** 2).sum()); q = tdist.ppf(0.975, NB - 1)
            ci = [max(0.0, s0 - q * se), min(1.0, s0 + q * se)]
        else: ci = None
        calL, calG = calibrate(Li - Li_B, blocks), calibrate(Gi_B - Gi, blocks)
        pts[f"{A}_L"], pts[f"{A}_G"] = Li, Gi
        r = dict(calL=calL, calG=calG, dL=float(dL), seL=float(seL), pL=pL, dG=float(dG), seG=float(seG), pG=pG, reversal=bool(max(pL, pG) < 0.05),
                 p_reversal=max(pL, pG), sx=s0, sx_ci=ci, sx_none_blocks=int(sum(v is None for v in [sx(Dd[blocks != b].mean(0), n) for b in range(NB)])))
        res[nm][A] = r
        print(nm, A, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    np.savez_compressed(f"{os.path.dirname(OUT)}/lgpts_{nm.replace(' ', '_')}.npz", **pts)
    json.dump(res, open(OUT, "w"), indent=1); del NU, RX
print("done", flush=True)
