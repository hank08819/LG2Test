"""Supply chain: UCI Online Retail II (UK online wholesaler, Dec 2009 - Dec 2011, about 1M invoice lines).
prep : customer x product matrix of purchased quantity. Invoices starting with 'C' (cancellations), non-positive quantities,
       missing customer IDs and non-product codes (POST, M, D, DOT, BANK CHARGES, ...) are dropped; the 2,000 products bought by
       most customers are kept, and customers with at least 5 distinct of them. Entries log(1 + quantity), rows scaled to unit
       length, so the reference distance is the cosine distance between purchase profiles.
emb  : PCA, NPCA, t-SNE (PCA init), SR-tSNE2 (lambda = 5), UMAP, PaCMAP, seed 0, scored as exp_own.py (L, G, rho, triplet,
       O-DLP@10, nDCG@10); NPCA loadings saved with product descriptions.
The emb step also runs on the 8-K memory store (embed_8k.py, stem 8k) and on PBMC 3k (prep_pbmc.py, stem pbmc; NPCA on the genes, published maps scored).
usage: python exp_retail.py <out dir> prep|emb [stem: retail (default) | 8k]"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT, mode = sys.argv[1], sys.argv[2]; ST = sys.argv[3] if len(sys.argv) > 3 else "retail"; os.makedirs(OUT, exist_ok=True)
if mode == "prep":
    import pandas as pd, zipfile
    z = zipfile.ZipFile(os.path.expanduser("~/finsc/data/online_retail_ii/online_retail_ii.zip"))
    df = pd.concat(pd.read_excel(z.open("online_retail_II.xlsx"), sheet_name=None).values(), ignore_index=True)
    print("raw", df.shape, flush=True)
    df = df[~df["Invoice"].astype(str).str.startswith("C") & (df["Quantity"] > 0) & df["Customer ID"].notna()]
    df = df[df["StockCode"].astype(str).str.match(r"^\d{5}[A-Za-z]{0,2}$")]
    q = df.groupby(["Customer ID", "StockCode"])["Quantity"].sum().reset_index()
    top = q.groupby("StockCode")["Customer ID"].nunique().sort_values(ascending=False).index[:2000]
    q = q[q["StockCode"].isin(top)]
    keep = q.groupby("Customer ID")["StockCode"].nunique(); keep = keep[keep >= 5].index; q = q[q["Customer ID"].isin(keep)]
    cu = {c: i for i, c in enumerate(sorted(q["Customer ID"].unique()))}; pr = {p: j for j, p in enumerate(sorted(top, key=str))}
    X = np.zeros((len(cu), len(pr)), np.float32); X[q["Customer ID"].map(cu).values, q["StockCode"].map(pr).values] = np.log1p(q["Quantity"].values)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    desc = df.drop_duplicates("StockCode").set_index("StockCode")["Description"]
    names = np.array([f"{p} {str(desc.get(p, ''))}" for p in sorted(top, key=str)])
    ctry = df.drop_duplicates("Customer ID").set_index("Customer ID")["Country"]
    np.savez_compressed(f"{OUT}/retail.npz", X=X, names=names, country=np.array([ctry[c] for c in sorted(cu)]))
    print("matrix", X.shape, "nonzero", float((X > 0).mean()), flush=True); sys.exit()

from lgcore import ranks, profile, scores, triplet_acc_D
from exp_own_lib import npca2
from tune_sr import native
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
z = np.load(f"{OUT}/{ST}.npz", allow_pickle=True); X = z["X"].astype(np.float64); n = len(X)
Dref = squareform(pdist(X)); RX = ranks(Dref); iu = np.triu_indices(n, 1)
D2 = Dref.copy(); np.fill_diagonal(D2, np.inf); H = np.argsort(D2, 1, kind="stable")[:, :10]; del D2
res, maps = {}, {}
def run(nm, f):
    t0 = time.process_time(); Y = np.asarray(f(), np.float64); c = time.process_time() - t0
    DY = squareform(pdist(Y)); K, _, nu = profile(RX, ranks(DY)); r = scores(K, nu, n); od, nd = native(Y, H)
    r = dict(L=float(r["L"]), G=float(r["G"]), rho=float(spearmanr(DY[iu], Dref[iu]).correlation), triplet=float(triplet_acc_D(Dref, Y)), odlp=od, ndcg=nd, cpu=c)
    res[nm] = r; maps[nm] = Y.astype(np.float32); np.save(f"{OUT}/nu_{ST}_{nm}.npy", nu.astype(np.float32))
    print(nm, {k: round(v, 4) for k, v in r.items()}, flush=True)
    json.dump(res, open(f"{OUT}/{ST}.json", "w")); np.savez_compressed(f"{OUT}/{ST}_maps.npz", **{k.replace("-", "_"): v for k, v in maps.items()})
    return Y
run("PCA", lambda: PCA(2).fit_transform(X))
from npca import NPCA
XN = z["Xnpca"].astype(np.float64) if "Xnpca" in z.files else X          # pbmc: NPCA on the genes
C = np.cov(XN, rowvar=False, bias=True) * n; m = NPCA(n_components=2, alpha=0.1, n_iter=3000, device="cpu"); m.mean_ = XN.mean(0); m._fit_covariance(C)
run("NPCA", lambda: (XN - XN.mean(0)) @ m.components_)
U = m.components_; nmz = z["names"] if "names" in z.files else np.arange(U.shape[0]).astype(str); top = {f"axis{a+1}": [(str(nmz[j]), float(U[j, a])) for j in np.argsort(-U[:, a])[:10]] for a in range(2)}
json.dump(dict(top=top, nonzero=float((U > 0).mean())), open(f"{OUT}/{ST}_npca_loadings.json", "w"), indent=1)
Y0 = run("t-SNE", lambda: TSNE(perplexity=30, init="pca", random_state=0).fit_transform(X))
def sr():
    from openTSNE.affinity import PerplexityBasedNN
    from srtsne2.refine_fast import refine_fast
    P = PerplexityBasedNN(X, perplexity=30, n_jobs=6, random_state=0, verbose=False).P
    nn = NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X, return_distance=False); nn = np.stack([r[r != i][:10] for i, r in enumerate(nn)])
    return refine_fast(Y0.copy(), P, nn, lam=5.0, m=10, n_iter=1000, method="bh", n_jobs=6, device="cpu", seed=0)
run("SR-tSNE2", sr)
import umap, pacmap
run("UMAP", lambda: umap.UMAP(random_state=0).fit_transform(X))
run("PaCMAP", lambda: pacmap.PaCMAP(n_components=2, random_state=0).fit_transform(X))
for k in ("pub_UMAP", "pub_tSNE"):
    if k in z.files: run(k.replace("pub_", "published ").replace("tSNE", "t-SNE"), lambda: z[k])
print("done", flush=True)
