"""DREAMS on the Online Retail II customer x product matrix (exp_retail.py prep), recipe of exp_dreams.py, seed 0.
usage: python exp_dreams_retail.py <dir> [stem: retail (default) | 8k]"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lgcore import ranks, profile, scores, triplet_acc_D
from tune_sr import native
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from openTSNE import TSNE
D = sys.argv[1]; ST = sys.argv[2] if len(sys.argv) > 2 else "retail"; X = np.load(f"{D}/{ST}.npz", allow_pickle=True)["X"].astype(np.float32); n = len(X)
Dref = squareform(pdist(X.astype(np.float64))); RX = ranks(Dref); iu = np.triu_indices(n, 1)
D2 = Dref.copy(); np.fill_diagonal(D2, np.inf); H = np.argsort(D2, 1, kind="stable")[:, :10]; del D2
p2 = PCA(n_components=2, random_state=0).fit_transform(X); init = (p2 / p2[:, 0].std() * 1e-4).astype(np.float32)
Y = np.asarray(TSNE(initialization=init, regularization=True, reg_lambda=0.15, reg_embedding=init, reg_scaling="norm",
                    reg_scaling_dims="one", random_state=0, n_jobs=6, verbose=False).fit(X), np.float64)
DY = squareform(pdist(Y)); K, _, nu = profile(RX, ranks(DY)); r = scores(K, nu, n); od, nd = native(Y, H)
r = dict(L=float(r["L"]), G=float(r["G"]), rho=float(spearmanr(DY[iu], Dref[iu]).correlation), triplet=float(triplet_acc_D(Dref, Y)), odlp=od, ndcg=nd)
print("DREAMS", {k: round(v, 4) for k, v in r.items()}, flush=True)
np.save(f"{D}/nu_{ST}_DREAMS.npy", nu.astype(np.float32)); np.save(f"{D}/{ST}_DREAMS_map.npy", Y.astype(np.float32))
res = json.load(open(f"{D}/{ST}.json")); res["DREAMS"] = r; json.dump(res, open(f"{D}/{ST}.json", "w"))
