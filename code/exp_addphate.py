"""Add PHATE (default settings, seed 0) to a store already run by exp_retail.py: scores, profile and map, same scoring.
usage: python exp_addphate.py <dir> <stem>"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lgcore import ranks, profile, scores, triplet_acc_D
from tune_sr import native
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import phate
D, ST = sys.argv[1:3]; X = np.load(f"{D}/{ST}.npz", allow_pickle=True)["X"].astype(np.float64); n = len(X)
Dref = squareform(pdist(X)); RX = ranks(Dref); iu = np.triu_indices(n, 1)
D2 = Dref.copy(); np.fill_diagonal(D2, np.inf); H = np.argsort(D2, 1, kind="stable")[:, :10]; del D2
t0 = time.process_time(); Y = np.asarray(phate.PHATE(random_state=0, n_jobs=6, verbose=0).fit_transform(X), np.float64); c = time.process_time() - t0
DY = squareform(pdist(Y)); K, _, nu = profile(RX, ranks(DY)); r = scores(K, nu, n); od, nd = native(Y, H)
r = dict(L=float(r["L"]), G=float(r["G"]), rho=float(spearmanr(DY[iu], Dref[iu]).correlation), triplet=float(triplet_acc_D(Dref, Y)), odlp=od, ndcg=nd, cpu=c)
print("PHATE", {k: round(v, 4) for k, v in r.items()}, flush=True)
np.save(f"{D}/nu_{ST}_PHATE.npy", nu.astype(np.float32)); np.save(f"{D}/{ST}_PHATE_map.npy", Y.astype(np.float32))
res = json.load(open(f"{D}/{ST}.json")); res["PHATE"] = r; json.dump(res, open(f"{D}/{ST}.json", "w"))
