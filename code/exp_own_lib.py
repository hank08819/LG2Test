"""The author's two methods on the eight datasets of the paper, scored exactly as exp_paper.py scores the other ten.
  NPCA     nonnegative PCA (Han 2010, IEEE/ACM TCBB 7:537): projected gradient ascent on the covariance of the standardized data,
           two components, alpha = 0.1, 3,000 steps from |leading eigenvectors| (deterministic); centered scores.
  SR-tSNE2 the n log n refinement of t-SNE (srtsne2.refine_fast): it starts from t-SNE with PCA initialization of the same seed and
           settings as the paper (scikit-learn, perplexity 30) and adds a smooth rank loss on the 10 nearest neighbors in the input,
           lambda = 5 (n < 50,000), m = 10 intruders, 1,000 iterations, Barnes-Hut KL gradient; ten seeds.
  The t-SNE start is scored too, so that the refinement is compared with its own start under this software version.
usage: python exp_own.py <out dir> <dataset names, comma-separated>"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lgcore import ranks, profile, scores, triplet_acc_D, sheppard_prediction, per_point_overlap, bootstrap_ci
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from npca import NPCA
NSEED = 10

def load(name):                                   # identical to exp_multiscale.load used by the paper
    if name == "Iris": X, y = datasets.load_iris(return_X_y=True)
    elif name == "Wine": X, y = datasets.load_wine(return_X_y=True)
    elif name == "Breast cancer": X, y = datasets.load_breast_cancer(return_X_y=True)
    elif name == "Digits": X, y = datasets.load_digits(return_X_y=True)
    elif name == "MNIST-5k":
        from mlxtend.data import mnist_data
        X, y = mnist_data(); X = X / 255.0; return X, y.astype(int), squareform(pdist(X))
    elif name in ("Swiss roll", "S-curve"):
        if name == "Swiss roll":
            X, t = datasets.make_swiss_roll(1500, random_state=0); s = 0.5 * (t * np.sqrt(1 + t**2) + np.arcsinh(t)); T = np.c_[s, X[:, 1]]
        else:
            X, t = datasets.make_s_curve(1500, random_state=0); T = np.c_[t, X[:, 1]]
        X = StandardScaler().fit_transform(X); return X, None, squareform(pdist(T))
    elif name == "Tree":
        import phate
        X, b = phate.tree.gen_dla(n_dim=100, n_branch=10, branch_length=200, rand_multiplier=2, seed=37, sigma=4)
        D = shortest_path(kneighbors_graph(X, 10, mode="distance"), directed=False); return X, b, D
    X = StandardScaler().fit_transform(X); return X, y, squareform(pdist(X))

def npca2(X):
    C = np.cov(X, rowvar=False, bias=True) * len(X); m = NPCA(n_components=2, alpha=0.1, n_iter=3000, device="cpu")
    m.mean_ = X.mean(0); m._fit_covariance(C); U = m.components_
    return (X - X.mean(0)) @ U

def srtsne2(X, Y0, seed):
    from openTSNE.affinity import PerplexityBasedNN
    from srtsne2.refine_fast import refine_fast
    P = PerplexityBasedNN(X, perplexity=30, n_jobs=4, random_state=seed, verbose=False).P
    nn = NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X, return_distance=False)
    nn = np.stack([r[r != i][:10] for i, r in enumerate(nn)])
    return refine_fast(Y0, P, nn, lam=5.0, m=10, n_iter=1000, method="bh", n_jobs=4, device="cpu", seed=seed)

def main(OUT, names):
  os.makedirs(OUT, exist_ok=True)
  for nm in names:
      X, y, Dref = load(nm); n = len(X); X = X.astype(np.float64); RX = ranks(Dref); iu = np.triu_indices(n, 1)
      res = dict(n=n, methods={})
      def score(Y, s, tag):
          DY = squareform(pdist(Y)); RY = ranks(DY); K, QNX, RNX = profile(RX, RY); r = scores(K, RNX, n)
          r["rho"] = float(spearmanr(DY[iu], Dref[iu]).correlation); r["triplet"] = triplet_acc_D(Dref, Y)
          if y is not None and len(np.unique(y)) >= 4:
              cl = np.unique(y); CY = np.array([Y[y == c].mean(0) for c in cl]); CX = np.array([X[y == c].mean(0) for c in cl])
              r["centroid"] = float(spearmanr(pdist(CY), pdist(CX)).correlation)
          if s == 0:
              r["sheppard_pred"] = sheppard_prediction(Dref, DY); pL = per_point_overlap(RX, RY, 10)
              kg = [int(round(f * n)) for f in (0.3, 0.4, 0.5) if int(round(f * n)) <= n - 2]
              pG = np.mean([per_point_overlap(RX, RY, k) for k in kg], axis=0); base_L = 10 / (n - 1)
              r["L_boot"] = [(v - base_L) / (1 - base_L) for v in bootstrap_ci(pL)]
              bG = np.mean([k / (n - 1) for k in kg]); r["G_boot_raw"] = [(v - bG) / (1 - bG) for v in bootstrap_ci(pG)]
              np.save(f"{OUT}/nu_{nm}_{tag}.npy".replace(" ", "_"), RNX.astype(np.float32))
          return r
      t0 = time.process_time(); Y = npca2(X); r = score(Y, 0, "NPCA"); r["cpu"] = time.process_time() - t0; res["methods"]["NPCA"] = [r]
      print(nm, "NPCA", {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
      for mn in ("t-SNE (PCA init, start)", "SR-tSNE2"): res["methods"][mn] = []
      for s in range(NSEED):
          t0 = time.process_time(); Y0 = TSNE(perplexity=30, init="pca", random_state=s).fit_transform(X); c0 = time.process_time() - t0
          r0 = score(Y0, s, "tSNE_start"); r0["cpu"] = c0; res["methods"]["t-SNE (PCA init, start)"].append(r0)
          t0 = time.process_time(); Y1 = np.asarray(srtsne2(X, Y0.astype(np.float64), s)); c1 = time.process_time() - t0
          assert np.all(np.isfinite(Y1)); r1 = score(Y1, s, "SR-tSNE2"); r1["cpu"] = c0 + c1; res["methods"]["SR-tSNE2"].append(r1)
          print(nm, s, "start L %.3f G %.3f | SR-tSNE2 L %.3f G %.3f" % (r0["L"], r0["G"], r1["L"], r1["G"]), flush=True)
          json.dump(res, open(f"{OUT}/own_{nm.replace(' ', '_')}.json", "w"), default=float)
  print("done", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
