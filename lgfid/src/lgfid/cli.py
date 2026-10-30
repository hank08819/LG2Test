"""Command line: lgfid score DATA MAP [MAP ...]   |   lgfid test DATA MAP_A MAP_B   |   lgfid pca DATA [--save MAP.npy]
DATA and MAP are .npy, .csv or .tsv files with one row per point (no header row, or a header with --header)."""
import argparse, json, os, sys
import numpy as np
from .core import lg_scores, reversal_test, plot_profile, pca_stability


def load(path, header):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy": return np.load(path)
    if ext == ".npz": z = np.load(path); return z[z.files[0]]
    return np.loadtxt(path, delimiter="\t" if ext == ".tsv" else ",", skiprows=1 if header else 0)


def main(argv=None):
    p = argparse.ArgumentParser(prog="lgfid", description="Local (L) and global (G) fidelity of maps, and the LG reversal test.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("score", "test"):
        s = sub.add_parser(name)
        s.add_argument("data"); s.add_argument("maps", nargs="+")
        s.add_argument("--metric", default="euclidean", help="distance in the data: euclidean (default), cosine, ...")
        s.add_argument("--k", type=int, default=10); s.add_argument("--band", type=float, nargs=2, default=(0.3, 0.5))
        s.add_argument("--max-n", type=int, default=5000); s.add_argument("--seed", type=int, default=0)
        s.add_argument("--header", action="store_true"); s.add_argument("--out", help="write the results as JSON")
    sub.choices["score"].add_argument("--plot", help="save the profile plot (png or pdf)")
    t = sub.choices["test"]; t.add_argument("--blocks", type=int, default=20); t.add_argument("--alpha", type=float, default=0.05)
    t.add_argument("--calibrate", action="store_true", help="also estimate the rejection rate under a centered null (slow)")
    q = sub.add_parser("pca", help="check whether PCA is a stable reference map; optionally save the full-data PCA map")
    q.add_argument("data"); q.add_argument("--reps", type=int, default=10); q.add_argument("--frac", type=float, default=0.5)
    q.add_argument("--seed", type=int, default=0); q.add_argument("--header", action="store_true"); q.add_argument("--save", help="write the full-data PCA map (.npy)")
    a = p.parse_args(argv)
    if a.cmd == "pca":
        r = pca_stability(load(a.data, a.header), frac=a.frac, n_rep=a.reps, seed=a.seed)
        print(f"lambda_2 / lambda_3 = {r['eig_ratio']:.3f}; cos of the largest angle between subset and full-data PCA planes: min {r['min_cos']:.3f} over {a.reps} subsets")
        print("PCA plane stable" if r["stable"] else "PCA plane turns between subsets: compare maps against the full-data PCA map (--save)")
        if a.save: np.save(a.save, r["map"])
        return
    X = load(a.data, a.header); maps = [load(m, a.header) for m in a.maps]
    if a.cmd == "score":
        res = {os.path.basename(m): lg_scores(X, Y, k=a.k, band=tuple(a.band), metric=a.metric, max_n=a.max_n, seed=a.seed, return_profile=bool(a.plot))
               for m, Y in zip(a.maps, maps)}
        for nm, r in res.items():
            extra = f"  (G sd over subsamples {r['G_sd']:.4f})" if r["mode"] == "subsample" else ""
            print(f"{nm}: L = {r['L']:.4f}   G = {r['G']:.4f}   O-DLP@{a.k} = {r['odlp']:.4f}   n = {r['n']}  [{r['mode']}]{extra}")
        if a.plot:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            ax = plot_profile(list(res.values()), labels=list(res)); ax.figure.tight_layout(); ax.figure.savefig(a.plot, dpi=200)
            for r in res.values(): r.pop("profile", None)
    else:
        if len(maps) != 2: sys.exit("test needs exactly two maps: MAP_A (claimed more local) and MAP_B (claimed more global)")
        res = reversal_test(X, maps[0], maps[1], k=a.k, band=tuple(a.band), metric=a.metric, n_blocks=a.blocks, alpha=a.alpha, max_n=a.max_n, calibrate=a.calibrate, seed=a.seed)
        A, B = (os.path.basename(m) for m in a.maps)
        print(f"A = {A}: L = {res['L_A']:.4f}  G = {res['G_A']:.4f}\nB = {B}: L = {res['L_B']:.4f}  G = {res['G_B']:.4f}")
        print(f"A more local : L_A - L_B = {res['dL']:+.4f} (SE {res['seL']:.4f}), one-sided p = {res['pL']:.4g}")
        print(f"B more global: G_B - G_A = {res['dG']:+.4f} (SE {res['seG']:.4f}), one-sided p = {res['pG']:.4g}")
        verdict = "DECLARED" if res["reversal"] else "not declared"
        print(f"reversal (A local, B global) {verdict} at level {a.alpha}: p = max(pL, pG) = {res['p_reversal']:.4g}")
        cx = res["crossover"]
        print("crossover: B never reaches A" if cx is None else f"crossover s_x = {cx:.3f}" + (f"  95% CI [{res['crossover_ci'][0]:.3f}, {res['crossover_ci'][1]:.3f}]" if res["crossover_ci"] else ""))
        if "note" in res: print("note:", res["note"])
        if "calibration" in res: print(f"calibration: rejection rate under the null {res['calibration']['L']:.3f} (L), {res['calibration']['G']:.3f} (G)")
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
