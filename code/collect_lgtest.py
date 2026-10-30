"""Table of the LG reversal test (each map against PCA) over the V7 stores: dG, pG, p_reversal, crossover; '-' where not yet run."""
import json, os, sys
R = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/LHP/PAOWN/large")
ST = ["mnist", "cifar10", "fmnist", "speech", "renal", "brain", "covertype", "agnews", "instacart", "path100k", "seaad1m", "marco1m"]
MS = ["t-SNE", "t-SNE-200", "UMAP", "SR-tSNE2"]
print("store".ljust(10), *[m.ljust(26) for m in MS])
for s in ST:
    f = f"{R}/{s}/lgtest.json"; t = json.load(open(f)) if os.path.exists(f) else {}
    row = []
    for m in MS:
        if m not in t: row.append("-".ljust(26)); continue
        r = t[m]; row.append(("%s dG %+.3f p %.3g" % ("REV" if r["reversal"] else "no ", r["dG"], max(r["pL"], r["pG"]))).ljust(26))
    print(s.ljust(10), *row)
