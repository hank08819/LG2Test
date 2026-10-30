"""Restore the result files used by the paper's figures and tables.

The published package stores the 400+ result files in a few containers:
  data/records.json   every .json result, keyed by its relative path
  data/texts.json     every .txt / .log result, keyed by its relative path
  data/arrays_*.npz   every .npy (key = path) and .npz (key = path::array) result; a sparse array is stored
                      as path::name__csr_{data,indices,indptr,shape} and restored dense; the pathology
                      thumbnails keep the 553 tiles that fig_path.py draws
Running  python3 unpack_data.py [out_dir]  writes them back to out_dir (default: .),
under results/ (this paper) and v6_inputs/ (the three smaller datasets), with the
original names, so that the scripts in code/ run unchanged."""
import os, sys, json, glob, numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "."
here = os.path.dirname(os.path.abspath(__file__))

def put(rel):
    p = os.path.join(out, rel); os.makedirs(os.path.dirname(p), exist_ok=True); return p

for rel, obj in json.load(open(f"{here}/data/records.json")).items():
    json.dump(obj, open(put(rel), "w"))
for rel, txt in json.load(open(f"{here}/data/texts.json")).items():
    open(put(rel), "w").write(txt)
n = 0; groups = {}                      # one .npz may be spread over several containers
for f in sorted(glob.glob(f"{here}/data/arrays_*.npz")):
    z = np.load(f, allow_pickle=True)
    for key in z.files:
        if "::" in key:
            rel, name = key.split("::", 1); groups.setdefault(rel, {})[name] = z[key]
        else:
            np.save(put(key), z[key]); n += 1
for rel, arrs in groups.items():
    for name in [a[:-10] for a in list(arrs) if a.endswith("__csr_data")]:
        from scipy.sparse import csr_matrix
        p = {c: arrs.pop(f"{name}__csr_{c}") for c in ("data", "indices", "indptr", "shape")}
        arrs[name] = csr_matrix((p["data"], p["indices"], p["indptr"]), shape=tuple(p["shape"])).toarray()
    np.savez(put(rel), **arrs); n += 1
print("restored json/txt/array files under", os.path.abspath(out), "arrays:", n)
