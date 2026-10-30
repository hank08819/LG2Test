"""The eight benchmark datasets of the SR-tSNE study, in the features of that study (MNIST and Fashion-MNIST raw pixels, CIFAR-10
ResNet-18 features, Speech Commands wav2vec 2.0, AG News BERT, Covertype 54 attributes, Renal-Cortex and Brain-Cortex 50 PCs),
copied read-only from its data files and saved as <out>/<stem>.npz with X (float32) and label.
usage: python prep_srbench.py <srbench dir> <out dir>"""
import sys, os, numpy as np
S, O = sys.argv[1:3]; os.makedirs(O, exist_ok=True)
SRC = {"mnist": "DLP_init_pca/mnist/mnist_full_precomputed.npz", "cifar10": "DLP_init_pca/cifar10/cifar10_cnn_features.npz",
       "fmnist": "newarms/data02/fmnist_Xnn.npz", "speech": "instance_level/data02/speechcmd_precomputed.npz",
       "renal": "sc_manyclass/data/renalcortex.npz", "brain": "sc_manyclass/data/braincx.npz",
       "covertype": "newarms/data02/covertype_Xnn.npz", "agnews": "newarms/data02/agnews_Xnn.npz"}
for st, f in SRC.items():
    p = f"{S}/{f}"
    if not os.path.exists(p) or os.path.exists(f"{O}/{st}.npz"): continue
    z = np.load(p, allow_pickle=True); lk = next((k for k in ("labels", "label", "y") if k in z.files), None)
    X = np.asarray(z["X"], np.float32); lab = np.asarray(z[lk]) if lk else np.zeros(len(X), int)
    np.savez(f"{O}/{st}.npz", X=X, label=lab); print(st, X.shape, "labels", lk, len(np.unique(lab)), "keys", z.files, flush=True)
