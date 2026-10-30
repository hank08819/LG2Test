"""k-means (k=8) of the 1M MS MARCO BGE embeddings; label the 20k plotted points; name clusters by top TF-IDF words.
usage: python marco_clusters.py <marco1m.npz> <collection.tsv> <S.npy> <out npz>"""
import sys, numpy as np, re
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
z = np.load(sys.argv[1]); X = z["X"].astype(np.float32); pid = z["pid"]; S = np.load(sys.argv[3])
km = MiniBatchKMeans(n_clusters=8, random_state=0, batch_size=8192, n_init=3).fit(X)
lab = km.labels_; rng = np.random.default_rng(0)
need = {}
for c in range(8):
    for i in rng.choice(np.where(lab == c)[0], 1500, replace=False): need[int(pid[i])] = c
txt = {c: [] for c in range(8)}
with open(sys.argv[2]) as fh:
    for line in fh:
        i, t = line.split("\t", 1); i = int(i)
        if i in need: txt[need[i]].append(t)
docs = [" ".join(txt[c]) for c in range(8)]
tv = TfidfVectorizer(stop_words="english", max_features=50000, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", sublinear_tf=True)
M = tv.fit_transform(docs).toarray(); voc = np.array(tv.get_feature_names_out())
names = [", ".join(voc[np.argsort(-M[c])[:3]]) for c in range(8)]
share = np.bincount(lab, minlength=8) / len(lab)
for c in range(8): print(c, round(share[c], 3), names[c])
np.savez(sys.argv[4], lab=lab[S], names=np.array(names), share=share)
