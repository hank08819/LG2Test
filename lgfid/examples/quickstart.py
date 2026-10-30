"""Score PCA and t-SNE maps of the digits, plot their profiles, and test whether t-SNE is local and PCA global."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from lgfid import lg_scores, reversal_test, plot_profile

X = StandardScaler().fit_transform(load_digits().data)
maps = {"PCA": PCA(2).fit_transform(X), "t-SNE": TSNE(init="pca", random_state=0).fit_transform(X)}
res = {m: lg_scores(X, Y, return_profile=True) for m, Y in maps.items()}
for m, r in res.items(): print(f"{m}: L = {r['L']:.3f}, G = {r['G']:.3f}, O-DLP@10 = {r['odlp']:.3f}")
t = reversal_test(X, maps["t-SNE"], maps["PCA"])
print(f"t-SNE local and PCA global: p = {t['p_reversal']:.2g}, declared = {t['reversal']}, crossover k/n = {t['crossover']:.2f} {t['crossover_ci']}")
ax = plot_profile(list(res.values()), labels=list(res)); ax.figure.tight_layout(); ax.figure.savefig("profile_digits.png", dpi=200)
