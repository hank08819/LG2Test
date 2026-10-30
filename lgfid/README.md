# lgfid: local and global fidelity of a map, and the LG reversal test

Give it your data and a two-dimensional map of it (PCA, t-SNE, UMAP, any method). It returns

* **L**, local fidelity: how well each point keeps its 10 nearest neighbors (chance-corrected, 0 = random map, 1 = perfect);
* **G**, global fidelity: the same statistic read at neighborhoods of 30% to 50% of the data;
* **O-DLP@10**, the share of the 10 nearest neighbors returned at their exact rank;
* the whole **profile** ν_k against k/n, from which both are read;

and, for two maps A and B, the **LG reversal test**: is A significantly more local *and* B significantly more global?
It gives a p-value with a stated error rate (two one-sided tests with block-jackknife errors, joined by the
intersection–union principle) and the **crossover scale** k/n at which B overtakes A, with a 95% interval.

## Install

```bash
pip install -e .            # from this folder; add [plot] for the profile plot
```

Requires numpy, scipy and scikit-learn.

## Command line

```bash
lgfid score data.npy map_pca.npy map_tsne.npy --plot profile.png
lgfid test  data.npy map_tsne.npy map_pca.npy          # A = t-SNE (claimed local), B = PCA (claimed global)
```

Files are `.npy`, `.npz`, `.csv` or `.tsv`, one row per point, rows in the same order in the data and the maps
(`--header` skips a header line). Use `--metric cosine` for embeddings compared by cosine distance.

On the scikit-learn digits (standardized, n = 1,797):

```
$ lgfid score data.npy map_pca.npy map_tsne.npy
map_pca.npy: L = 0.1161   G = 0.4584   O-DLP@10 = 0.0138   n = 1797  [exact]
map_tsne.npy: L = 0.5685   G = 0.3366   O-DLP@10 = 0.1422   n = 1797  [exact]

$ lgfid test data.npy map_tsne.npy map_pca.npy
A = map_tsne.npy: L = 0.5685  G = 0.3366
B = map_pca.npy: L = 0.1161  G = 0.4584
A more local : L_A - L_B = +0.4524 (SE 0.0111), one-sided p = 2.894e-20
B more global: G_B - G_A = +0.1219 (SE 0.0328), one-sided p = 0.0007427
reversal (A local, B global) DECLARED at level 0.05: p = max(pL, pG) = 0.0007427
crossover s_x = 0.196  95% CI [0.115, 0.277]
```

## Python

```python
from lgfid import lg_scores, reversal_test, plot_profile
r = lg_scores(X, Y)                     # {'L': ..., 'G': ..., 'odlp': ..., 'n': ..., 'mode': 'exact'}
t = reversal_test(X, Y_tsne, Y_pca)     # {'p_reversal': ..., 'reversal': True, 'crossover': ..., 'crossover_ci': [...], ...}
```

See `examples/quickstart.py`.

## Size

Up to `max_n = 5000` points everything is exact (memory grows as n²). Above that, L is still exact over all points
(nearest-neighbor search), and G and the profile are averaged over ten uniform subsamples of 2,000 points; the
estimate from a few hundred points is already within about 0.01 of that from thousands. The reversal test above
`max_n` runs on a uniform subsample of `max_n` points and says so.

## Definitions

ν_k = ((n−1)·DLP_k − k)/(n−1−k), where DLP_k is the mean fraction of each point's k nearest neighbors in the data that
are also among its k nearest in the map. L = ν_10, G = mean of ν_k over 0.3 ≤ k/n ≤ 0.5. Per point,
ν_i(k) uses O_i(k) = #{j : max(r^X_ij, r^Y_ij) < k}, and the means over points are L and G exactly; the test compares
the per-point scores of the two maps, with standard errors from a delete-a-group jackknife over 20 k-means blocks of the
data (t with 19 degrees of freedom). `calibrate=True` (`--calibrate`) also reports the rejection rate under a centered
null with the data's own block dependence.

## Tests

```bash
pip install pytest && pytest tests      # identity and random maps, the digits values of the paper, the test, the CLI
```

## Is PCA a stable reference?

```
lgfid pca data.npy --save pca_map.npy
```

refits the PCA plane on 10 random halves of the data and reports the smallest cosine between a refit and the full-data plane.
Below 0.9 the plane turns between samples (close second and third eigenvalues), and G of a PCA map fitted on a sample changes with it;
compare maps against the saved full-data map. In Python: `lgfid.pca_stability(X)`.
