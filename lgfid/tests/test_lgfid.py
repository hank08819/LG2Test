import json, subprocess, sys
import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from lgfid import lg_scores, reversal_test


@pytest.fixture(scope="module")
def digits():
    X = StandardScaler().fit_transform(load_digits().data)
    return X, PCA(2).fit_transform(X), TSNE(perplexity=30, init="pca", random_state=0).fit_transform(X)


def test_identity_map_is_perfect():
    X = np.random.default_rng(0).normal(size=(300, 2))
    r = lg_scores(X, X)
    assert r["L"] == pytest.approx(1.0) and r["G"] == pytest.approx(1.0) and r["odlp"] == pytest.approx(1.0)


def test_random_map_is_near_zero():
    rng = np.random.default_rng(1); X = rng.normal(size=(800, 20))
    r = lg_scores(X, rng.normal(size=(800, 2)))
    assert abs(r["L"]) < 0.02 and abs(r["G"]) < 0.02


def test_digits_pca_matches_paper(digits):
    X, Ypca, _ = digits
    r = lg_scores(X, Ypca)
    assert r["L"] == pytest.approx(0.11611, abs=1e-4) and r["G"] == pytest.approx(0.45843, abs=1e-4)


def test_reversal_on_digits(digits):
    X, Ypca, Ytsne = digits
    r = reversal_test(X, Ytsne, Ypca)
    assert r["reversal"] and r["dL"] > 0 and r["dG"] > 0 and r["crossover"] is not None
    swapped = reversal_test(X, Ypca, Ytsne)          # PCA is not the more local map
    assert not swapped["reversal"]


def test_subsample_mode_agrees_with_exact():
    rng = np.random.default_rng(2); X = rng.normal(size=(3000, 10)); Y = X[:, :2] + 0.3 * rng.normal(size=(3000, 2))
    ex = lg_scores(X, Y); sub = lg_scores(X, Y, max_n=1000, m=1000, n_sub=5)
    assert sub["mode"] == "subsample" and sub["L"] == pytest.approx(ex["L"], abs=1e-9) and sub["G"] == pytest.approx(ex["G"], abs=0.02)


def test_cli(tmp_path, digits):
    X, Ypca, Ytsne = digits
    for nm, a in (("X", X), ("pca", Ypca), ("tsne", Ytsne)): np.save(tmp_path / f"{nm}.npy", a)
    out = tmp_path / "s.json"
    subprocess.run([sys.executable, "-m", "lgfid.cli", "score", str(tmp_path / "X.npy"), str(tmp_path / "pca.npy"), "--out", str(out)], check=True)
    assert json.load(open(out))["pca.npy"]["L"] == pytest.approx(0.11611, abs=1e-4)
    p = subprocess.run([sys.executable, "-m", "lgfid.cli", "test", str(tmp_path / "X.npy"), str(tmp_path / "tsne.npy"), str(tmp_path / "pca.npy")], check=True, capture_output=True, text=True)
    assert "DECLARED" in p.stdout


def test_pca_stability():
    from lgfid import pca_stability
    rng = np.random.default_rng(0)
    sep = rng.normal(size=(2000, 6)) * np.array([5, 3, 1, 1, 1, 1])
    tie = rng.normal(size=(2000, 6)) * np.array([5, 2, 2, 1, 1, 1])
    a, b = pca_stability(sep), pca_stability(tie)
    assert a["stable"] and a["eig_ratio"] > 5 and a["map"].shape == (2000, 2)
    assert not b["stable"] and b["eig_ratio"] < 1.2
