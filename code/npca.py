"""Nonnegative principal component analysis (NPCA).

Reference: H. Han, Nonnegative principal component analysis for cancer molecular pattern discovery,
IEEE/ACM Transactions on Computational Biology and Bioinformatics 7, 537-549 (2010).

Problem (equation 4 of the reference): for a data matrix with gene covariance C (p x p),

    maximize over U >= 0      J(U) = 1/2 tr(U^T C U) - alpha * || I - U^T U ||_F^2

where U is p x L (one nonnegative column per component) and alpha > 0 controls how close to
orthonormal the columns are. Gradient (equation 6):  dJ/dU = C U + 4 alpha U (I - U^T U).
The problem is solved by projected gradient ascent: a gradient step followed by clipping at zero.

Every column of U is a nonnegative, sparse set of gene weights, that is, an additive gene program.
In scKQE-2, one component feeds one qubit, so each qubit then represents one gene program and its
rotation angle is the activity of that program in the cell.

Implementation notes
- C is scaled so that its largest eigenvalue is 1; alpha, the step and the number of iterations are
  given in these units (alpha = 0.1, step 0.2 halved every 500 of 3,000 iterations in the experiments).
- The start is U0 = |V|, the absolute values of the leading eigenvectors of C: deterministic, so
  the result is identical on every run. No random number is drawn anywhere.
- Columns can shrink to zero during the iterations; `active_` lists the components that survive.
- Scores: `transform(X, mode="nonnegative")` returns X U (X >= 0 and U >= 0, so scores are >= 0)
  divided by one global scale; `transform(X, mode="centered")` returns (X - mean) U with each
  column standardized. The nonnegative mode gave the best clustering in our tests.
- Runs on the GPU through PyTorch when available, otherwise on the CPU with NumPy.
"""
import numpy as np
import scipy.sparse as sp

__all__ = ["NPCA", "gene_covariance", "npca"]


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def gene_covariance(X, chunk=20000, device=None):
    """Centered gene covariance X_c^T X_c (p x p, float64) of a sparse or dense matrix X (cells x genes),
    accumulated in blocks of rows so that X is never densified at once. Returns (C, mean) as NumPy arrays."""
    X = sp.csr_matrix(X) if not sp.isspmatrix_csr(X) else X
    n, p = X.shape
    mu = np.asarray(X.mean(0)).ravel().astype(np.float64)
    torch = _torch()
    if torch is not None and (device or "cuda") != "cpu" and torch.cuda.is_available():
        dev = torch.device(device or "cuda")
        C = torch.zeros((p, p), dtype=torch.float64, device=dev)
        for s0 in range(0, n, chunk):
            B = torch.as_tensor(X[s0:s0 + chunk].toarray(), dtype=torch.float64, device=dev)
            C += B.T @ B
            del B
        C = C.cpu().numpy()
    else:
        C = np.zeros((p, p), dtype=np.float64)
        for s0 in range(0, n, chunk):
            B = X[s0:s0 + chunk].toarray().astype(np.float64)
            C += B.T @ B
    C -= n * np.outer(mu, mu)
    C = (C + C.T) / 2
    return C, mu


class NPCA:
    """Nonnegative PCA by projected gradient ascent on the gene covariance.

    Parameters
    ----------
    n_components : number of components L (columns of U).
    alpha : orthonormality penalty, in units where the largest eigenvalue of C is 1.
    n_iter : number of gradient steps.
    step : initial step size; halved every `halve_every` steps.
    device : "cuda", "cpu" or None (GPU when available).

    Attributes after `fit`
    ----------------------
    components_ : U, genes x L, nonnegative.
    mean_ : gene means used for the centered scores.
    active_ : indices of the components with at least one nonzero weight.
    nonzero_fraction_ : share of nonzero entries in U.
    orthogonality_ : largest off-diagonal entry of U^T U (0 = orthogonal columns).
    objective_ : value of J(U) at the end.
    """

    def __init__(self, n_components=132, alpha=0.1, n_iter=3000, step=0.2, halve_every=500, device=None):
        self.n_components, self.alpha, self.n_iter, self.step, self.halve_every, self.device = n_components, alpha, n_iter, step, halve_every, device

    # ------------------------------------------------------------------ fitting
    def fit(self, X):
        C, mu = gene_covariance(X, device=self.device)
        self.mean_ = mu
        self._fit_covariance(C)
        return self

    def _fit_covariance(self, C):
        L = self.n_components
        torch = _torch()
        use_gpu = torch is not None and (self.device or "cuda") != "cpu" and torch.cuda.is_available()
        if use_gpu:
            dev = torch.device(self.device or "cuda")
            Cg = torch.as_tensor(C, dtype=torch.float64, device=dev)
            w, V = torch.linalg.eigh(Cg)
            lam = float(w[-1]); Cn = Cg / lam
            U = V[:, -L:].flip(1).abs().clone()                     # deterministic start: |leading eigenvectors|
            I = torch.eye(L, dtype=torch.float64, device=dev); eta = self.step
            for it in range(self.n_iter):
                G = Cn @ U + 4 * self.alpha * U @ (I - U.T @ U)      # gradient of J (equation 6)
                U = torch.clamp(U + eta * G, min=0)                   # projected step
                if (it + 1) % self.halve_every == 0:
                    eta *= 0.5
            gram = U.T @ U
            self.objective_ = float(0.5 * torch.trace(U.T @ Cn @ U) - self.alpha * ((I - gram) ** 2).sum())
            self.orthogonality_ = float((gram - torch.diag(torch.diag(gram))).abs().max())
            U = U.cpu().numpy()
        else:
            w, V = np.linalg.eigh(C)
            lam = float(w[-1]); Cn = C / lam
            U = np.abs(V[:, ::-1][:, :L])
            I = np.eye(L); eta = self.step
            for it in range(self.n_iter):
                G = Cn @ U + 4 * self.alpha * U @ (I - U.T @ U)
                U = np.maximum(U + eta * G, 0)
                if (it + 1) % self.halve_every == 0:
                    eta *= 0.5
            gram = U.T @ U
            self.objective_ = float(0.5 * np.trace(U.T @ Cn @ U) - self.alpha * ((I - gram) ** 2).sum())
            self.orthogonality_ = float(np.abs(gram - np.diag(np.diag(gram))).max())
        self.components_ = U.astype(np.float64)
        self.active_ = np.where((U > 0).any(0))[0]
        self.nonzero_fraction_ = float((U > 0).mean())
        return self

    # ------------------------------------------------------------------ scores
    def transform(self, X, mode="nonnegative", active_only=False):
        """Component scores of the cells (cells x L, float32).
        mode="nonnegative": X U, divided by one global standard deviation (X >= 0 gives scores >= 0).
        mode="centered": (X - mean) U, each column centered and standardized.
        active_only=True drops the components that shrank to zero."""
        U = self.components_
        if active_only:
            U = U[:, self.active_]
        X = sp.csr_matrix(X) if not sp.isspmatrix_csr(X) else X
        if mode == "nonnegative":
            E = np.asarray(X @ U)
            E = E / (E.std() + 1e-12)
        elif mode == "centered":
            E = np.asarray(X @ U) - (self.mean_ @ U)[None, :]
            E = (E - E.mean(0)) / (E.std(0) + 1e-12)
        else:
            raise ValueError("mode must be 'nonnegative' or 'centered'")
        return E.astype(np.float32)

    def fit_transform(self, X, **kw):
        return self.fit(X).transform(X, **kw)

    def top_genes(self, gene_names, n=10):
        """For every component: list of its n largest-weight genes (empty list for a zero component)."""
        names = np.asarray(gene_names)
        out = []
        for q in range(self.components_.shape[1]):
            u = self.components_[:, q]
            out.append([] if not (u > 0).any() else list(names[np.argsort(-u)[:n]]))
        return out


def npca(X, n_components=132, alpha=0.1, mode="nonnegative", **kw):
    """One-call interface: returns (scores, model)."""
    m = NPCA(n_components=n_components, alpha=alpha, **kw).fit(X)
    return m.transform(X, mode=mode), m


if __name__ == "__main__":
    # self-test on a small nonnegative matrix with three planted gene programs
    rs = np.random.RandomState(0)
    n, p = 3000, 300
    prog = np.zeros((3, p)); prog[0, :60] = 1; prog[1, 60:140] = 1; prog[2, 140:200] = 1
    act = rs.gamma(2.0, 1.0, (n, 3)) * (rs.rand(n, 3) < 0.4)
    X = np.log1p(rs.poisson(act @ prog * 3 + 0.05))
    m = NPCA(n_components=6, alpha=0.1, n_iter=1500, device="cpu").fit(X)
    E = m.transform(X)
    print("components", m.components_.shape, "active", len(m.active_), "nonzero fraction", round(m.nonzero_fraction_, 3),
          "max off-diagonal", round(m.orthogonality_, 3), "scores >= 0:", bool((E >= 0).all()))
    for q in m.active_[:3]:
        u = m.components_[:, q]; top = np.argsort(-u)[:5]
        print(f"component {q}: top genes {top.tolist()}")
    E2 = NPCA(n_components=6, alpha=0.1, n_iter=1500, device="cpu").fit(X).transform(X)
    print("identical on rerun:", np.array_equal(E, E2))
