"""SR-tSNE2 refinement: fast KL gradient (numpy, openTSNE kernels) + autograd smooth rank term (torch)."""
import time
import numpy as np
import torch
from scipy.sparse import csr_matrix

from .fast_kl import kl_gradient_bh, kl_gradient_fft
from .intruders import mine_intruders_kdtree
from .loss import smooth_rank_loss


def refine_fast(Y_init, P, neighbors, lam, m=10, refresh=50, n_iter=1000, lr=10.0, momentum=0.8,
                tau_start=5.0, tau_end=0.05, exclude=None, method="bh", theta=0.2, n_jobs=8,
                device="cpu", verbose=False, seed=42):
    """Same objective, schedule and optimizer as the exact SR-tSNE refinement; KL part in O(n log n).

    P: scipy sparse (n,n) symmetric joint probabilities summing to one. neighbors: (n,k) high-dimensional neighbors.
    exclude: neighbors that may not be mined as intruders (defaults to `neighbors`, the paper's setting).
    """
    torch.manual_seed(seed)
    P = csr_matrix(P)
    n = len(Y_init)
    Y = np.array(Y_init, dtype=np.float64, copy=True)
    vel = np.zeros_like(Y)
    dev = torch.device(device)
    nidx_t = torch.tensor(np.asarray(neighbors, dtype=np.int64), device=dev)
    excl = np.asarray(neighbors if exclude is None else exclude, dtype=np.int64)
    kl_grad = kl_gradient_bh if method == "bh" else kl_gradient_fft
    intr = None
    t0 = time.time()
    for t in range(1, n_iter + 1):
        tau = tau_start + (tau_end - tau_start) * (t / n_iter)
        _, g = kl_grad(Y, P, theta=theta, n_jobs=n_jobs) if method == "bh" else kl_grad(Y, P, n_jobs=n_jobs)
        rank_val = 0.0
        if lam > 0 and m > 0:
            if intr is None or (t - 1) % refresh == 0:
                intr = mine_intruders_kdtree(Y, excl, m, n_jobs=n_jobs)
            Yt = torch.tensor(Y, dtype=torch.float32, device=dev, requires_grad=True)
            rank = smooth_rank_loss(Yt, nidx_t, torch.tensor(intr, device=dev), tau)
            rank.backward()
            g = g + lam * Yt.grad.detach().cpu().numpy().astype(np.float64)
            rank_val = float(rank)
        vel = momentum * vel - lr * g
        Y = Y + vel
        Y -= Y.mean(0)
        if verbose and (t == 1 or t % 100 == 0):
            print(f"  iter {t:5d}/{n_iter}  rank={rank_val:.4f}  {time.time()-t0:.0f}s", flush=True)
    return Y
