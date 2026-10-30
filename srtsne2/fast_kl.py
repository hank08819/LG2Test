"""t-SNE KL gradient in O(n log n) via openTSNE's compiled Barnes-Hut / FFT-interpolation kernels.

Convention: returns d KL / d Y for KL(P||Q) = sum_ij p_ij log(p_ij / q_ij) with P a symmetric CSR matrix that sums to one
and q_ij the normalized Student-t kernel, i.e. the same quantity that the exact autograd loss of the reference
implementation differentiates. The openTSNE kernels return that gradient divided by four (they omit the factor 4 of
the classic t-SNE gradient), so the result is multiplied by four; `experiments/exp_gradient_check.py` verifies this
against the exact gradient (cosine > 0.99 at theta <= 0.3).
"""
import numpy as np
from scipy.sparse import csr_matrix
from openTSNE import _tsne
from openTSNE.quad_tree import QuadTree

EPS = np.finfo(np.float64).eps


def kl_gradient_bh(Y, P, theta=0.5, n_jobs=8, eval_error=False):
    """Barnes-Hut approximation. Y (n,2) float64, P scipy CSR (n,n) symmetric, sum 1."""
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    P = P.tocsr() if not isinstance(P, csr_matrix) else P
    grad = np.zeros_like(Y)
    tree = QuadTree(Y)
    sum_Q = _tsne.estimate_negative_gradient_bh(tree, Y, grad, theta=theta, dof=1.0, num_threads=n_jobs, pairwise_normalization=True)
    del tree
    sum_P, kl = _tsne.estimate_positive_gradient_nn(P.indices, P.indptr, P.data, Y, Y, grad, 1.0, n_jobs, eval_error)
    if eval_error:
        kl += sum_P * np.log(sum_Q + EPS)
    return (kl if eval_error else None), 4.0 * grad


def kl_gradient_fft(Y, P, n_jobs=8, n_interpolation_points=3, min_num_intervals=50, ints_in_interval=1.0, eval_error=False):
    """FFT-accelerated interpolation (FIt-SNE), O(n). For n above ~100k."""
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    P = P.tocsr() if not isinstance(P, csr_matrix) else P
    grad = np.zeros_like(Y)
    sum_Q = _tsne.estimate_negative_gradient_fft_2d(Y, grad, n_interpolation_points=n_interpolation_points,
                                                    min_num_intervals=min_num_intervals, ints_in_interval=ints_in_interval, dof=1.0)
    sum_P, kl = _tsne.estimate_positive_gradient_nn(P.indices, P.indptr, P.data, Y, Y, grad, 1.0, n_jobs, eval_error)
    if eval_error:
        kl += sum_P * np.log(sum_Q + EPS)
    return (kl if eval_error else None), 4.0 * grad
