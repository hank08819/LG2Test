"""Loss terms: exact (chunked) t-SNE KL divergence and the smooth rank loss with intruders."""
import torch
from torch.utils.checkpoint import checkpoint


def _chunk_partition(Y_chunk, Y_full, start, size):
    """Contribution of one row block to Z = sum_{i != j} 1 / (1 + ||y_i - y_j||^2)."""
    W = 1.0 / (1.0 + torch.cdist(Y_chunk, Y_full).pow(2))
    rows = torch.arange(size, device=Y_chunk.device)
    return W.sum() - W[rows, rows + start].sum()


def kl_loss_chunked(Y, P_val, P_row, P_col, chunk_size):
    """t-SNE KL divergence KL(P || Q) up to the constant sum p log p.

    KL = sum_ij p_ij log(1 + d_ij^2) + log Z, with Z the sum of the Student-t kernel over all
    ordered pairs. The first sum runs over the non-zero entries of P (sparse); Z is exact and is
    evaluated in row blocks under gradient checkpointing so that memory is O(chunk_size * n).
    """
    n = Y.shape[0]
    diff = Y[P_row] - Y[P_col]
    term1 = (P_val * torch.log1p((diff ** 2).sum(-1))).sum()
    Z = Y.new_zeros(())
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        Z = Z + checkpoint(_chunk_partition, Y[s:e], Y, s, e - s, use_reentrant=False)
    return term1 + torch.log(Z)


@torch.no_grad()
def mine_intruders(Y, exclude_idx, m, chunk=4096):
    """For every point, the m nearest points in the embedding that are not in its exclusion list.

    exclude_idx holds the high-dimensional neighbors that must not count as intruders (the point
    itself is always excluded). Computed in row blocks of `chunk` to bound memory.
    """
    n = Y.shape[0]
    if m + exclude_idx.shape[1] + 1 > n:
        raise ValueError("m + number of excluded neighbors must be smaller than n - 1")
    out = torch.empty(n, m, dtype=torch.long, device=Y.device)
    all_rows = torch.arange(n, device=Y.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        D = torch.cdist(Y[s:e], Y)
        r = torch.arange(e - s, device=Y.device)
        D[r, all_rows[s:e]] = float("inf")
        D.scatter_(1, exclude_idx[s:e], float("inf"))
        out[s:e] = D.topk(m, dim=1, largest=False).indices
    return out


def soft_ranks(d2, tau):
    """Sigmoid soft rank of every candidate within its own list. d2: (n, C) squared distances."""
    diff = ((d2.unsqueeze(2) - d2.unsqueeze(1)) / tau).clamp(-50.0, 50.0)
    sig = torch.sigmoid(diff)
    eye = torch.eye(d2.shape[1], dtype=torch.bool, device=d2.device).unsqueeze(0)
    return 1.0 + sig.masked_fill(eye, 0.0).sum(dim=2)


def smooth_rank_loss(Y, nidx, intruder_idx, tau):
    """Smooth rank loss over the union of the k true neighbors and the m intruders.

    Neighbor j (0-based) carries relevance k - j; intruders carry relevance 0. The discounted
    gain of the true neighbors, divided by its ideal value, is 1 exactly when every neighbor sits
    at its own rank ahead of every intruder. The loss is 1 minus the mean of that ratio.
    """
    n, k = nidx.shape
    cand = torch.cat([nidx, intruder_idx], dim=1)
    d2 = ((Y.unsqueeze(1) - Y[cand]) ** 2).sum(-1)
    r = soft_ranks(d2, tau)
    rel = (k - torch.arange(k, dtype=Y.dtype, device=Y.device))
    dcg = (rel.unsqueeze(0) / torch.log2(1.0 + r[:, :k])).sum(dim=1)
    idcg = (rel / torch.log2(torch.arange(2, k + 2, dtype=Y.dtype, device=Y.device))).sum()
    return (1.0 - dcg / idcg).mean()
