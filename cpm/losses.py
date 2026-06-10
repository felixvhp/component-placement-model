import torch


def _log_det(K, n, eps=1e-6):
    try:
        K = K + torch.eye(n, device=K.device) * eps
        Lc = torch.linalg.cholesky(K)
        if torch.isnan(Lc).any():
            raise RuntimeError("Cholesky produced NaN on CUDA")
        return -2 * torch.log(torch.diagonal(Lc)).sum() / n
    except Exception:
        # CPU eigvalsh fallback in double precision, with escalating jitter for
        # ill-conditioned / near-singular K (e.g. when samples are near-identical).
        Kc = K.detach().cpu().double()
        I = torch.eye(n, dtype=Kc.dtype)
        for jitter in (eps, 1e-4, 1e-2, 1e-1):
            try:
                ev = torch.linalg.eigvalsh(Kc + I * jitter)
                ev = torch.clamp(ev, min=eps)
                return (-(torch.log(ev).sum() / n)).to(K.device, dtype=K.dtype)
            except Exception:
                continue
        # Last resort: degenerate kernel (no diversity) → log-det of ~I.
        return torch.zeros((), device=K.device, dtype=K.dtype)


# -------------------------------------------------
#            Mutual Information Loss
# -------------------------------------------------


def mi_loss(latent_codes, latent_code_guess, temperature=0.1):
    """InfoNCE mutual-information loss between true and predicted latent codes."""
    batch_size = latent_codes.shape[0]

    latent_codes = torch.nn.functional.normalize(latent_codes, dim=1)
    latent_code_guess = torch.nn.functional.normalize(latent_code_guess, dim=1)

    sim_matrix = torch.matmul(latent_code_guess, latent_codes.T) / temperature
    labels = torch.arange(batch_size, device=latent_codes.device)
    return torch.nn.functional.cross_entropy(sim_matrix, labels)


# -------------------------------------------------
#            DPP Losses
# -------------------------------------------------


def DPP_loss(x, sigma=1, eps=1e-6):
    """
    Determinantal Point Process loss over the batch.
    Encourages diverse box configurations across the batch.
    x : [batch_size, seq_len, box_dim]
    """
    batch_size, seq_len, _ = x.shape
    assert seq_len > 1, "seq_len must be > 1 for DPP loss"

    x_flat = x.reshape(batch_size, -1)

    mean, std = x_flat.mean(dim=0), x_flat.std(dim=0)
    x_norm = (x_flat - mean) / (std + 1e-6)

    D2 = torch.cdist(x_norm, x_norm, compute_mode="donot_use_mm_for_euclid_dist").pow(2)
    K = torch.exp(-D2 / (2 * sigma**2))

    return _log_det(K, batch_size, eps)


def DPP_AR_loss(x, sigma=1, eps=1e-6):
    """DPP loss on flattened box sequences (e.g. aspect ratios)."""
    x = x.view(x.size(0), -1)
    D2 = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist").pow(2)
    K = torch.exp(-D2 / (2 * sigma**2))
    return _log_det(K, K.shape[0], eps)


def DPP_loss_adjacency(A, sigma=1, eps=1e-6):
    """DPP loss on adjacency matrices."""
    batch_size, n1, n2 = A.shape
    assert n1 == n2, "A must be square"
    A_flat = A.reshape(batch_size, -1)
    D2 = torch.cdist(A_flat, A_flat, compute_mode="donot_use_mm_for_euclid_dist").pow(2)
    K = torch.exp(-D2 / (2 * sigma**2))
    return _log_det(K, batch_size, eps)


def DPP_loss_adjacency_normalized(A, sigma=1, eps=1e-6):
    """Scale-invariant variant of DPP_loss_adjacency.

    Normalizes pairwise distance by N^2 so the metric is comparable across
    sequence lengths. Use at eval time only.
    """
    batch_size, n1, n2 = A.shape
    assert n1 == n2, "A must be square"
    A_flat = A.reshape(batch_size, -1)
    D2 = torch.cdist(A_flat, A_flat).pow(2) / (n1 * n2)
    K = torch.exp(-D2 / (2 * sigma**2))
    return _log_det(K, batch_size, eps)


# -------------------------------------------------
#  Performance-Augmented DPP (PA-DPP)
#  Follows wing_generator.py: L = K * outer(q, q)
# -------------------------------------------------


def _pa_log_det(K, q, n, eps=1e-6):
    global _cholesky_fails
    K = K + torch.eye(n, device=K.device) * eps
    L = K * torch.outer(q, q)
    try:
        Lc = torch.linalg.cholesky(L)
        if torch.isnan(Lc).any():
            raise RuntimeError("Cholesky produced NaN on CUDA")
        return -2 * torch.log(torch.diagonal(Lc)).sum() / n
    except Exception:
        ev = torch.linalg.eigvalsh(L.cpu())
        ev = torch.clamp(ev, min=eps)
        return -(torch.log(ev).sum() / n).to(K.device)


def PA_DPP_AR_loss(x, q, sigma=1, eps=1e-6):
    n = x.size(0)
    x = x.view(n, -1)
    D2 = torch.cdist(x, x).pow(2)
    K = torch.exp(-D2 / (2 * sigma**2))
    return _pa_log_det(K, q, n, eps)


def PA_DPP_loss_adjacency(A, q, sigma=1, eps=1e-6):
    batch_size, n1, n2 = A.shape
    assert n1 == n2, "A must be square"
    A_flat = A.reshape(batch_size, -1)
    D2 = torch.cdist(A_flat, A_flat).pow(2)
    K = torch.exp(-D2 / (2 * sigma**2))
    return _pa_log_det(K, q, batch_size, eps)
