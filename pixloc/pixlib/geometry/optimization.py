from packaging import version
import torch
import logging

logger = logging.getLogger(__name__)

if version.parse(torch.__version__) >= version.parse('1.9'):
    cholesky = torch.linalg.cholesky
else:
    cholesky = torch.cholesky


def optimizer_step(g, H, lambda_=0, mute=False, mask=None, eps=1e-6):
    """One optimization step with Gauss-Newton or Levenberg-Marquardt.
    Args:
        g: batched gradient tensor of size (..., N).
        H: batched hessian tensor of size (..., N, N).
        lambda_: damping factor for LM (use GN if lambda_=0).
        mask: denotes valid elements of the batch (optional).
    """
    if lambda_ is 0:  # noqa
        diag = torch.zeros_like(g)
    else:
        diag = H.diagonal(dim1=-2, dim2=-1) * lambda_
    H = H + diag.clamp(min=eps).diag_embed()

    if mask is not None:
        # make sure that masked elements are not singular
        H = torch.where(mask[..., None, None], H, torch.eye(H.shape[-1]).to(H))
        # set g to 0 to delta is 0 for masked elements
        g = g.masked_fill(~mask[..., None], 0.)

    H_, g_ = H.cpu(), g.cpu()
    try:
        U = cholesky(H_)
    except RuntimeError as e:
        if 'singular U' in str(e):
            if not mute:
                logger.debug(
                    'Cholesky decomposition failed, fallback to LU.')
            delta = -torch.solve(g_[..., None], H_)[0][..., 0]
        else:
            raise
    else:
        delta = -torch.cholesky_solve(g_[..., None], U)[..., 0]

    return delta.to(H.device)


def skew_symmetric(v):
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3).
    """
    z = torch.zeros_like(v[..., 0])
    M = torch.stack([
        z, -v[..., 2], v[..., 1],
        v[..., 2], z, -v[..., 0],
        -v[..., 1], v[..., 0], z,
    ], dim=-1).reshape(v.shape[:-1]+(3, 3))
    return M


def so3exp_map(w, eps: float = 1e-7):
    """Compute rotation matrices from batched twists.
    Args:
        w: batched 3D axis-angle vectors of size (..., 3).
    Returns:
        A batch of rotation matrices of size (..., 3, 3).
    """
    theta = w.norm(p=2, dim=-1, keepdim=True)
    small = theta < eps
    div = torch.where(small, torch.ones_like(theta), theta)
    W = skew_symmetric(w / div)
    theta = theta[..., None]  # ... x 1 x 1
    res = W * torch.sin(theta) + (W @ W) * (1 - torch.cos(theta))
    res = torch.where(small[..., None], W, res)  # first-order Taylor approx
    return torch.eye(3).to(W) + res


def J_normalization(x):
    """Jacobian of the L2 normalization, assuming that we normalize
       along the last dimension.
    """
    x_normed = torch.nn.functional.normalize(x, dim=-1)
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)

    Id = torch.diag_embed(torch.ones_like(x_normed))
    J = (Id - x_normed.unsqueeze(-1) @ x_normed.unsqueeze(-2))
    J = J / norm.unsqueeze(-1)
    return J
