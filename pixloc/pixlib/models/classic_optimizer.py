import logging
from typing import Tuple, Optional
import torch
from torch import Tensor

from .base_optimizer import BaseOptimizer
from .utils import masked_mean
from ..geometry import Camera, Pose
from ..geometry.optimization import optimizer_step
from ..geometry import losses  # noqa

logger = logging.getLogger(__name__)


class ClassicOptimizer(BaseOptimizer):
    default_conf = dict(
        lambda_=1e-2,
        lambda_max=1e4,
    )

    def _run(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor,
             T_init: Pose, camera: Camera, mask: Optional[Tensor] = None,
             W_ref_query: Optional[Tuple[Tensor, Tensor]] = None):

        T = T_init
        J_scaling = None
        if self.conf.normalize_features:
            F_ref = torch.nn.functional.normalize(F_ref, dim=-1)
        args = (camera, p3D, F_ref, F_query, W_ref_query)
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        lambda_ = torch.full_like(failed, self.conf.lambda_, dtype=T.dtype)
        mult = torch.full_like(lambda_, 10)
        recompute = True

        # compute the initial cost
        with torch.no_grad():
            res, valid_i, w_unc_i = self.cost_fn.residuals(T_init, *args)[:3]
            cost_i = self.loss_fn((res.detach()**2).sum(-1))[0]
            if w_unc_i is not None:
                cost_i *= w_unc_i.detach()
            valid_i &= mask
            cost_best = masked_mean(cost_i, valid_i, -1)

        for i in range(self.conf.num_iters):
            if recompute:
                res, valid, w_unc, _, J = self.cost_fn.residual_jacobian(
                        T, *args)
                if mask is not None:
                    valid &= mask
                failed = failed | (valid.long().sum(-1) < 10)  # too few points

                cost = (res**2).sum(-1)
                cost, w_loss, _ = self.loss_fn(cost)
                weights = w_loss * valid.float()
                if w_unc is not None:
                    weights *= w_unc
                if self.conf.jacobi_scaling:
                    J, J_scaling = self.J_scaling(J, J_scaling, valid)
                g, H = self.build_system(J, res, weights)

            delta = optimizer_step(g, H, lambda_.unqueeze(-1), mask=~failed)
            if self.conf.jacobi_scaling:
                delta = delta * J_scaling

            dt, dw = delta.split([3, 3], dim=-1)
            T_delta = Pose.from_aa(dw, dt)
            T_new = T_delta @ T

            # compute the new cost and update if it decreased
            with torch.no_grad():
                res = self.cost_fn.residual(T_new, *args)[0]
                cost_new = self.loss_fn((res**2).sum(-1))[0]
                cost_new = masked_mean(cost_new, valid, -1)
            accept = cost_new < cost_best
            lambda_ = lambda_ * torch.where(accept, 1/mult, mult)
            lambda_ = lambda_.clamp(max=self.conf.lambda_max, min=1e-7)
            T = Pose(torch.where(accept[..., None], T_new._data, T._data))
            cost_best = torch.where(accept, cost_new, cost_best)
            recompute = accept.any()

            self.log(i=i, T_init=T_init, T=T, T_delta=T_delta, cost=cost,
                     valid=valid, w_unc=w_unc, w_loss=w_loss, accept=accept,
                     lambda_=lambda_, H=H, J=J)

            stop = self.early_stop(i=i, T_delta=T_delta, grad=g, cost=cost)
            if self.conf.lambda_ == 0:  # Gauss-Newton
                stop |= (~recompute)
            else:  # LM saturates
                stop |= bool(torch.all(lambda_ >= self.conf.lambda_max))
            if stop:
                break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed
