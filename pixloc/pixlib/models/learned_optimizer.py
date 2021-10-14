import logging
from typing import Tuple, Optional
import torch
from torch import nn, Tensor

from .base_optimizer import BaseOptimizer
from ..geometry import Camera, Pose
from ..geometry.optimization import optimizer_step
from ..geometry import losses  # noqa

logger = logging.getLogger(__name__)


class DampingNet(nn.Module):
    def __init__(self, conf, num_params=6):
        super().__init__()
        self.conf = conf
        if conf.type == 'constant':
            const = torch.zeros(num_params)
            self.register_parameter('const', torch.nn.Parameter(const))
        else:
            raise ValueError(f'Unsupported type of damping: {conf.type}.')

    def forward(self):
        min_, max_ = self.conf.log_range
        lambda_ = 10.**(min_ + self.const.sigmoid()*(max_ - min_))
        return lambda_


class LearnedOptimizer(BaseOptimizer):
    default_conf = dict(
        damping=dict(
            type='constant',
            log_range=[-6, 5],
        ),
        feature_dim=None,

        # deprecated entries
        lambda_=0.,
        learned_damping=True,
    )

    def _init(self, conf):
        self.dampingnet = DampingNet(conf.damping)
        assert conf.learned_damping
        super()._init(conf)

    def _run(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor,
             T_init: Pose, camera: Camera, mask: Optional[Tensor] = None,
             W_ref_query: Optional[Tuple[Tensor, Tensor]] = None):

        T = T_init
        J_scaling = None
        if self.conf.normalize_features:
            F_ref = torch.nn.functional.normalize(F_ref, dim=-1)
        args = (camera, p3D, F_ref, F_query, W_ref_query)
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        lambda_ = self.dampingnet()

        for i in range(self.conf.num_iters):
            res, valid, w_unc, _, J = self.cost_fn.residual_jacobian(T, *args)
            if mask is not None:
                valid &= mask
            failed = failed | (valid.long().sum(-1) < 10)  # too few points

            # compute the cost and aggregate the weights
            cost = (res**2).sum(-1)
            cost, w_loss, _ = self.loss_fn(cost)
            weights = w_loss * valid.float()
            if w_unc is not None:
                weights *= w_unc
            if self.conf.jacobi_scaling:
                J, J_scaling = self.J_scaling(J, J_scaling, valid)

            # solve the linear system
            g, H = self.build_system(J, res, weights)
            delta = optimizer_step(g, H, lambda_, mask=~failed)
            if self.conf.jacobi_scaling:
                delta = delta * J_scaling

            # compute the pose update
            dt, dw = delta.split([3, 3], dim=-1)
            T_delta = Pose.from_aa(dw, dt)
            T = T_delta @ T

            self.log(i=i, T_init=T_init, T=T, T_delta=T_delta, cost=cost,
                     valid=valid, w_unc=w_unc, w_loss=w_loss, H=H, J=J)
            if self.early_stop(i=i, T_delta=T_delta, grad=g, cost=cost):
                break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed
