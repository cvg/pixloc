import torch
from typing import Optional, Tuple
from torch import Tensor

from . import Pose, Camera
from .optimization import J_normalization
from .interpolation import Interpolator


class DirectAbsoluteCost:
    def __init__(self, interpolator: Interpolator, normalize: bool = False):
        self.interpolator = interpolator
        self.normalize = normalize

    def residuals(
            self, T_w2q: Pose, camera: Camera, p3D: Tensor,
            F_ref: Tensor, F_query: Tensor,
            confidences: Optional[Tuple[Tensor, Tensor]] = None,
            do_gradients: bool = False):

        p3D_q = T_w2q * p3D
        p2D, visible = camera.world2image(p3D_q)
        F_p2D_raw, valid, gradients = self.interpolator(
            F_query, p2D, return_gradients=do_gradients)
        valid = valid & visible

        if confidences is not None:
            C_ref, C_query = confidences
            C_query_p2D, _, _ = self.interpolator(
                C_query, p2D, return_gradients=False)
            weight = C_ref * C_query_p2D
            weight = weight.squeeze(-1).masked_fill(~valid, 0.)
        else:
            weight = None

        if self.normalize:
            F_p2D = torch.nn.functional.normalize(F_p2D_raw, dim=-1)
        else:
            F_p2D = F_p2D_raw

        res = F_p2D - F_ref
        info = (p3D_q, F_p2D_raw, gradients)
        return res, valid, weight, F_p2D, info

    def jacobian(
            self, T_w2q: Pose, camera: Camera,
            p3D_q: Tensor, F_p2D_raw: Tensor, J_f_p2D: Tensor):

        J_p3D_T = T_w2q.J_transform(p3D_q)
        J_p2D_p3D, _ = camera.J_world2image(p3D_q)

        if self.normalize:
            J_f_p2D = J_normalization(F_p2D_raw) @ J_f_p2D

        J_p2D_T = J_p2D_p3D @ J_p3D_T
        J = J_f_p2D @ J_p2D_T
        return J, J_p2D_T

    def residual_jacobian(
            self, T_w2q: Pose, camera: Camera, p3D: Tensor,
            F_ref: Tensor, F_query: Tensor,
            confidences: Optional[Tuple[Tensor, Tensor]] = None):

        res, valid, weight, F_p2D, info = self.residuals(
            T_w2q, camera, p3D, F_ref, F_query, confidences, True)
        J, _ = self.jacobian(T_w2q, camera, *info)
        return res, valid, weight, F_p2D, J
