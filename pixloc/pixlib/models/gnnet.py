"""
An implementation of
    GN-Net: The Gauss-Newton Loss for Multi-Weather Relocalization
    Lukas von Stumberg, Patrick Wenzel, Qadeer Khan, Daniel Cremers
    International Conference on Robotics and Automation (ICRA) 2020

Adapted from code written by Zimeng Jiang, Le Chen, and Lixin Xue.
"""

import torch
from torch.nn import functional as nnF
import numpy as np

from .base_model import BaseModel
from . import get_model
from .optimization.losses import scaled_barron
from .utils import masked_mean
from ...utils.interpolation import interpolate_tensor


def distance_s2d(pts, dense):
    '''...,N,D vs ...,D,H,W'''
    dist = -2*torch.einsum('...nd,...dhw->...nhw', pts, dense)
    dist.add_(torch.sum(pts**2, -1)[..., None, None])
    dist.add_(torch.sum(dense**2, -3)[..., None, :, :])
    return dist


class GNNet(BaseModel):
    default_conf = {
        'extractor': {
            'name': '???',
        },
        'normalize_features': False,
        'loss': {
            'margin_positive': 0.2,
            'margin_negative': 1.,
            'windowed_negative_sampling': True,
            'ratio_radius_negative': 0.1,
            'num_top_negative_sampling': 200,
            'gauss_newton_magnitude': 1.,
            'gauss_newton_reg_weight': 1.,
            'gauss_newton_weight': 1.,
            'contrastive_weight': 1.,
        },
        'optimizer': {
            'name': 'basic_optimizer',
        },
    }
    required_data_keys = {
        'ref': ['image', 'camera'],
        'query': ['image', 'camera'],
    }
    strict_conf = False  # need to pass new confs to children models

    def _init(self, conf):
        self.extractor = get_model(conf.extractor.name)(conf.extractor)
        assert hasattr(self.extractor, 'scales')
        self.optimizer = get_model(conf.optimizer.name)(conf.optimizer)

    def contrastive_loss(self, F0_p, F1_p, F1, pts1):
        dist_pos = torch.norm(F0_p - F1_p, p=2, dim=-1)
        loss_pos = torch.clamp(dist_pos-self.conf.loss.margin_positive, min=0)
        loss_pos = loss_pos**2

        all_dist_neg = distance_s2d(F0_p.detach(), F1.detach())

        c, h, w = F1.shape[-3:]
        if self.conf.loss.windowed_negative_sampling:
            radius = self.conf.loss.ratio_radius_negative * max(h, w)
            mask_x = torch.abs(torch.arange(w, device=pts1.device)[None, None]
                               - pts1[..., :1]) < radius
            mask_y = torch.abs(torch.arange(h, device=pts1.device)[None, None]
                               - pts1[..., 1:]) < radius
            mask_neg = mask_x[..., None, :] & mask_y[..., None]
            all_dist_neg.masked_fill_(mask_neg, float('inf'))
        all_dist_neg = all_dist_neg.reshape(all_dist_neg.shape[:-2]+(-1,))

        n = self.conf.loss.num_top_negative_sampling
        if n > 0:
            inds = torch.topk(all_dist_neg, n, dim=-1, largest=False).indices
            sampled = torch.randint_like(inds[..., 0], high=n)
            inds = inds.gather(-1, sampled[..., None]).squeeze(-1)
        else:
            n = all_dist_neg.shape[-1]
            inds = torch.randint(n, all_dist_neg.shape[:-1])
            inds = inds.to(all_dist_neg.device)

        # For an unknown reason, gather generates huge grad tensor,
        # resulting in OOM, while indexing works just fine.
        inds = torch.stack([inds // w, inds % w], -1)  # y, x
        F1_neg = torch.stack([F.permute(1, 2, 0)[tuple(i.T)]
                              for F, i in zip(F1, inds)], 0)
        # F1_neg = F1.reshape(-1, 1, c, h*w).expand(inds.shape + (-1, -1))
        # inds = inds[..., None, None].expand(-1, -1, c, 1)
        # F1_neg = F1_neg.gather(-1, inds).squeeze(-1)

        # Different formulation in GN-Net vs LM-Reloc: square after or before.
        # Here we pick the formulation of LM-Reloc.
        dist_neg = torch.sum((F0_p - F1_neg)**2, -1)
        # dist_neg = torch.norm(F0_p - F1_neg, p=2, dim=-1)
        loss_neg = torch.clamp(self.conf.loss.margin_negative-dist_neg, min=0)
        # loss_neg = loss_neg**2
        return loss_pos + loss_neg

    def gauss_newton_loss(self, F0_p, F1_p, F0, pts0, mask):
        noise = (torch.rand_like(pts0)*2-1)
        pts0_noise = pts0 + noise * self.conf.loss.gauss_newton_magnitude

        F0_noise, mask_noise, J = interpolate_tensor(
                F0, pts0_noise, pad=4, return_gradients=True)
        residuals = F0_noise - F1_p

        H = torch.einsum('...dp,...dq->...pq', J, J)
        H = H + torch.eye(H.shape[-1]).to(H) * 1e-6
        g = torch.einsum('...dp,...d->...p', J, residuals)

        H = torch.where(mask[..., None, None], H, torch.eye(H.shape[-1]).to(H))
        H_, g_ = H.cpu(), g.cpu()
        try:
            U = torch.cholesky(H_, upper=True)
        except Exception as e:  # noqa: F841
            import ipdb; ipdb.set_trace()  # noqa: E702

        delta_ = -torch.cholesky_solve(g_[..., None], U, upper=True)[..., 0]
        delta = delta_.to(H)

        pts0_step = pts0_noise + delta
        error = pts0 - pts0_step
        nll = torch.einsum('...p,...pq,...q->...', error, H, error) / 2
        reg = np.log(2*np.pi) - H.logdet() / 2
        nll += self.conf.loss.gauss_newton_reg_weight * reg

        return nll, mask_noise

    def _forward(self, data):
        def process_siamese(data_i):
            pred_i = self.extractor(data_i)
            pred_i['camera_pyr'] = [data_i['camera'].scale(1/s)
                                    for s in self.extractor.scales]
            return pred_i

        pred = {i: process_siamese(data[i]) for i in ['ref', 'query']}
        return pred

    def loss(self, pred, data):
        p3D_r = data['ref']['points3D']
        p3D_q = data['T_r2q_gt'] * p3D_r
        contrastive_losses = []
        gn_losses = []
        for i in range(len(self.extractor.scales)):
            Fr = pred['ref']['feature_maps'][i]
            Fq = pred['query']['feature_maps'][i]
            cam_r = pred['ref']['camera_pyr'][i]
            cam_q = pred['query']['camera_pyr'][i]

            if self.conf.normalize_features:
                Fr = nnF.normalize(Fr, dim=1)
                Fq = nnF.normalize(Fq, dim=1)

            p2D_r, visible_r = cam_r.world2image(p3D_r)
            p2D_q, visible_q = cam_q.world2image(p3D_q)

            Fr_p, valid_r, _ = interpolate_tensor(Fr, p2D_r, pad=4)
            Fq_p, valid_q, _ = interpolate_tensor(Fq, p2D_q, pad=4)
            valid = visible_r & visible_q & valid_r & valid_q

            gn0, valid_0_noise = self.gauss_newton_loss(Fr_p, Fq_p, Fr,
                                                        p2D_r, valid)
            contrastive0 = self.contrastive_loss(Fr_p, Fq_p, Fq, p2D_q)

            gn1, valid_1_noise = self.gauss_newton_loss(Fq_p, Fr_p, Fq,
                                                        p2D_q, valid)
            contrastive1 = self.contrastive_loss(Fq_p, Fr_p, Fr, p2D_r)

            gn0 = masked_mean(gn0, valid & valid_0_noise, -1)
            gn1 = masked_mean(gn1, valid & valid_1_noise, -1)
            gn_loss = (gn0 + gn1) / 2
            contrastive = masked_mean((contrastive0+contrastive1)/2, valid, -1)

            gn_losses.append(gn_loss)
            contrastive_losses.append(contrastive)

        gn_loss = torch.stack(gn_losses, 0).mean(0)
        contrastive = torch.stack(contrastive_losses, 0).mean(0)

        total = 0.
        if self.conf.loss.gauss_newton_weight > 0:
            total += self.conf.loss.gauss_newton_weight * gn_loss
        if self.conf.loss.contrastive_weight > 0:
            total += self.conf.loss.contrastive_weight * contrastive

        losses = {}
        losses['total'] = total
        losses['gauss_newton_loss'] = gn_loss
        losses['contrastive_loss'] = contrastive

        return losses

    def metrics(self, pred, data):
        p3D_ref = data['ref']['points3D']
        T_init = data['T_r2q_init']
        T_r2q_gt = data['T_r2q_gt']
        T_q2r_gt = T_r2q_gt.inv()

        T_r2q_init = []
        T_r2q_opt = []
        for i in reversed(range(len(self.extractor.scales))):
            F_ref = pred['ref']['feature_maps'][i]
            F_q = pred['query']['feature_maps'][i]
            cam_ref = pred['ref']['camera_pyr'][i]
            cam_q = pred['query']['camera_pyr'][i]

            p2D_ref, visible = cam_ref.world2image(p3D_ref)
            F_ref, mask, _ = self.optimizer.interpolator(F_ref, p2D_ref)
            mask &= visible

            if self.conf.normalize_features:
                F_ref = nnF.normalize(F_ref, dim=2)
                F_q = nnF.normalize(F_q, dim=1)

            T_opt, failed = self.optimizer(dict(
                p3D=p3D_ref, F_ref=F_ref, F_q=F_q, T_init=T_init, cam_q=cam_q,
                mask=mask))

            T_r2q_init.append(T_init)
            T_r2q_opt.append(T_opt)
            T_init = T_opt.detach()

        def scaled_pose_error(T_r2q):
            err_R, err_t = (T_r2q @ T_q2r_gt).magnitude()
            err_t /= torch.norm(T_q2r_gt.t, dim=-1)
            return err_R, err_t

        metrics = {}
        # Compute pose errors
        for i, T_opt in enumerate(T_r2q_opt):
            err = scaled_pose_error(T_opt)
            metrics[f'R_error/{i}'], metrics[f't_error/{i}'] = err
        metrics['R_error'], metrics['t_error'] = err
        err_init = scaled_pose_error(T_r2q_init[0])
        metrics['R_error/init'], metrics['t_error/init'] = err_init

        # Compute reprojection errors
        def project(T_r2q):
            return cam_q.world2image(T_r2q * p3D_ref)

        p2D_q_gt, mask = project(T_r2q_gt)
        p2D_q_i, mask_i = project(T_r2q_init[0])

        def reprojection_error(T_r2q):
            p2D_q, _ = project(T_r2q)
            err = torch.sum((p2D_q_gt - p2D_q)**2, dim=-1)
            err = scaled_barron(1., 2.)(err)[0]/4
            err = masked_mean(err, mask, -1)
            return err

        for i, T_opt in enumerate(T_r2q_opt):
            err = reprojection_error(T_opt).clamp(max=50)
            metrics[f'loss/reprojection_error/{i}'] = err
        metrics['loss/reprojection_error'] = err

        err_init = reprojection_error(T_r2q_init[0])
        metrics['loss/reprojection_error/init'] = err_init

        return metrics
