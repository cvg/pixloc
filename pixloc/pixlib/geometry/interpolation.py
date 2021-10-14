import torch
import cv2
import numpy as np
from typing import Tuple


@torch.jit.script
def interpolate_tensor_bicubic(tensor, pts, return_gradients: bool = False):
    # According to R. Keys "Cubic convolution interpolation for digital image processing".
    # references:
    # https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/cubic_interpolation.h
    # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/UpSample.h#L285
    # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/UpSampleBicubic2d.cpp#L63
    # https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/cubic_interpolation.h
    spline_base = torch.tensor([[-1, 2, -1, 0],
                                [3, -5, 0, 2],
                                [-3, 4, 1, 0],
                                [1, -1, 0, 0]]).float() / 2

    # This is the original written by Måns, does not seem consistent with OpenCV remap
    # spline_base = torch.tensor([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]]).float().T / 6
    spline_base = spline_base.to(tensor)

    pts_0 = torch.floor(pts)
    res = pts - pts_0
    x, y = pts_0[:, 0], pts_0[:, 1]

    c, h, w = tensor.shape
    f_patches = torch.zeros((c, len(pts), 4, 4)).to(tensor)
    # TODO: could we do this faster with gather or grid_sampler nearest?
    for i in [-1, 0, 1, 2]:
        for j in [-1, 0, 1, 2]:
            x_ = (x+j).long().clamp(min=0, max=w-1).long()
            y_ = (y+i).long().clamp(min=0, max=h-1).long()
            f_patches[:, :, i+1, j+1] = tensor[:, y_, x_]

    t = torch.stack([res**3, res**2, res, torch.ones_like(res)], -1)
    coeffs = torch.einsum('mk,nck->cnm', spline_base, t)
    coeffs_x, coeffs_y = coeffs[0], coeffs[1]
    interp = torch.einsum('ni,nj,cnij->nc', coeffs_y, coeffs_x, f_patches)

    if return_gradients:
        dt_xy = torch.stack([
            3*res**2, 2*res, torch.ones_like(res), torch.zeros_like(res)], -1)
        B_dt_xy = torch.einsum('mk,nck->cnm', spline_base, dt_xy)
        B_dt_x, B_dt_y = B_dt_xy[0], B_dt_xy[1]

        J_out_x = torch.einsum('ni,nj,cnij->nc', coeffs_y, B_dt_x, f_patches)
        J_out_y = torch.einsum('ni,nj,cnij->nc', B_dt_y, coeffs_x, f_patches)
        J_out_xy = torch.stack([J_out_x, J_out_y], -1)
    else:
        J_out_xy = torch.zeros(len(pts), c, 2).to(interp)

    return interp, J_out_xy


@torch.jit.script
def interpolate_tensor_bilinear(tensor, pts, return_gradients: bool = False):
    if tensor.dim() == 3:
        assert pts.dim() == 2
        batched = False
        tensor, pts = tensor[None], pts[None]
    else:
        batched = True

    b, c, h, w = tensor.shape
    scale = torch.tensor([w-1, h-1]).to(pts)
    pts = (pts / scale) * 2 - 1
    pts = pts.clamp(min=-2, max=2)  # ideally use the mask instead
    interpolated = torch.nn.functional.grid_sample(
            tensor, pts[:, None], mode='bilinear', align_corners=True)
    interpolated = interpolated.reshape(b, c, -1).transpose(-1, -2)

    if return_gradients:
        dxdy = torch.tensor([[1, 0], [0, 1]])[:, None].to(pts) / scale * 2
        dx, dy = dxdy.chunk(2, dim=0)
        pts_d = torch.cat([pts-dx, pts+dx, pts-dy, pts+dy], 1)
        tensor_d = torch.nn.functional.grid_sample(
                tensor, pts_d[:, None], mode='bilinear', align_corners=True)
        tensor_d = tensor_d.reshape(b, c, -1).transpose(-1, -2)
        tensor_x0, tensor_x1, tensor_y0, tensor_y1 = tensor_d.chunk(4, dim=1)
        gradients = torch.stack([
            (tensor_x1 - tensor_x0)/2, (tensor_y1 - tensor_y0)/2], dim=-1)
    else:
        gradients = torch.zeros(b, pts.shape[1], c, 2).to(tensor)

    if not batched:
        interpolated, gradients = interpolated[0], gradients[0]
    return interpolated, gradients


def mask_in_image(pts, image_size: Tuple[int, int], pad: int = 1):
    w, h = image_size
    image_size_ = torch.tensor([w-pad-1, h-pad-1]).to(pts)
    return torch.all((pts >= pad) & (pts <= image_size_), -1)


@torch.jit.script
def interpolate_tensor(tensor, pts, mode: str = 'linear',
                       pad: int = 1, return_gradients: bool = False):
    '''Interpolate a 3D tensor at given 2D locations.
    Args:
        tensor: with shape (C, H, W) or (B, C, H, W).
        pts: points with shape (N, 2) or (B, N, 2)
        mode: interpolation mode, `'linear'` or `'cubic'`
        pad: padding for the returned mask of valid keypoints
        return_gradients: whether to return the first derivative
            of the interpolated values (currentl only in cubic mode).
    Returns:
        tensor: with shape (N, C) or (B, N, C)
        mask: boolean mask, true if pts are in [pad, W-1-pad] x [pad, H-1-pad]
        gradients: (N, C, 2) or (B, N, C, 2), 0-filled if not return_gradients
    '''
    h, w = tensor.shape[-2:]
    if mode == 'cubic':
        pad += 1  # bicubic needs one more pixel on each side
    mask = mask_in_image(pts, (w, h), pad=pad)
    # Ideally we want to use mask to clamp outlier pts before interpolationm
    # but this line throws some obscure errors about inplace ops.
    # pts = pts.masked_fill(mask.unsqueeze(-1), 0.)

    if mode == 'cubic':
        interpolated, gradients = interpolate_tensor_bicubic(
                tensor, pts, return_gradients)
    elif mode == 'linear':
        interpolated, gradients = interpolate_tensor_bilinear(
                tensor, pts, return_gradients)
    else:
        raise NotImplementedError(mode)
    return interpolated, mask, gradients


class Interpolator:
    def __init__(self, mode: str = 'linear', pad: int = 1):
        self.mode = mode
        self.pad = pad

    def __call__(self, tensor: torch.Tensor, pts: torch.Tensor,
                 return_gradients: bool = False):
        return interpolate_tensor(
            tensor, pts, self.mode, self.pad, return_gradients)


def test_interpolate_cubic_opencv(f, pts):
    interp = interpolate_tensor_bicubic(f, pts)[0].cpu().numpy()
    interp_linear = interpolate_tensor(f, pts)[0].cpu().numpy()

    pts_ = pts.cpu().numpy()
    interp_cv2_cubic = []
    interp_cv2_linear = []
    for f_i in f.cpu().numpy():
        interp_i = cv2.remap(f_i, pts_[None], None, cv2.INTER_CUBIC)[0]
        interp_cv2_cubic.append(interp_i)
        interp_i = cv2.remap(f_i, pts_[None], None, cv2.INTER_LINEAR)[0]
        interp_cv2_linear.append(interp_i)
    interp_cv2_cubic = np.stack(interp_cv2_cubic, -1)
    interp_cv2_linear = np.stack(interp_cv2_linear, -1)

    diff = np.abs(interp - interp_cv2_cubic)
    print('OpenCV cubic vs custom cubic:')
    print('Mean/med/max abs diff', np.mean(diff), np.median(diff), np.max(diff))
    print('Rel diff', np.median(diff/np.abs(interp_cv2_cubic))*100, '%')

    diff = np.abs(interp_cv2_linear - interp_cv2_cubic)
    print('OpenCV cubic vs linear:')
    print('Mean/med/max abs diff', np.mean(diff), np.median(diff), np.max(diff))
    print('Rel diff', np.median(diff/np.abs(interp_cv2_cubic))*100, '%')

    diff = np.abs(interp_linear - interp_cv2_linear)
    print('OpenCV linear vs grid sample:')
    print('Mean/med/max abs diff', np.mean(diff), np.median(diff), np.max(diff))
    print('Rel diff', np.median(diff/np.abs(interp_cv2_linear))*100, '%')


def test_interpolate_cubic_gradients(tensor, pts):
    def compute_J(fn_J, inp):
        with torch.enable_grad():
            return torch.autograd.functional.jacobian(fn_J, inp)

    tensor, pts = tensor.double(), pts.double()

    _, J_analytical = interpolate_tensor_bicubic(
            tensor, pts, return_gradients=True)

    J = compute_J(
        lambda xy: interpolate_tensor_bicubic(tensor, xy.reshape(-1, 2))[0],
        pts.reshape(-1))
    J = J.reshape(J.shape[:2]+(-1, 2))
    J = J[range(len(pts)), :, range(len(pts)), :]

    print('Gradients consistent with autograd:',
          torch.allclose(J_analytical, J))


def test_run_all(seed=0):
    torch.random.manual_seed(seed)
    w, h = 480, 240

    pts = torch.rand(1000, 2) * torch.tensor([w-1, h-1])
    tensor = torch.rand(16, h, w)*100

    test_interpolate_cubic_opencv(tensor, pts)
    test_interpolate_cubic_gradients(tensor, pts)


if __name__ == '__main__':
    test_run_all()
