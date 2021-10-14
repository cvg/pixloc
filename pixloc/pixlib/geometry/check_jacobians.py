import torch
import logging

from . import Pose, Camera
from .costs import DirectAbsoluteCost
from .interpolation import Interpolator

logger = logging.getLogger(__name__)


def compute_J(fn_J, inp):
    with torch.enable_grad():
        return torch.autograd.functional.jacobian(fn_J, inp)


def compute_J_batched(fn, inp):
    inp_ = inp.reshape(-1)
    fn_ = lambda x: fn(x.reshape(inp.shape))  # noqa
    J = compute_J(fn_, inp_)
    if len(J.shape) != 3:
        raise ValueError('Only supports a single leading batch dimension.')
    J = J.reshape(J.shape[:-1] + inp.shape)
    J = J.diagonal(dim1=0, dim2=-2).permute(2, 0, 1)
    return J


def local_param(delta):
    dt, dw = delta.split(3, dim=-1)
    return Pose.from_aa(dw, dt)


def toy_problem(seed=0, n_points=500):
    torch.random.manual_seed(seed)
    aa = torch.randn(3) / 10
    t = torch.randn(3) / 5
    T_w2q = Pose.from_aa(aa, t)

    w, h = 640, 480
    fx, fy = 300., 350.
    cx, cy = w/2, h/2
    radial = [0.1, 0.01]
    camera = Camera(torch.tensor([w, h, fx, fy, cx, cy] + radial)).float()
    torch.testing.assert_allclose((w, h), camera.size.long())
    torch.testing.assert_allclose((fx, fy), camera.f)
    torch.testing.assert_allclose((cx, cy), camera.c)

    p3D = torch.randn(n_points, 3)
    p3D[:, -1] += 2

    dim = 16
    F_ref = torch.randn(n_points, dim)
    F_query = torch.randn(dim, h, w)

    return T_w2q, camera, p3D, F_ref, F_query


def print_J_diff(prefix, J, J_auto):
    logger.info('Check J %s: pass=%r, max_diff=%e, shape=%r',
                prefix,
                torch.allclose(J, J_auto),
                torch.abs(J-J_auto).max(),
                tuple(J.shape))


def test_J_pose(T: Pose, p3D: torch.Tensor):
    J = T.J_transform(T * p3D)
    fn = lambda d: (local_param(d) @ T) * p3D  # noqa
    delta = torch.zeros(6).to(p3D)
    J_auto = compute_J(fn, delta)
    print_J_diff('pose transform', J, J_auto)


def test_J_undistort(camera: Camera, p3D: torch.Tensor):
    p2D, valid = camera.project(p3D)
    J = camera.J_undistort(p2D)
    J_auto = compute_J_batched(camera.undistort, p2D)
    J, J_auto = J[valid], J_auto[valid]
    print_J_diff('undistort', J, J_auto)


def test_J_world2image(camera: Camera, p3D: torch.Tensor):
    _, valid = camera.world2image(p3D)
    J, _ = camera.J_world2image(p3D)
    J_auto = compute_J_batched(lambda x: camera.world2image(x)[0], p3D)
    J, J_auto = J[valid], J_auto[valid]
    print_J_diff('world2image', J, J_auto)


def test_J_geometric_cost(T_w2q: Pose, camera: Camera, p3D: torch.Tensor):
    def forward(T):
        p3D_q = T * p3D
        p2D, visible = camera.world2image(p3D_q)
        return p2D, visible, p3D_q

    _, valid, p3D_q = forward(T_w2q)
    J = camera.J_world2image(p3D_q)[0] @ T_w2q.J_transform(p3D_q)
    delta = torch.zeros(6).to(p3D)
    fn = lambda d: forward(local_param(d) @ T_w2q)[0]  # noqa
    J_auto = compute_J(fn, delta)
    J, J_auto = J[valid], J_auto[valid]
    print_J_diff('geometric cost', J, J_auto)


def test_J_direct_absolute_cost(T_w2q: Pose, camera: Camera, p3D: torch.Tensor,
                                F_ref, F_query):
    interpolator = Interpolator(mode='cubic', pad=2)
    cost = DirectAbsoluteCost(interpolator, normalize=True)

    args = (camera, p3D, F_ref, F_query)
    res, valid, weight, F_q_p2D, info = cost.residuals(
            T_w2q, *args, do_gradients=True)
    J, _ = cost.jacobian(T_w2q, camera, *info)

    delta = torch.zeros(6).to(p3D)
    fn = lambda d: cost.residuals(local_param(d) @ T_w2q, *args)[0]   # noqa
    J_auto = compute_J(fn, delta)

    J, J_auto = J[valid], J_auto[valid]
    print_J_diff('direct absolute cost', J, J_auto)


def main():
    T_w2q, camera, p3D, F_ref, F_query = toy_problem()
    test_J_pose(T_w2q, p3D)
    test_J_undistort(camera, p3D)
    test_J_world2image(camera, p3D)

    # perform the checsk in double precision to factor out numerical errors
    T_w2q, camera, p3D, F_ref, F_query = (
        x.to(torch.double) for x in (T_w2q, camera, p3D, F_ref, F_query))

    test_J_geometric_cost(T_w2q, camera, p3D)
    test_J_direct_absolute_cost(T_w2q, camera, p3D, F_ref, F_query)


if __name__ == '__main__':
    main()
