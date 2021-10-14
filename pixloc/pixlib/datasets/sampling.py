from typing import Union, Tuple
import torch
import numpy as np
import cv2

from ..geometry import Pose, Camera


def sample_pose_reprojection(
            T_r2q: Pose, camera: Camera, p3D_r: np.ndarray, seed: int,
            num_samples: int, max_err: Union[int, float, Tuple[int, float]],
            min_vis: int = 10):

    R0, t0 = T_r2q.R, T_r2q.t
    w0 = cv2.Rodrigues(R0.numpy())[0][:, 0]

    s = torch.linspace(0, 1, num_samples+1)[:, None]
    Ts = Pose.from_aa(torch.from_numpy(w0)[None] * s, t0[None] * s)

    p2Ds, vis = camera.world2image(Ts * p3D_r)
    p2Ds, vis = p2Ds.numpy(), vis.numpy()

    p2D0, vis0 = p2Ds[-1], vis[-1]
    err = np.linalg.norm(p2Ds - p2D0, axis=-1)
    err = np.where(vis & vis0, err, np.nan)
    valid = ~np.all(np.isnan(err), -1)
    err = np.where(valid[:, None], err, np.inf)
    err = np.nanmedian(err, -1)
    nvis = np.sum(vis & vis0, -1)

    if not isinstance(max_err, (int, float)):
        max_err = np.random.RandomState(seed).uniform(*max_err)
    valid = (nvis >= min_vis) & (err < max_err)
    if valid.any():
        idx = np.where(valid)[0][0]
    else:
        idx = -1
    return Ts[idx]


def sample_pose_interval(T_r2q: Pose, interval: Tuple[float], seed: int):
    a = np.random.RandomState(seed).uniform(*interval)
    R, t = T_r2q.numpy()
    t = t * a
    w = cv2.Rodrigues(R)[0][:, 0] * a
    T = Pose.from_Rt(cv2.Rodrigues(w)[0], t)
    return T
