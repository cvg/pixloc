from pathlib import Path
from typing import Optional, List
import logging
import shutil
import json
import io
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .viz_2d import save_plot
from ..localization import Model3D
from ..pixlib.geometry import Pose, Camera
from ..utils.quaternions import rotmat2qvec

logger = logging.getLogger(__name__)
try:
    import ffmpeg
except ImportError:
    logger.info('Cannot import ffmpeg.')


def subsample_steps(T_w2q: Pose, p2d_q: np.ndarray, mask_q: np.ndarray,
                    camera_size: np.ndarray, thresh_dt: float = 0.1,
                    thresh_px: float = 0.005) -> List[int]:
    """Subsample steps of the optimization based on camera or point
       displacements. Main use case: compress an animation
       but keep it smooth and interesting.
    """
    mask = mask_q.any(0)
    dp2ds = np.linalg.norm(np.diff(p2d_q, axis=0), axis=-1)
    dp2ds = np.median(dp2ds[:, mask], 1)
    dts = (T_w2q[:-1] @ T_w2q[1:].inv()).magnitude()[0].numpy()
    assert len(dts) == len(dp2ds)

    thresh_dp2 = camera_size.min()*thresh_px  # from percent to pixel

    num = len(dp2ds)
    keep = []
    count_dp2 = 0
    count_dt = 0
    for i, dp2 in enumerate(dp2ds):
        count_dp2 += dp2
        count_dt += dts[i]
        if (i == 0 or i == (num-1)
                or count_dp2 >= thresh_dp2 or count_dt >= thresh_dt):
            count_dp2 = 0
            count_dt = 0
            keep.append(i)
    return keep


class VideoWriter:
    """Write frames sequentially as images, create a video, and clean up."""
    def __init__(self, tmp_dir: Path, ext='.jpg'):
        self.tmp_dir = Path(tmp_dir)
        self.ext = ext
        self.count = 0
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True)

    def add_frame(self):
        save_plot(self.tmp_dir / f'{self.count:0>5}{self.ext}')
        plt.close()
        self.count += 1

    def to_video(self, out_path: Path, duration: Optional[float] = None,
                 fps: int = 5, crf: int = 23, verbose: bool = False):
        assert self.count > 0
        if duration is not None:
            fps = self.count / duration
        frames = self.tmp_dir / f'*{self.ext}'
        logger.info('Running ffmpeg.')
        (
            ffmpeg
            .input(frames, pattern_type='glob', framerate=fps)
            .filter('crop', 'trunc(iw/2)*2', 'trunc(ih/2)*2')
            .output(out_path, crf=crf, vcodec='libx264', pix_fmt='yuv420p')
            .run(overwrite_output=True, quiet=not verbose)
        )
        shutil.rmtree(self.tmp_dir)


def display_video(path: Path):
    from IPython.display import HTML
    # prevent jupyter from caching the video file
    data = io.open(path, 'r+b').read()
    encoded = base64.b64encode(data).decode('ascii')
    return HTML(f"""
        <video width="100%" controls autoplay loop>
            <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
        </video>
    """)


def frustum_points(camera: Camera) -> np.ndarray:
    """Compute the corners of the frustum of a camera object."""
    W, H = camera.size.numpy()
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H],
                        [0, 0], [W/2, -H/5], [W, 0]])
    corners = (corners - camera.c.numpy()) / camera.f.numpy()
    return corners


def copy_compress_image(source: Path, target: Path, quality: int = 50):
    """Read an image and write it to a low-quality jpeg."""
    image = cv2.imread(str(source))
    cv2.imwrite(str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def format_json(x, decimals: int = 3):
    """Control the precision of numpy float arrays, convert boolean to int."""
    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.floating):
            if x.shape != (4,):  # qvec
                x = np.round(x, decimals=decimals)
        elif x.dtype == np.bool:
            x = x.astype(int)
        return x.tolist()
    if isinstance(x, float):
        return round(x, decimals)
    if isinstance(x, dict):
        return {k: format_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [format_json(v) for v in x]
    return x


def create_viz_dump(assets: Path, paths: Path, cam_q: Camera, name_q: str,
                    T_w2q: Pose, mask_q: np.ndarray, p2d_q: np.ndarray,
                    ref_ids: List[int], model3d: Model3D, p3d_ids: np.ndarray,
                    tfm: np.ndarray = np.eye(3)):
    assets.mkdir(parents=True, exist_ok=True)

    dump = {
        'p3d': {},
        'T': {},
        'camera': {},
        'image': {},
        'p2d': {},
    }

    p3d = np.stack([model3d.points3D[i].xyz for i in p3d_ids], 0)
    dump['p3d']['colors'] = [model3d.points3D[i].rgb for i in p3d_ids]
    dump['p3d']['xyz'] = p3d @ tfm.T

    dump['T']['refs'] = []
    dump['camera']['refs'] = []
    dump['image']['refs'] = []
    dump['p2d']['refs'] = []
    for idx, ref_id in enumerate(ref_ids):
        ref = model3d.dbs[ref_id]
        cam_r = Camera.from_colmap(model3d.cameras[ref.camera_id])
        T_w2r = Pose.from_colmap(ref)

        qtvec = (rotmat2qvec(T_w2r.R.numpy() @ tfm.T), T_w2r.t.numpy())
        dump['T']['refs'].append(qtvec)
        dump['camera']['refs'].append(frustum_points(cam_r))

        tmp_name = f'ref{idx}.jpg'
        dump['image']['refs'].append(tmp_name)
        copy_compress_image(
            paths.reference_images / ref.name, assets / tmp_name)

        p2d_, valid_ = cam_r.world2image(T_w2r * p3d)
        p2d_ = p2d_[valid_ & mask_q.any(0)] / cam_r.size
        dump['p2d']['refs'].append(p2d_.numpy())

    qtvec_q = [(rotmat2qvec(T.R.numpy() @ tfm.T), T.t.numpy()) for T in T_w2q]
    dump['T']['query'] = qtvec_q
    dump['camera']['query'] = frustum_points(cam_q)

    p2d_q_norm = [np.asarray(p[v]/cam_q.size) for p, v in zip(p2d_q, mask_q)]
    dump['p2d']['query'] = p2d_q_norm[-1]

    tmp_name = 'query.jpg'
    dump['image']['query'] = tmp_name
    copy_compress_image(paths.query_images / name_q, assets / tmp_name)

    with open(assets / 'dump.json', 'w') as fid:
        json.dump(format_json(dump), fid, separators=(',', ':'))

    # We dump 2D points as a separate json because it is much heavier
    # and thus slower to load.
    dump_p2d = {
        'query': p2d_q_norm,
        'masks': np.asarray(mask_q),
    }
    with open(assets / 'dump_p2d.json', 'w') as fid:
        json.dump(format_json(dump_p2d), fid, separators=(',', ':'))
