import numpy as np
from tqdm import tqdm
import logging
from multiprocessing import Pool
import functools
import pickle

from ..settings import DATA_PATH
from ..utils.colmap import read_model

logger = logging.getLogger(__name__)


def assemble_intrinsics(fx, fy, cx, cy):
    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx - 0.5  # COLMAP convention
    K[1, 2] = cy - 0.5
    return K


def get_camera_angles(R_c_to_w):
    trace = np.einsum('nji,mji->mn', R_c_to_w, R_c_to_w, optimize=True)
    dR = np.clip((trace - 1) / 2, -1., 1.)
    dR = np.rad2deg(np.abs(np.arccos(dR)))
    return dR


def in_plane_rotation_matrix(rot):
    a = np.deg2rad(-90*rot)
    R = np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a), np.cos(a), 0],
        [0, 0, 1]])
    return R


def rotate_intrinsics(K, image_shape, rot):
    """Correct the intrinsics after in-plane rotation.
    Args:
        K: the original (3, 3) intrinsic matrix.
        image_shape: shape of the image after rotation `[H, W]`.
        rot: the number of clockwise 90deg rotations.
    """
    h, w = image_shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 0:
        return K
    elif rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)
    else:
        raise ValueError


def find_in_plane_rotations(R_c_to_w):
    gravity = np.median(R_c_to_w @ np.array([0, 1, 0]), 0)
    gravity_2D = (R_c_to_w.transpose(0, 2, 1) @ gravity)[:, :2]
    gravity_angle = np.rad2deg(np.arctan2(gravity_2D[:, 0], gravity_2D[:, 1]))
    rotated = np.abs(gravity_angle) > 60

    rot90 = np.array([-90, 180, -180, 90])
    rot90_indices = np.array([1, 2, 2, 3])
    rotations = np.zeros(len(rotated), int)
    if np.any(rotated):
        rots = np.argmin(
            np.abs(rot90[None] - gravity_angle[rotated][:, None]), -1)
        rotations[rotated] = rot90_indices[rots]
    return rotations


def preprocess_scene(scene, root, min_common=50, verbose=False):
    logger.info(f'Preprocessing scene {scene}.')
    sfm = root / scene / 'sparse'
    if not sfm.exists():  # empty model
        logger.warning(f'Scene {scene} is empty.')
        return None
    cameras, images, points3D = read_model(sfm, ext='.bin')

    p3D_ids = sorted(points3D.keys())
    p3D_id_to_idx = dict(zip(p3D_ids, range(len(points3D))))
    p3D_xyz = np.stack([points3D[i].xyz for i in p3D_ids])
    track_lengths = np.stack([len(points3D[i].image_ids) for i in p3D_ids])

    images_ids = sorted(images.keys())
    n_images = len(images)
    if n_images == 0:
        return None
    if verbose:
        logger.info(f'Found {n_images} images and {len(p3D_ids)} points.')

    intrinsics = []
    poses = []
    p3D_observed = []
    image_names = []
    too_small = []
    for i in tqdm(images_ids, disable=not verbose):
        image = images[i]
        camera = cameras[image.camera_id]
        assert camera.model == 'PINHOLE', camera.model
        K = assemble_intrinsics(*camera.params)
        R = image.qvec2rotmat()
        t = image.tvec
        obs = np.stack([
            p3D_id_to_idx[i] for i in image.point3D_ids if i != -1])

        intrinsics.append(K)
        poses.append((R, t))
        p3D_observed.append(obs)
        image_names.append(image.name)
        too_small.append(min(camera.height, camera.width) < 480)

    R_w_to_c = np.stack([R for R, _ in poses], 0)
    R_c_to_w = R_w_to_c.transpose(0, 2, 1)
    rotations = find_in_plane_rotations(R_c_to_w)
    for idx, rot in enumerate(rotations):
        if rot == 0:
            continue
        R_rot = in_plane_rotation_matrix(rot)
        R, t = poses[idx]
        poses[idx] = (R_rot@R, R_rot@t)

        image = images[images_ids[idx]]
        camera = cameras[image.camera_id]
        shape = (camera.height, camera.width)
        K = intrinsics[idx]
        intrinsics[idx] = rotate_intrinsics(K, shape, rot)

    R_w_to_c = np.stack([R for R, _ in poses], 0)
    R_c_to_w = R_w_to_c.transpose(0, 2, 1)
    camera_angles = get_camera_angles(R_c_to_w)

    p3D_observed_sets = [set(p) for p in p3D_observed]
    overlaps = np.full([n_images]*2, -1.)
    for idx1 in tqdm(range(n_images), disable=not verbose):
        for idx2 in range(idx1+1, n_images):
            if too_small[idx1] or too_small[idx2]:
                continue
            n_common = len(p3D_observed_sets[idx1] & p3D_observed_sets[idx2])
            if n_common < min_common:
                continue
            overlaps[idx1, idx2] = n_common / len(p3D_observed[idx1])
            overlaps[idx2, idx1] = n_common / len(p3D_observed[idx2])

    data = {
        'points3D': p3D_xyz,
        'track_lengths': track_lengths,
        'intrinsics': intrinsics,
        'poses': poses,
        'p3D_observed': p3D_observed,
        'image_names': image_names,
        'rotations': rotations,
        'overlap_matrix': overlaps,
        'angle_matrix': camera_angles,
    }
    return data


def preprocess_and_write(scene, root, out_dir, **kwargs):
    path = out_dir / (scene + '.pkl')
    if path.exists():
        return

    try:
        data = preprocess_scene(scene, root, **kwargs)
    except:  # noqa  E722
        logger.info(f'Error for scene {scene}.')
        raise
    if data is None:
        return

    logger.info(f'Writing scene {scene} to {path}.')
    with open(path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    root = DATA_PATH / 'megadepth/Undistorted_SfM/'
    out_dir = DATA_PATH / 'megadepth_pixloc_training/'
    out_dir.mkdir(exist_ok=True)

    scenes = sorted([s.name for s in root.iterdir() if s.is_dir()])
    logger.info(f'Found {len(scenes)} scenes.')

    fn = functools.partial(preprocess_and_write, root=root, out_dir=out_dir)
    with Pool(5) as p:
        p.map(fn, scenes)
