import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import functools
import pickle
import scipy.spatial

from .. import logger
from ..settings import DATA_PATH, LOC_PATH
from ..utils.colmap import read_model
from ..utils.io import parse_image_lists


def preprocess_slice(slice_, root, sfm_path, min_common=50, verbose=False):
    logger.info(f'Preprocessing {slice_}.')
    root = root / slice_

    sfm = Path(str(sfm_path).format(slice_))
    assert sfm.exists(), sfm
    cameras, images, points3D = read_model(sfm, ext='.bin')

    query_poses_paths = root / 'camera-poses/*.txt'
    query_images = parse_image_lists(query_poses_paths, with_poses=True)
    assert len(query_images) > 0

    p3D_ids = sorted(points3D.keys())
    p3D_id_to_idx = dict(zip(p3D_ids, range(len(points3D))))
    p3D_xyz = np.stack([points3D[i].xyz for i in p3D_ids])
    track_lengths = np.stack([len(points3D[i].image_ids) for i in p3D_ids])

    ref_ids = sorted(images.keys())
    n_ref = len(images)
    if verbose:
        logger.info(f'Found {n_ref} ref images and {len(p3D_ids)} points.')

    ref_poses = []
    ref_image_names = []
    p3D_observed = []
    for i in ref_ids:
        image = images[i]
        R = image.qvec2rotmat()
        t = image.tvec
        obs = np.stack([
            p3D_id_to_idx[i] for i in image.point3D_ids if i != -1])

        assert (root / 'database' / image.name).exists()

        ref_poses.append((R, t))
        ref_image_names.append(image.name)
        p3D_observed.append(obs)

    query_poses = []
    query_image_names = []
    for _, image in query_images:
        R = image.qvec2rotmat()
        t = -R @ image.tvec
        query_poses.append((R, t))
        query_image_names.append(image.name)

        assert (root / 'query' / image.name).exists()

    p3D_observed_sets = [set(p) for p in p3D_observed]
    ref_overlaps = np.full([n_ref]*2, -1.)
    for idx1 in tqdm(range(n_ref), disable=not verbose):
        for idx2 in range(n_ref):
            if idx1 == idx2:
                continue

            common = p3D_observed_sets[idx1] & p3D_observed_sets[idx2]
            if len(common) < min_common:
                continue
            ref_overlaps[idx1, idx2] = len(common)/len(p3D_observed_sets[idx1])

    Rs_r = np.stack([p[0] for p in ref_poses])
    ts_r = np.stack([p[1] for p in ref_poses])
    Rs_q = np.stack([p[0] for p in query_poses])
    ts_q = np.stack([p[1] for p in query_poses])
    distances = scipy.spatial.distance.cdist(
        -np.einsum('nij,ni->nj', Rs_q, ts_q),
        -np.einsum('nij,ni->nj', Rs_r, ts_r))
    trace = np.einsum('nij,mij->nm', Rs_q, Rs_r, optimize=True)
    dR = np.clip((trace - 1) / 2, -1., 1.)
    dR = np.rad2deg(np.abs(np.arccos(dR)))
    mask = (dR < 30)
    masked_distances = np.where(mask, distances, np.inf)

    closest = np.argmin(masked_distances, 1)
    dist_closest = masked_distances.min(1)
    query_overlaps = np.stack([ref_overlaps[c] for c in closest], 0)
    query_overlaps[dist_closest > 1.] = -1

    data = {
        'points3D': p3D_xyz,
        'track_lengths': track_lengths,
        'ref_poses': ref_poses,
        'ref_image_names': ref_image_names,
        'p3D_observed': p3D_observed,
        'query_poses': query_poses,
        'query_image_names': query_image_names,
        'query_closest_indices': closest,
        'ref_overlap_matrix': ref_overlaps,
        'query_overlap_matrix': query_overlaps,
        'query_to_ref_distance_matrix': distances,
    }
    return data


def preprocess_and_write(slice_, root, out_dir, **kwargs):
    path = out_dir / (slice_ + '.pkl')
    if path.exists():
        return

    try:
        data = preprocess_slice(slice_, root, **kwargs)
    except:  # noqa  E722
        logger.info(f'Error for slice {slice_}.')
        raise
    if data is None:
        return

    logger.info(f'Writing slice {slice_} to {path}.')
    with open(path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    root = DATA_PATH / 'CMU/'
    sfm = LOC_PATH / 'CMU/{}/sfm_superpoint+superglue/model/'
    out_dir = DATA_PATH / 'cmu_pixloc_training/'
    out_dir.mkdir(exist_ok=True)

    slices = [6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25]
    slices = [f'slice{i}' for i in slices]
    logger.info(f'Found {len(slices)} slices.')

    fn = functools.partial(
            preprocess_and_write, root=root, sfm_path=sfm, out_dir=out_dir)
    with Pool(5) as p:
        p.map(fn, slices)
