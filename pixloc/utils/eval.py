import logging

from pathlib import Path
from typing import Union, Dict, Tuple, Optional
import numpy as np
from .io import parse_image_list
from .colmap import qvec2rotmat, read_images_binary, read_images_text

logger = logging.getLogger(__name__)


def evaluate(gt_sfm_model: Path, predictions: Union[Dict, Path],
             test_file_list: Optional[Path] = None,
             only_localized: bool = False):
    """Compute the evaluation metrics for 7Scenes and Cambridge Landmarks.
       The other datasets are evaluated on visuallocalization.net
    """
    if not isinstance(predictions, dict):
        predictions = parse_image_list(predictions, with_poses=True)
        predictions = {n: (im.qvec, im.tvec) for n, im in predictions}

    # ground truth poses from the sfm model
    images_bin = gt_sfm_model / 'images.bin'
    images_txt = gt_sfm_model / 'images.txt'
    if images_bin.exists():
        images = read_images_binary(images_bin)
    elif images_txt.exists():
        images = read_images_text(images_txt)
    else:
        raise ValueError(gt_sfm_model)
    name2id = {image.name: i for i, image in images.items()}

    if test_file_list is None:
        test_names = list(name2id)
    else:
        with open(test_file_list, 'r') as f:
            test_names = f.read().rstrip().split('\n')

    # translation and rotation errors
    errors_t = []
    errors_R = []
    for name in test_names:
        if name not in predictions:
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.
        else:
            image = images[name2id[name]]
            R_gt, t_gt = image.qvec2rotmat(), image.tvec
            qvec, t = predictions[name]
            R = qvec2rotmat(qvec)
            e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
        errors_t.append(e_t)
        errors_R.append(e_R)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    out = f'\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg'

    out += '\nPercentage of test images localized within:'
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f'\n\t{th_t*100:.0f}cm, {th_R:.0f}deg : {ratio*100:.2f}%'
    logger.info(out)


def cumulative_recall(errors: np.ndarray) -> Tuple[np.ndarray]:
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    return errors, recall*100
