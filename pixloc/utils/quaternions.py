import numpy as np


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def weighted_qvecs(qvecs, weights):
    """Adapted from Tolga Birdal:
       https://github.com/tolgabirdal/averaging_quaternions/blob/master/wavg_quaternion_markley.m
    """
    outer = np.einsum('ni,nj,n->ij', qvecs, qvecs, weights)
    avg = np.linalg.eigh(outer)[1][:, -1]  # eigenvector of largest eigenvalue
    avg *= np.sign(avg[0])
    return avg


def weighted_pose(t_w2c, q_w2c, weights):
    weights = np.array(weights)
    R_w2c = np.stack([qvec2rotmat(q) for q in q_w2c], 0)

    t_c2w = -np.einsum('nij,ni->nj', R_w2c, np.array(t_w2c))
    t_approx_c2w = np.sum(t_c2w * weights[:, None], 0)

    q_c2w = np.array(q_w2c) * np.array([[1, -1, -1, -1]])  # invert
    q_c2w *= np.sign(q_c2w[:, 0])[:, None]  # handle antipodal
    q_approx_c2w = weighted_qvecs(q_c2w, weights)

    # convert back to camera coordinates
    R_approx = qvec2rotmat(q_approx_c2w).T
    t_approx = -R_approx @ t_approx_c2w

    return R_approx, t_approx
