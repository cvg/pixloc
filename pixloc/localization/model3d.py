import logging
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np

from ..utils.colmap import read_model
from ..utils.quaternions import weighted_pose

logger = logging.getLogger(__name__)


class Model3D:
    def __init__(self, path):
        logger.info('Reading COLMAP model %s.', path)
        self.cameras, self.dbs, self.points3D = read_model(path)
        self.name2id = {i.name: i.id for i in self.dbs.values()}

    def covisbility_filtering(self, dbids):
        clusters = do_covisibility_clustering(dbids, self.dbs, self.points3D)
        dbids = clusters[0]
        return dbids

    def pose_approximation(self, qname, dbids, global_descriptors, alpha=8):
        """Described in:
                Benchmarking Image Retrieval for Visual Localization.
                Noé Pion, Martin Humenberger, Gabriela Csurka,
                Yohann Cabon, Torsten Sattler. 3DV 2020.
        """
        dbs = [self.dbs[i] for i in dbids]

        dbdescs = np.stack([global_descriptors[im.name] for im in dbs])
        qdesc = global_descriptors[qname]
        sim = dbdescs @ qdesc
        weights = sim**alpha
        weights /= weights.sum()

        tvecs = [im.tvec for im in dbs]
        qvecs = [im.qvec for im in dbs]
        return weighted_pose(tvecs, qvecs, weights)

    def get_dbid_to_p3dids(self, p3did_to_dbids):
        """Link the database images to selected 3D points."""
        dbid_to_p3dids = defaultdict(list)
        for p3id, obs_dbids in p3did_to_dbids.items():
            for obs_dbid in obs_dbids:
                dbid_to_p3dids[obs_dbid].append(p3id)
        return dict(dbid_to_p3dids)

    def get_p3did_to_dbids(self, dbids: List, loc: Optional[Dict] = None,
                           inliers: Optional[List] = None,
                           point_selection: str = 'all',
                           min_track_length: int = 3):
        """Return a dictionary mapping 3D point ids to their covisible dbids.
        This function can use hloc sfm logs to only select inliers.
        Which can be further used to select top reference images / in
        sufficient track length selection of points.
        """
        p3did_to_dbids = defaultdict(set)
        if point_selection == 'all':
            for dbid in dbids:
                p3dids = self.dbs[dbid].point3D_ids
                for p3did in p3dids[p3dids != -1]:
                    p3did_to_dbids[p3did].add(dbid)
        elif point_selection in ['inliers', 'matched']:
            if loc is None:
                raise ValueError('"{point_selection}" point selection requires'
                                 ' localization logs.')

            # The given SfM model must match the localization SfM model!
            for (p3did, dbidxs), inlier in zip(loc["keypoint_index_to_db"][1],
                                               inliers):
                if inlier or point_selection == 'matched':
                    obs_dbids = set(loc["db"][dbidx] for dbidx in dbidxs)
                    obs_dbids &= set(dbids)
                    if len(obs_dbids) > 0:
                        p3did_to_dbids[p3did] |= obs_dbids
        else:
            raise ValueError(f"{point_selection} point selection not defined.")

        # Filter unstable points (min track length)
        p3did_to_dbids = {
            i: v
            for i, v in p3did_to_dbids.items()
            if len(self.points3D[i].image_ids) >= min_track_length
        }

        return p3did_to_dbids

    def rerank_and_filter_db_images(self, dbids: List, ninl_dbs: List,
                                    num_dbs: int, min_matches_db: int = 0):
        """Re-rank the images by inlier count and filter invalid images."""
        dbids = [dbids[i] for i in np.argsort(-ninl_dbs)
                 if ninl_dbs[i] > min_matches_db]
        # Keep top num_images matched image images
        dbids = dbids[:num_dbs]
        return dbids

    def get_db_inliers(self, loc: Dict, dbids: List, inliers: List):
        """Get the number of inliers for each db."""
        inliers = loc["PnP_ret"]["inliers"]
        dbids = loc["db"]
        ninl_dbs = np.zeros(len(dbids))
        for (_, dbidxs), inl in zip(loc["keypoint_index_to_db"][1], inliers):
            if not inl:
                continue
            for dbidx in dbidxs:
                ninl_dbs[dbidx] += 1
        return ninl_dbs


def do_covisibility_clustering(frame_ids, all_images, points3D):
    clusters = []
    visited = set()

    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = all_images[exploration_frame].point3D_ids
            connected_frames = set(
                j for i in observed if i != -1 for j in points3D[i].image_ids)
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters
