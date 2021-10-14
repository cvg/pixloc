import logging
from typing import Dict, Optional, List

from .base_refiner import BaseRefiner
from ..pixlib.geometry import Pose, Camera
from ..utils.colmap import qvec2rotmat

logger = logging.getLogger(__name__)


class PoseRefiner(BaseRefiner):
    default_config = dict(
        min_matches_total=10,
    )

    def refine(self, qname: str, qcamera: Camera, loc: Dict) -> Dict:
        # Unpack initial query pose
        T_init = Pose.from_Rt(qvec2rotmat(loc["PnP_ret"]["qvec"]),
                              loc["PnP_ret"]["tvec"])
        fail = {'success': False, 'T_init': T_init}

        num_inliers = loc["PnP_ret"]["num_inliers"]
        if num_inliers < self.conf.min_matches_total:
            logger.debug(f"Too few inliers: {num_inliers}")
            return fail

        # Fetch database inlier matches count
        dbids = loc["db"]
        inliers = loc["PnP_ret"]["inliers"]
        ninl_dbs = self.model3d.get_db_inliers(loc, dbids, inliers)

        # Re-rank and filter database images
        dbids = self.model3d.rerank_and_filter_db_images(
                dbids, ninl_dbs, self.conf.num_dbs, self.conf.min_matches_db)

        # Abort if no image matches the minimum number of inliers criterion
        if len(dbids) == 0:
            logger.debug("No DB image with min num matches")
            return fail

        # Select the 3D points and collect their observations
        p3did_to_dbids = self.model3d.get_p3did_to_dbids(
                dbids, loc, inliers, self.conf.point_selection,
                self.conf.min_track_length)

        # Abort if there are not enough 3D points after filtering
        if len(p3did_to_dbids) < self.conf.min_points_opt:
            logger.debug("Not enough valid 3D points to optimize")
            return fail

        ret = self.refine_query_pose(qname, qcamera, T_init, p3did_to_dbids)
        ret = {**ret, 'dbids': dbids}
        return ret


class RetrievalRefiner(BaseRefiner):
    default_config = dict(
        multiscale=None,
        filter_covisibility=False,
        do_pose_approximation=False,
        do_inlier_ranking=False,
    )

    def __init__(self, *args, **kwargs):
        self.global_descriptors = kwargs.pop('global_descriptors', None)
        super().__init__(*args, **kwargs)

    def refine(self, qname: str, qcamera: Camera, dbids: List[int],
               loc: Optional[Dict] = None) -> Dict:

        if self.conf.do_inlier_ranking:
            assert loc is not None

        if self.conf.do_inlier_ranking and loc['PnP_ret']['success']:
            inliers = loc['PnP_ret']['inliers']
            ninl_dbs = self.model3d.get_db_inliers(loc, dbids, inliers)
            dbids = self.model3d.rerank_and_filter_db_images(
                    dbids, ninl_dbs, self.conf.num_dbs,
                    self.conf.min_matches_db)
        else:
            assert self.conf.point_selection == 'all'
            dbids = dbids[:self.conf.num_dbs]
            if self.conf.do_pose_approximation or self.conf.filter_covisibility:
                dbids = self.model3d.covisbility_filtering(dbids)
            inliers = None

        if self.conf.do_pose_approximation:
            if self.global_descriptors is None:
                raise RuntimeError(
                        'Pose approximation requires global descriptors')
            Rt_init = self.model3d.pose_approximation(
                    qname, dbids, self.global_descriptors)
        else:
            id_init = dbids[0]
            image_init = self.model3d.dbs[id_init]
            Rt_init = (image_init.qvec2rotmat(), image_init.tvec)
        T_init = Pose.from_Rt(*Rt_init)
        fail = {'success': False, 'T_init': T_init, 'dbids': dbids}

        p3did_to_dbids = self.model3d.get_p3did_to_dbids(
                dbids, loc, inliers, self.conf.point_selection,
                self.conf.min_track_length)

        # Abort if there are not enough 3D points after filtering
        if len(p3did_to_dbids) < self.conf.min_points_opt:
            logger.debug("Not enough valid 3D points to optimize")
            return fail

        ret = self.refine_query_pose(qname, qcamera, T_init, p3did_to_dbids,
                                     self.conf.multiscale)
        ret = {**ret, 'dbids': dbids}
        return ret
