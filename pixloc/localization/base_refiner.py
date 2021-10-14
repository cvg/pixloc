import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
from omegaconf import DictConfig, OmegaConf as oc
import numpy as np
import torch

from .feature_extractor import FeatureExtractor
from .model3d import Model3D
from .tracker import BaseTracker
from ..pixlib.geometry import Pose, Camera
from ..pixlib.datasets.view import read_image
from ..utils.data import Paths

logger = logging.getLogger(__name__)


class BaseRefiner:
    base_default_config = dict(
        layer_indices=None,
        min_matches_db=10,
        num_dbs=1,
        min_track_length=3,
        min_points_opt=10,
        point_selection='all',
        average_observations=False,
        normalize_descriptors=True,
        compute_uncertainty=True,
    )

    default_config = dict()
    tracker: BaseTracker = None

    def __init__(self,
                 device: torch.device,
                 optimizer: torch.nn.Module,
                 model3d: Model3D,
                 feature_extractor: FeatureExtractor,
                 paths: Paths,
                 conf: Union[DictConfig, Dict]):
        self.device = device
        self.optimizer = optimizer
        self.model3d = model3d
        self.feature_extractor = feature_extractor
        self.paths = paths

        self.conf = oc.merge(
            oc.create(self.base_default_config),
            oc.create(self.default_config),
            oc.create(conf))

    def log_dense(self, **kwargs):
        if self.tracker is not None:
            self.tracker.log_dense(**kwargs)

    def log_optim(self, **kwargs):
        if self.tracker is not None:
            self.tracker.log_optim_done(**kwargs)

    def refine(self, **kwargs):
        ''' Implement this in the child class'''
        raise NotImplementedError

    def refine_pose_using_features(self,
                                   features_query: List[torch.tensor],
                                   scales_query: List[float],
                                   qcamera: Camera,
                                   T_init: Pose,
                                   features_p3d: List[List[torch.Tensor]],
                                   p3dids: List[int]) -> Dict:
        """Perform the pose refinement using given dense query feature-map.
        """
        # decompose descriptors and uncertainities, normalize descriptors
        weights_ref = []
        features_ref = []
        for level in range(len(features_p3d[0])):
            feats = torch.stack([feat[level] for feat in features_p3d], dim=0)
            feats = feats.to(self.device)
            if self.conf.compute_uncertainty:
                feats, weight = feats[:, :-1], feats[:, -1:]
                weights_ref.append(weight)
            if self.conf.normalize_descriptors:
                feats = torch.nn.functional.normalize(feats, dim=1)
            assert not feats.requires_grad
            features_ref.append(feats)

        # query dense features decomposition and normalization
        features_query = [feat.to(self.device) for feat in features_query]
        if self.conf.compute_uncertainty:
            weights_query = [feat[-1:] for feat in features_query]
            features_query = [feat[:-1] for feat in features_query]
        if self.conf.normalize_descriptors:
            features_query = [torch.nn.functional.normalize(feat, dim=0)
                              for feat in features_query]

        p3d = np.stack([self.model3d.points3D[p3did].xyz for p3did in p3dids])

        T_i = T_init
        ret = {'T_init': T_init}
        # We will start with the low res feature map first
        for idx, level in enumerate(reversed(range(len(features_query)))):
            F_q, F_ref = features_query[level], features_ref[level]
            qcamera_feat = qcamera.scale(scales_query[level])

            if self.conf.compute_uncertainty:
                W_ref_query = (weights_ref[level], weights_query[level])
            else:
                W_ref_query = None

            logger.debug(f'Optimizing at level {level}.')
            opt = self.optimizer
            if isinstance(opt, (tuple, list)):
                if self.conf.layer_indices:
                    opt = opt[self.conf.layer_indices[level]]
                else:
                    opt = opt[level]
            T_opt, fail = opt.run(p3d, F_ref, F_q, T_i.to(F_q),
                                  qcamera_feat.to(F_q),
                                  W_ref_query=W_ref_query)

            self.log_optim(i=idx, T_opt=T_opt, fail=fail, level=level,
                           p3d=p3d, p3d_ids=p3dids,
                           T_init=T_init, camera=qcamera_feat)
            if fail:
                return {**ret, 'success': False}
            T_i = T_opt

        # Compute relative pose w.r.t. initilization
        T_opt = T_opt.cpu().double()
        dR, dt = (T_init.inv() @ T_opt).magnitude()
        return {
            **ret,
            'success': True,
            'T_refined': T_opt,
            'diff_R': dR.item(),
            'diff_t': dt.item(),
        }

    def refine_query_pose(self, qname: str, qcamera: Camera, T_init: Pose,
                          p3did_to_dbids: Dict[int, List],
                          multiscales: Optional[List[int]] = None) -> Dict:

        dbid_to_p3dids = self.model3d.get_dbid_to_p3dids(p3did_to_dbids)
        if multiscales is None:
            multiscales = [1]

        rnames = [self.model3d.dbs[i].name for i in dbid_to_p3dids.keys()]
        images_ref = [read_image(self.paths.reference_images / n)
                      for n in rnames]

        for image_scale in multiscales:
            # Compute the reference observations
            # TODO: can we compute this offline before hand?
            dbid_p3did_to_feats = dict()
            for idx, dbid in enumerate(dbid_to_p3dids):
                p3dids = dbid_to_p3dids[dbid]

                features_ref_dense, scales_ref = self.dense_feature_extraction(
                        images_ref[idx], rnames[idx], image_scale)
                dbid_p3did_to_feats[dbid] = self.interp_sparse_observations(
                        features_ref_dense, scales_ref, dbid, p3dids)
                del features_ref_dense

            p3did_to_feat = self.aggregate_features(
                    p3did_to_dbids, dbid_p3did_to_feats)
            if self.conf.average_observations:
                p3dids = list(p3did_to_feat.keys())
                p3did_to_feat = [p3did_to_feat[p3did] for p3did in p3dids]
            else:  # duplicate the observations
                p3dids, p3did_to_feat = list(zip(*[
                    (p3did, feat) for p3did, feats in p3did_to_feat.items()
                    for feat in zip(*feats)]))

            # Compute dense query feature maps
            image_query = read_image(self.paths.query_images / qname)
            features_query, scales_query = self.dense_feature_extraction(
                        image_query, qname, image_scale)

            ret = self.refine_pose_using_features(features_query, scales_query,
                                                  qcamera, T_init,
                                                  p3did_to_feat, p3dids)
            if not ret['success']:
                logger.info(f"Optimization failed for query {qname}")
                break
            else:
                T_init = ret['T_refined']
        return ret

    def dense_feature_extraction(self, image: np.array, name: str,
                                 image_scale: int = 1
                                 ) -> Tuple[List[torch.Tensor], List[int]]:
        features, scales, weight = self.feature_extractor(
                image, image_scale)
        self.log_dense(name=name, image=image, image_scale=image_scale,
                       features=features, scales=scales, weight=weight)

        if self.conf.compute_uncertainty:
            assert weight is not None
            # stack them into a single tensor (makes the bookkeeping easier)
            features = [torch.cat([f, w], 0) for f, w in zip(features, weight)]

        # Filter out some layers or keep them all
        if self.conf.layer_indices is not None:
            features = [features[i] for i in self.conf.layer_indices]
            scales = [scales[i] for i in self.conf.layer_indices]

        return features, scales

    def interp_sparse_observations(self,
                                   feature_maps: List[torch.Tensor],
                                   feature_scales: List[float],
                                   image_id: float,
                                   p3dids: List[int],
                                   ) -> Dict[int, torch.Tensor]:
        image = self.model3d.dbs[image_id]
        camera = Camera.from_colmap(self.model3d.cameras[image.camera_id])
        T_w2cam = Pose.from_colmap(image)
        p3d = np.array([self.model3d.points3D[p3did].xyz for p3did in p3dids])
        p3d_cam = T_w2cam * p3d

        # interpolate sparse descriptors and store
        feature_obs = []
        masks = []
        for i, (feats, sc) in enumerate(zip(feature_maps, feature_scales)):
            p2d_feat, valid = camera.scale(sc).world2image(p3d_cam)
            opt = self.optimizer
            opt = opt[len(opt)-i-1] if isinstance(opt, (tuple, list)) else opt
            obs, mask, _ = opt.interpolator(feats, p2d_feat.to(feats))
            assert not obs.requires_grad
            feature_obs.append(obs)
            masks.append(mask & valid.to(mask))

        mask = torch.all(torch.stack(masks, dim=0), dim=0)

        # We can't stack features because they have different # of channels
        feature_obs = [[feature_obs[i][j] for i in range(len(feature_maps))]
                       for j in range(len(p3dids))]  # N x K x D

        feature_dict = {p3id: feature_obs[i]
                        for i, p3id in enumerate(p3dids) if mask[i]}

        return feature_dict

    def aggregate_features(self,
                           p3did_to_dbids: Dict,
                           dbid_p3did_to_feats: Dict,
                           ) -> Dict[int, List[torch.Tensor]]:
        """Aggregate descriptors from covisible images through averaging.
        """
        p3did_to_feat = defaultdict(list)
        for p3id, obs_dbids in p3did_to_dbids.items():
            features = []
            for obs_imgid in obs_dbids:
                if p3id not in dbid_p3did_to_feats[obs_imgid]:
                    continue
                features.append(dbid_p3did_to_feats[obs_imgid][p3id])
            if len(features) > 0:
                # list with one entry per layer, grouping all 3D observations
                for level in range(len(features[0])):
                    observation = [f[level] for f in features]
                    if self.conf.average_observations:
                        observation = torch.stack(observation, 0)
                        if self.conf.compute_uncertainty:
                            feat, w = observation[:, :-1], observation[:, -1:]
                            feat = (feat * w).sum(0) / w.sum(0)
                            observation = torch.cat([feat, w.mean(0)], -1)
                        else:
                            observation = observation.mean(0)
                    p3did_to_feat[p3id].append(observation)
        return dict(p3did_to_feat)
