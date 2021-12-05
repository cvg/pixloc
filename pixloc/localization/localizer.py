import logging
import pickle
from typing import Optional, Dict, Tuple, Union
from omegaconf import DictConfig, OmegaConf as oc
from tqdm import tqdm
import torch

from .model3d import Model3D
from .feature_extractor import FeatureExtractor
from .refiners import PoseRefiner, RetrievalRefiner

from ..utils.data import Paths
from ..utils.io import parse_image_lists, parse_retrieval, load_hdf5
from ..utils.quaternions import rotmat2qvec
from ..pixlib.utils.experiments import load_experiment
from ..pixlib.models import get_model
from ..pixlib.geometry import Camera

logger = logging.getLogger(__name__)
# TODO: despite torch.no_grad in BaseModel, requires_grad flips in ref interp
torch.set_grad_enabled(False)


class Localizer:
    def __init__(self, paths: Paths, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')

        self.model3d = Model3D(paths.reference_sfm)
        cameras = parse_image_lists(paths.query_list, with_intrinsics=True)
        self.queries = {n: c for n, c in cameras}

        # Loading feature extractor and optimizer from experiment or scratch
        conf = oc.create(conf)
        conf_features = conf.features.get('conf', {})
        conf_optim = conf.get('optimizer', {})
        if conf.get('experiment'):
            pipeline = load_experiment(
                    conf.experiment,
                    {'extractor': conf_features, 'optimizer': conf_optim})
            pipeline = pipeline.to(device)
            logger.debug(
                'Use full pipeline from experiment %s with config:\n%s',
                conf.experiment, oc.to_yaml(pipeline.conf))
            extractor = pipeline.extractor
            optimizer = pipeline.optimizer
            if isinstance(optimizer, torch.nn.ModuleList):
                optimizer = list(optimizer)
        else:
            assert 'name' in conf.features
            extractor = get_model(conf.features.name)(conf_features)
            optimizer = get_model(conf.optimizer.name)(conf_optim)

        self.paths = paths
        self.conf = conf
        self.device = device
        self.optimizer = optimizer
        self.extractor = FeatureExtractor(
            extractor, device, conf.features.get('preprocessing', {}))

    def run_query(self, name: str, camera: Camera):
        raise NotImplementedError

    def run_batched(self, skip: Optional[int] = None,
                    ) -> Tuple[Dict[str, Tuple], Dict]:
        output_poses = {}
        output_logs = {
            'paths': self.paths.asdict(),
            'configuration': oc.to_yaml(self.conf),
            'localization': {},
        }

        logger.info('Starting the localization process...')
        query_names = list(self.queries.keys())[::skip or 1]
        for name in tqdm(query_names):
            camera = Camera.from_colmap(self.queries[name])
            try:
                ret = self.run_query(name, camera)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    logger.info('Out of memory')
                    torch.cuda.empty_cache()
                    ret = {'success': False}
                else:
                    raise
            output_logs['localization'][name] = ret
            if ret['success']:
                R, tvec = ret['T_refined'].numpy()
            elif 'T_init' in ret:
                R, tvec = ret['T_init'].numpy()
            else:
                continue
            output_poses[name] = (rotmat2qvec(R), tvec)

        return output_poses, output_logs


class RetrievalLocalizer(Localizer):
    def __init__(self, paths: Paths, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        super().__init__(paths, conf, device)

        if paths.global_descriptors is not None:
            global_descriptors = load_hdf5(paths.global_descriptors)
        else:
            global_descriptors = None

        self.refiner = RetrievalRefiner(
            self.device, self.optimizer, self.model3d, self.extractor, paths,
            self.conf.refinement, global_descriptors=global_descriptors)

        if paths.hloc_logs is not None:
            logger.info('Reading hloc logs...')
            with open(paths.hloc_logs, 'rb') as f:
                self.logs = pickle.load(f)['loc']
            self.retrieval = {q: [self.model3d.dbs[i].name for i in loc['db']]
                              for q, loc in self.logs.items()}
        elif paths.retrieval_pairs is not None:
            self.logs = None
            self.retrieval = parse_retrieval(paths.retrieval_pairs)
        else:
            raise ValueError

    def run_query(self, name: str, camera: Camera):
        dbs = [self.model3d.name2id[r] for r in self.retrieval[name]]
        loc = None if self.logs is None else self.logs[name]
        ret = self.refiner.refine(name, camera, dbs, loc=loc)
        return ret


class PoseLocalizer(Localizer):
    def __init__(self, paths: Paths, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        super().__init__(paths, conf, device)

        self.refiner = PoseRefiner(
            device, self.optimizer, self.model3d, self.extractor, paths,
            self.conf.refinement)

        logger.info('Reading hloc logs...')
        with open(paths.hloc_logs, 'rb') as f:
            self.logs = pickle.load(f)['loc']

    def run_query(self, name: str, camera: Camera):
        loc = self.logs[name]
        if loc['PnP_ret']['success']:
            ret = self.refiner.refine(name, camera, loc)
        else:
            ret = {'success': False}
        return ret
