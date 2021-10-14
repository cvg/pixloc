from pathlib import Path
import collections
from tqdm import tqdm
import numpy as np
import logging
import torch
import pickle

from .base_dataset import BaseDataset
from .view import read_view
from .sampling import sample_pose_interval, sample_pose_reprojection
from ..geometry import Camera, Pose
from ...settings import DATA_PATH

logger = logging.getLogger(__name__)


class MegaDepth(BaseDataset):
    default_conf = {
        'dataset_dir': 'megadepth/',
        'depth_subpath': 'phoenix/S6/zl548/MegaDepth_v1/{}/dense0/depths/',
        'image_subpath': 'Undistorted_SfM/{}/images/',
        'info_dir': 'megadepth_pixloc_training/',

        'train_split': 'train_scenes.txt',
        'val_split': 'valid_scenes.txt',
        'train_num_per_scene': 500,
        'val_num_per_scene': 10,

        'two_view': True,
        'min_overlap': 0.3,
        'max_overlap': 1.,
        'sort_by_overlap': False,
        'init_pose': None,
        'init_pose_max_error': 63,
        'init_pose_num_samples': 20,

        'read_depth': False,
        'grayscale': False,
        'resize': None,
        'resize_by': 'max',
        'crop': None,
        'pad': None,
        'optimal_crop': True,
        'seed': 0,

        'max_num_points3D': 500,
        'force_num_points3D': False,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        assert split != 'test', 'Not supported'
        return _Dataset(self.conf, split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        if conf.init_pose is None:
            raise ValueError('The initial pose sampling strategy is required.')

        self.root = Path(DATA_PATH, conf.dataset_dir)
        with open(Path(__file__).parent / conf[split+'_split'], 'r') as f:
            self.scenes = f.read().split()
        self.conf, self.split = conf, split

        self.sample_new_items(conf.seed)

    def sample_new_items(self, seed):
        logger.info(f'Sampling new images or pairs with seed {seed}')
        self.images, self.poses, self.intrinsics = {}, {}, {}
        self.rotations, self.points3D, self.p3D_observed = {}, {}, {}
        self.items = []
        for scene in tqdm(self.scenes):
            path = Path(DATA_PATH, self.conf.info_dir, scene + '.pkl')
            if not path.exists():
                logger.warning(f'Scene {scene} does not have an info file')
                continue
            with open(path, 'rb') as f:
                info = pickle.load(f)
            num = self.conf[self.split+'_num_per_scene']

            self.images[scene] = info['image_names']
            self.rotations[scene] = info['rotations']
            self.points3D[scene] = info['points3D']
            self.p3D_observed[scene] = info['p3D_observed']
            self.poses[scene] = info['poses']
            self.intrinsics[scene] = info['intrinsics']

            if self.conf.two_view:
                mat = info['overlap_matrix']
                pairs = (
                    (mat > self.conf.min_overlap)
                    & (mat <= self.conf.max_overlap))
                pairs = np.stack(np.where(pairs), -1)
                if len(pairs) > num:
                    selected = np.random.RandomState(seed).choice(
                        len(pairs), num, replace=False)
                    pairs = pairs[selected]
                pairs = [(scene, i, j, mat[i, j]) for i, j in pairs]
                self.items.extend(pairs)
            else:
                ids = np.arange(len(self.images[scene]))
                if len(ids) > num:
                    ids = np.random.RandomState(seed).choice(
                        ids, num, replace=False)
                ids = [(scene, i) for i in ids]
                self.items.extend(ids)

        if self.conf.two_view and self.conf.sort_by_overlap:
            self.items.sort(key=lambda i: i[-1], reverse=True)
        else:
            np.random.RandomState(seed).shuffle(self.items)

    def _read_view(self, scene, idx, common_p3D_idx, is_reference=False):
        path = self.root / self.conf.image_subpath.format(scene)
        path /= self.images[scene][idx]

        if self.conf.read_depth:
            raise NotImplementedError

        K = self.intrinsics[scene][idx]
        camera = Camera.from_colmap(dict(
            model='PINHOLE', width=K[0, 2]*2, height=K[1, 2]*2,
            params=K[[0, 1, 0, 1], [0, 1, 2, 2]]))
        T = Pose.from_Rt(*self.poses[scene][idx])
        rotation = self.rotations[scene][idx]
        p3D = self.points3D[scene]
        data = read_view(self.conf, path, camera, T, p3D, common_p3D_idx,
                         rotation=rotation, random=(self.split == 'train'))
        data['index'] = idx
        assert (tuple(data['camera'].size.numpy())
                == data['image'].shape[1:][::-1])

        if is_reference:
            obs = self.p3D_observed[scene][idx]
            if self.conf.crop:
                _, valid = data['camera'].world2image(data['T_w2cam']*p3D[obs])
                obs = obs[valid.numpy()]
            num_diff = self.conf.max_num_points3D - len(obs)
            if num_diff < 0:
                obs = np.random.choice(obs, self.conf.max_num_points3D)
            elif num_diff > 0 and self.conf.force_num_points3D:
                add = np.random.choice(
                    np.delete(np.arange(len(p3D)), obs), num_diff)
                obs = np.r_[obs, add]
            data['points3D'] = data['T_w2cam'] * p3D[obs]
        return data

    def __getitem__(self, idx):
        if self.conf.two_view:
            scene, idx_r, idx_q, overlap = self.items[idx]
            common = np.array(list(set(self.p3D_observed[scene][idx_r])
                                   & set(self.p3D_observed[scene][idx_q])))

            data_r = self._read_view(scene, idx_r, common, is_reference=True)
            data_q = self._read_view(scene, idx_q, common)
            data = {
                'ref': data_r,
                'query': data_q,
                'overlap': overlap,
                'T_r2q_gt': data_q['T_w2cam'] @ data_r['T_w2cam'].inv(),
            }

            if self.conf.init_pose == 'identity':
                T_init = Pose.from_4x4mat(np.eye(4))
            elif self.conf.init_pose == 'max_error':
                T_init = sample_pose_reprojection(
                        data['T_r2q_gt'], data_q['camera'], data_r['points3D'],
                        self.conf.seed+idx, self.conf.init_pose_num_samples,
                        self.conf.init_pose_max_error)
            elif isinstance(self.conf.init_pose, collections.abc.Sequence):
                T_init = sample_pose_interval(
                    data['T_r2q_gt'], self.conf.init_pose, self.conf.seed+idx)
            else:
                raise ValueError(self.conf.init_pose)
            data['T_r2q_init'] = T_init
        else:
            scene, idx = self.items[idx]
            data = self._read_view(scene, idx, is_reference=True)
        data['scene'] = scene
        return data

    def __len__(self):
        return len(self.items)
