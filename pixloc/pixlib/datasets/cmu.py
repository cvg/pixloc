from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
import torch
import pickle

from .base_dataset import BaseDataset
from .view import read_view
from ..geometry import Camera, Pose
from ...settings import DATA_PATH

logger = logging.getLogger(__name__)


CAMERAS = '''c0 OPENCV 1024 768 868.993378 866.063001 525.942323 420.042529 -0.399431 0.188924 0.000153 0.000571
c1 OPENCV 1024 768 873.382641 876.489513 529.324138 397.272397 -0.397066 0.181925 0.000176 -0.000579'''


class CMU(BaseDataset):
    default_conf = {
        'dataset_dir': 'CMU/',
        'info_dir': 'cmu_pixloc_training/',

        'train_slices': [8, 9, 10, 11, 12, 22, 23, 24, 25],
        'val_slices': [6, 13, 21],
        'train_num_per_slice': 1000,
        'val_num_per_slice': 80,

        'two_view': True,
        'min_overlap': 0.3,
        'max_overlap': 1.,
        'min_baseline': None,
        'max_baseline': None,
        'sort_by_overlap': False,

        'grayscale': False,
        'resize': None,
        'resize_by': 'max',
        'crop': None,
        'pad': None,
        'optimal_crop': True,
        'seed': 0,

        'max_num_points3D': 512,
        'force_num_points3D': False,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        assert split != 'test', 'Not supported'
        return _Dataset(self.conf, split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.root = Path(DATA_PATH, conf.dataset_dir)
        self.slices = conf.get(split+'_slices')
        self.conf, self.split = conf, split

        self.info = {}
        for slice_ in self.slices:
            path = Path(DATA_PATH, self.conf.info_dir, f'slice{slice_}.pkl')
            assert path.exists(), path
            with open(path, 'rb') as f:
                info = pickle.load(f)
            self.info[slice_] = {k: info[k] for k in info if 'matrix' not in k}

        self.cameras = {}
        for c in CAMERAS.split('\n'):
            data = c.split()
            name, camera_model, width, height = data[:4]
            params = np.array(data[4:], float)
            camera = Camera.from_colmap(dict(
                    model=camera_model, params=params,
                    width=int(width), height=int(height)))
            self.cameras[name] = camera

        self.sample_new_items(conf.seed)

    def sample_new_items(self, seed):
        logger.info(f'Sampling new images or pairs with seed {seed}')
        self.items = []
        for slice_ in tqdm(self.slices):
            num = self.conf[self.split+'_num_per_slice']

            if self.conf.two_view:
                path = Path(
                        DATA_PATH, self.conf.info_dir, f'slice{slice_}.pkl')
                assert path.exists(), path
                with open(path, 'rb') as f:
                    info = pickle.load(f)

                mat = info['query_overlap_matrix']
                pairs = (
                    (mat > self.conf.min_overlap)
                    & (mat <= self.conf.max_overlap))
                if self.conf.min_baseline:
                    pairs &= (info['query_to_ref_distance_matrix']
                              > self.conf.min_baseline)
                if self.conf.max_baseline:
                    pairs &= (info['query_to_ref_distance_matrix']
                              < self.conf.max_baseline)
                pairs = np.stack(np.where(pairs), -1)
                if len(pairs) > num:
                    selected = np.random.RandomState(seed).choice(
                        len(pairs), num, replace=False)
                    pairs = pairs[selected]
                pairs = [(slice_, i, j, mat[i, j]) for i, j in pairs]
                self.items.extend(pairs)
            else:
                ids = np.arange(len(self.images[slice_]))
                if len(ids) > num:
                    ids = np.random.RandomState(seed).choice(
                        ids, num, replace=False)
                ids = [(slice_, i) for i in ids]
                self.items.extend(ids)

        if self.conf.two_view and self.conf.sort_by_overlap:
            self.items.sort(key=lambda i: i[-1], reverse=True)
        else:
            np.random.RandomState(seed).shuffle(self.items)

    def _read_view(self, slice_, idx, common_p3D_idx, is_reference=False):
        prefix = 'ref' if is_reference else 'query'
        path = self.root / f'slice{slice_}/'
        path /= 'database' if is_reference else 'query'
        path /= self.info[slice_][f'{prefix}_image_names'][idx]

        camera = self.cameras[path.name.split('_')[2]]
        R, t = self.info[slice_][f'{prefix}_poses'][idx]
        T = Pose.from_Rt(R, t)
        p3D = self.info[slice_]['points3D']
        data = read_view(self.conf, path, camera, T, p3D, common_p3D_idx,
                         random=(self.split == 'train'))
        data['index'] = idx
        assert (tuple(data['camera'].size.numpy())
                == data['image'].shape[1:][::-1])

        if is_reference:
            obs = self.info[slice_]['p3D_observed'][idx]
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
            slice_, idx_q, idx_r, overlap = self.items[idx]
            obs_r = self.info[slice_]['p3D_observed'][idx_r]
            obs_q = self.info[slice_]['p3D_observed'][
                    self.info[slice_]['query_closest_indices'][idx_q]]
            common = np.array(list(set(obs_r) & set(obs_q)))

            data_r = self._read_view(slice_, idx_r, common, is_reference=True)
            data_q = self._read_view(slice_, idx_q, common)
            data = {
                'ref': data_r,
                'query': data_q,
                'overlap': overlap,
                'T_r2q_init': Pose.from_4x4mat(np.eye(4, dtype=np.float32)),
                'T_r2q_gt': data_q['T_w2cam'] @ data_r['T_w2cam'].inv(),
            }
        else:
            slice_, idx = self.items[idx]
            data = self._read_view(slice_, idx, is_reference=True)
        data['scene'] = slice_
        return data

    def __len__(self):
        return len(self.items)
