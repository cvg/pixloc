import logging
from typing import Dict, List, Union, Any
from pathlib import Path
from collections import defaultdict
import numpy as np
import h5py

from .colmap import Camera, Image

logger = logging.getLogger(__name__)


def parse_image_list(path: Path, with_intrinsics: bool = False,
                     with_poses: bool = False) -> List:
    images = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            if with_intrinsics:
                camera_model, width, height, *params = data
                params = np.array(params, float)
                camera = Camera(
                    None, camera_model, int(width), int(height), params)
                images.append((name, camera))
            elif with_poses:
                qvec, tvec = np.split(np.array(data, float), [4])
                image = Image(
                    id=None, qvec=qvec, tvec=tvec, camera_id=None, name=name,
                    xys=None, point3D_ids=None)
                images.append((name, image))
            else:
                images.append(name)

    logger.info(f'Imported {len(images)} images from {path.name}')
    return images


def parse_image_lists(paths: Path, **kwargs) -> List:
    images = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0, paths
    for lfile in files:
        images += parse_image_list(lfile, **kwargs)
    return images


def parse_retrieval(path: Path) -> Dict[str, List[str]]:
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            q, r = p.split()
            retrieval[q].append(r)
    return dict(retrieval)


def load_hdf5(path: Path) -> Dict[str, Any]:
    with h5py.File(path, 'r') as hfile:
        data = {}
        def collect(_, obj):  # noqa
            if isinstance(obj, h5py.Dataset):
                name = obj.parent.name.strip('/')
                data[name] = obj.__array__()
        hfile.visititems(collect)
    return data


def write_pose_results(pose_dict: Dict, outfile: Path,
                       prepend_camera_name: bool = False):
    logger.info('Writing the localization results to %s.', outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(str(outfile), 'w') as f:
        for imgname, (qvec, tvec) in pose_dict.items():
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = imgname.split('/')[-1]
            if prepend_camera_name:
                name = imgname.split('/')[-2] + '/' + name
            f.write(f'{name} {qvec} {tvec}\n')


def concat_results(paths: List[Path], names: List[Union[int, str]],
                   output_path: Path, key: str) -> Path:
    results = []
    for path in sorted(paths):
        with open(path, 'r') as fp:
            results.append(fp.read().rstrip('\n'))
    output_path = str(output_path).replace(
            f'{{{key}}}', '-'.join(str(n)[:3] for n in names))
    with open(output_path, 'w') as fp:
        fp.write('\n'.join(results))
    return Path(output_path)
