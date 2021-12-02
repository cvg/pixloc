import argparse
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional
from omegaconf import DictConfig, OmegaConf as oc

from .. import settings, logger


@dataclasses.dataclass
class Paths:
    query_images: Path
    reference_images: Path
    reference_sfm: Path
    query_list: Path

    dataset: Optional[Path] = None
    dumps: Optional[Path] = None

    retrieval_pairs: Optional[Path] = None
    results: Optional[Path] = None
    global_descriptors: Optional[Path] = None
    hloc_logs: Optional[Path] = None
    log_path: Optional[Path] = None
    ground_truth: Optional[Path] = None

    def interpolate(self, **kwargs) -> 'Paths':
        args = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if val is not None:
                val = str(val)
                for k, v in kwargs.items():
                    val = val.replace(f'{{{k}}}', str(v))
                val = Path(val)
            args[f.name] = val
        return self.__class__(**args)

    def asdict(self) -> Dict[str, Path]:
        return dataclasses.asdict(self)

    @classmethod
    def fields(cls) -> List[str]:
        return [f.name for f in dataclasses.fields(cls)]

    def add_prefixes(self, dataset: Path, dumps: Path,
                     eval_dir: Optional[Path] = Path('.')) -> 'Paths':
        paths = {}
        for attr in self.fields():
            val = getattr(self, attr)
            if val is not None:
                if attr in {'dataset', 'dumps'}:
                    paths[attr] = val
                elif attr in {'query_images',
                              'reference_images',
                              'ground_truth'}:
                    paths[attr] = dataset / val
                elif attr in {'results'}:
                    paths[attr] = eval_dir / val
                else:  # everything else is part of the hloc dumps
                    paths[attr] = dumps / val
        paths['dataset'] = dataset
        paths['dumps'] = dumps
        return self.__class__(**paths)


def create_argparser(dataset: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--results', type=Path)
    parser.add_argument('--reference_sfm', type=Path)
    parser.add_argument('--retrieval', type=Path)
    parser.add_argument('--global_descriptors', type=Path)
    parser.add_argument('--hloc_logs', type=Path)

    parser.add_argument('--dataset', type=Path,
                        default=settings.DATA_PATH / dataset)
    parser.add_argument('--dumps', type=Path,
                        default=settings.LOC_PATH / dataset)
    parser.add_argument('--eval_dir', type=Path,
                        default=settings.EVAL_PATH)

    parser.add_argument('--from_poses', action='store_true')
    parser.add_argument('--inlier_ranking', action='store_true')
    parser.add_argument('--skip', type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('dotlist', nargs='*')

    return parser


def parse_paths(args, default_paths: Paths) -> Paths:
    default_paths = default_paths.add_prefixes(
            args.dataset, args.dumps, args.eval_dir)
    paths = {}
    for attr in Paths.fields():
        val = getattr(args, attr, None)
        if val is None:
            val = getattr(default_paths, attr, None)
            if val is None:
                continue
        paths[attr] = val
    return Paths(**paths)


def parse_conf(args, default_confs: Dict) -> DictConfig:
    conf = default_confs['from_poses' if args.from_poses else 'from_retrieval']
    conf = oc.merge(oc.create(conf), oc.from_cli(args.dotlist))
    logger.info('Parsed configuration:\n%s', oc.to_yaml(conf))
    return conf
