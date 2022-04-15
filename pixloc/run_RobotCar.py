import pickle
from pathlib import Path

from . import set_logging_debug, logger
from .localization import RetrievalLocalizer, PoseLocalizer
from .utils.data import Paths, create_argparser, parse_paths, parse_conf
from .utils.io import write_pose_results, concat_results

default_paths = Paths(
    query_images='images/',
    reference_images='images/',
    reference_sfm='sfm_superpoint+superglue/',
    query_list='{condition}_queries_with_intrinsics.txt',
    global_descriptors='robotcar_ov-ref_tf-netvlad.h5',
    retrieval_pairs='pairs-query-netvlad10-percam-perloc.txt',
    results='pixloc_RobotCar_{condition}.txt',
)

experiment = 'pixloc_cmu'

default_confs = {
    'from_retrieval': {
        'experiment': experiment,
        'features': {},
        'optimizer': {
            'num_iters': 100,
            'pad': 2,
        },
        'refinement': {
            'num_dbs': 2,
            'point_selection': 'all',
            'normalize_descriptors': True,
            'average_observations': False,
            'filter_covisibility': False,
            'do_pose_approximation': False,
        },
    },
    'from_poses': {
        'experiment': experiment,
        'features': {},
        'optimizer': {
            'num_iters': 100,
            'pad': 2,
        },
        'refinement': {
            'num_dbs': 5,
            'min_points_opt': 100,
            'point_selection': 'inliers',
            'normalize_descriptors': True,
            'average_observations': False,
            'layer_indices': [0, 1],
        },
    },
}


CONDITIONS = ['dawn', 'dusk', 'night', 'night-rain', 'overcast-summer',
              'overcast-winter', 'rain', 'snow', 'sun']


def generate_query_list(paths, condition):
    h, w = 1024, 1024
    intrinsics_filename = 'intrinsics/{}_intrinsics.txt'
    cameras = {}
    for side in ['left', 'right', 'rear']:
        with open(paths.dataset / intrinsics_filename.format(side), 'r') as f:
            fx = f.readline().split()[1]
            fy = f.readline().split()[1]
            cx = f.readline().split()[1]
            cy = f.readline().split()[1]
            assert fx == fy
            params = ['SIMPLE_RADIAL', w, h, fx, cx, cy, 0.0]
            cameras[side] = [str(p) for p in params]

    queries = sorted((paths.query_images / condition).glob('**/*.jpg'))
    queries = [str(q.relative_to(paths.query_images)) for q in queries]

    out = [[q] + cameras[Path(q).parent.name] for q in queries]
    with open(paths.query_list, 'w') as f:
        f.write('\n'.join(map(' '.join, out)))


def main():
    parser = create_argparser('RobotCar')
    parser.add_argument('--conditions', default=CONDITIONS, choices=CONDITIONS,
                        nargs='+')
    args = parser.parse_args()

    set_logging_debug(args.verbose)
    paths = parse_paths(args, default_paths)
    conf = parse_conf(args, default_confs)
    logger.info('Will evaluate %s conditions.', len(args.conditions))

    all_results = []
    for condition in args.conditions:
        logger.info('Working on condition %s.', condition)
        paths_cond = paths.interpolate(condition=condition)
        all_results.append(paths_cond.results)
        if paths_cond.results.exists():
            continue
        if not paths_cond.query_list.exists():
            generate_query_list(paths_cond, condition)

        if args.from_poses:
            localizer = PoseLocalizer(paths_cond, conf)
        else:
            localizer = RetrievalLocalizer(paths_cond, conf)
        poses, logs = localizer.run_batched(skip=args.skip)
        write_pose_results(poses, paths_cond.results, prepend_camera_name=True)
        with open(f'{paths_cond.results}_logs.pkl', 'wb') as f:
            pickle.dump(logs, f)

    output_path = concat_results(
            all_results, args.conditions, paths.results, 'condition')
    logger.info(
        'Finished evaluating all conditions, you can now submit the file %s to'
        ' https://www.visuallocalization.net/submission/', output_path)


if __name__ == '__main__':
    main()
