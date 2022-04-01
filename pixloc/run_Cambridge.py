import pickle

from . import set_logging_debug, logger
from .localization import RetrievalLocalizer, PoseLocalizer
from .utils.data import Paths, create_argparser, parse_paths, parse_conf
from .utils.io import write_pose_results
from .utils.eval import evaluate

default_paths = Paths(
    query_images='{scene}/',
    reference_images='{scene}/',
    reference_sfm='{scene}/sfm_superpoint+superglue/',
    query_list='{scene}/query_list_with_intrinsics.txt',
    retrieval_pairs='{scene}/pairs-query-netvlad10.txt',
    ground_truth='CambridgeLandmarks_Colmap_Retriangulated_1024px/{scene}/',
    results='pixloc_Cambridge_{scene}.txt',
)

experiment = 'pixloc_megadepth'

default_confs = {
    'from_retrieval': {
        'experiment': experiment,
        'features': {},
        'optimizer': {
            'num_iters': 100,
            'pad': 2,  # to 1?
        },
        'refinement': {
            'num_dbs': 5,
            'multiscale': [4, 1],
            'point_selection': 'all',
            'normalize_descriptors': True,
            'average_observations': True,
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
            'average_observations': True,
            'layer_indices': [0, 1],
        },
    },
}

SCENES = ['ShopFacade', 'KingsCollege', 'GreatCourt', 'OldHospital',
          'StMarysChurch']


def main():
    parser = create_argparser('Cambridge')
    parser.add_argument('--scenes', default=SCENES, choices=SCENES, nargs='+')
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    set_logging_debug(args.verbose)
    paths = parse_paths(args, default_paths)
    conf = parse_conf(args, default_confs)

    all_poses = {}
    for scene in args.scenes:
        logger.info('Working on scene %s.', scene)
        paths_scene = paths.interpolate(scene=scene)
        if args.eval_only and paths_scene.results.exists():
            all_poses[scene] = paths_scene.results
            continue

        if args.from_poses:
            localizer = PoseLocalizer(paths_scene, conf)
        else:
            localizer = RetrievalLocalizer(paths_scene, conf)
        poses, logs = localizer.run_batched(skip=args.skip)
        write_pose_results(poses, paths_scene.results,
                           prepend_camera_name=True)
        with open(f'{paths_scene.results}_logs.pkl', 'wb') as f:
            pickle.dump(logs, f)
        all_poses[scene] = poses

    for scene in args.scenes:
        paths_scene = paths.interpolate(scene=scene)
        logger.info('Evaluate scene %s: %s', scene, paths_scene.results)
        evaluate(paths_scene.ground_truth / 'empty_all', all_poses[scene],
                 paths_scene.ground_truth / 'list_query.txt',
                 only_localized=(args.skip is not None and args.skip > 1))


if __name__ == '__main__':
    main()
