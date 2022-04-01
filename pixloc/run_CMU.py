import pickle

from . import set_logging_debug, logger
from .localization import RetrievalLocalizer, PoseLocalizer
from .utils.data import Paths, create_argparser, parse_paths, parse_conf
from .utils.io import write_pose_results, concat_results


default_paths = Paths(
    query_images='slice{slice}/query/',
    reference_images='slice{slice}/database',
    reference_sfm='slice{slice}/sfm_superpoint+superglue/model/',
    query_list='slice{slice}/queries_with_intrinsics.txt',
    global_descriptors='slice{slice}/cmu-slice{slice}_tf-netvlad.h5',
    retrieval_pairs='slice{slice}/pairs-query-netvlad10.txt',
    hloc_logs='slice{slice}/CMU_hloc_superpoint+superglue_netvlad10.txt_logs.pkl',
    results='pixloc_CMU_slice{slice}.txt',
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

TEST_URBAN = [2, 3, 4, 5, 6]
TEST_SUBURBAN = [13, 14, 15, 16, 17]
TEST_PARK = [18, 19, 20, 21]
TEST_SLICES_CMU = TEST_URBAN + TEST_SUBURBAN + TEST_PARK
TRAINING_SLICES_CMU = [7, 8, 9, 10, 11, 12, 22, 23, 24, 25]


def generate_query_list(paths, slice_):
    cameras = {}
    with open(paths.dataset / 'intrinsics.txt', 'r') as f:
        for line in f.readlines():
            if line[0] == '#' or line == '\n':
                continue
            data = line.split()
            cameras[data[0]] = data[1:]
    assert len(cameras) == 2

    queries = paths.dataset / f'slice{slice_}/test-images-slice{slice_}.txt'
    with open(queries, 'r') as f:
        queries = [q.rstrip('\n') for q in f.readlines()]

    out = [[q] + cameras[q.split('_')[2]] for q in queries]
    with open(paths.query_list, 'w') as f:
        f.write('\n'.join(map(' '.join, out)))


def parse_slice_arg(slice_str):
    if slice_str is None:
        slices = TEST_SLICES_CMU
        logger.info(
            'No slice list given, will evaluate all %d test slices; '
            'this might take a long time.', len(slices))
    elif '-' in slice_str:
        min_, max_ = slice_str.split('-')
        slices = list(range(int(min_), int(max_)+1))
    else:
        slices = eval(slice_str)
        if isinstance(slices, int):
            slices = [slices]
    return slices


def main():
    parser = create_argparser('CMU')
    parser.add_argument('--slices', type=str,
                        help='a single number, an interval (e.g. 2-6), '
                        'or a Python-style list or int (e.g. [2, 3, 4]')
    args = parser.parse_args()

    set_logging_debug(args.verbose)
    paths = parse_paths(args, default_paths)
    conf = parse_conf(args, default_confs)
    slices = parse_slice_arg(args.slices)

    all_results = []
    logger.info('Will evaluate slices %s.', slices)
    for slice_ in slices:
        logger.info('Working on slice %s.', slice_)
        paths_slice = paths.interpolate(slice=slice_)
        all_results.append(paths_slice.results)
        if paths_slice.results.exists():
            continue
        if not paths_slice.query_list.exists():
            generate_query_list(paths_slice, slice_)

        if args.from_poses:
            localizer = PoseLocalizer(paths_slice, conf)
        else:
            localizer = RetrievalLocalizer(paths_slice, conf)
        poses, logs = localizer.run_batched(skip=args.skip)
        write_pose_results(poses, paths_slice.results)
        with open(f'{paths_slice.results}_logs.pkl', 'wb') as f:
            pickle.dump(logs, f)

    output_path = concat_results(all_results, slices, paths.results, 'slice')
    logger.info(
        'Finished evaluating all slices, you can now submit the file %s to '
        'https://www.visuallocalization.net/submission/', output_path)


if __name__ == '__main__':
    main()
