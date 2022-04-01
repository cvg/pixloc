"""
A generic training script that works with any model and dataset.
"""

import argparse
from pathlib import Path
import signal
import shutil
import re
import os
import copy
from collections import defaultdict

from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from .datasets import get_dataset
from .models import get_model
from .utils.stdout_capturing import capture_outputs
from .utils.tools import AverageMetric, MedianMetric, set_seed, fork_rng
from .utils.tensor import batch_to_device
from .utils.experiments import (
    delete_old_checkpoints, get_last_checkpoint, get_best_checkpoint)
from ..settings import TRAINING_PATH
from .. import logger


default_train_conf = {
    'seed': '???',  # training seed
    'epochs': 1,  # number of epochs
    'optimizer': 'adam',  # name of optimizer in [adam, sgd, rmsprop]
    'opt_regexp': None,  # regular expression to filter parameters to optimize
    'optimizer_options': {},  # optional arguments passed to the optimizer
    'lr': 0.001,  # learning rate
    'lr_schedule': {'type': None, 'start': 0, 'exp_div_10': 0},
    'lr_scaling': [(100, ['dampingnet.const'])],
    'eval_every_iter': 1000,  # interval for evaluation on the validation set
    'log_every_iter': 200,  # interval for logging the loss to the console
    'keep_last_checkpoints': 10,  # keep only the last X checkpoints
    'load_experiment': None,  # initialize the model from a previous experiment
    'median_metrics': [],  # add the median of some metrics
    'best_key': 'loss/total',  # key to use to select the best checkpoint
    'dataset_callback_fn': None,  # data func called at the start of each epoch
    'clip_grad': None,
}
default_train_conf = OmegaConf.create(default_train_conf)


def do_evaluation(model, loader, device, loss_fn, metrics_fn, conf, pbar=True):
    model.eval()
    results = {}
    for data in tqdm(loader, desc='Evaluation', ascii=True, disable=not pbar):
        data = batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            pred = model(data)
            losses = loss_fn(pred, data)
            metrics = metrics_fn(pred, data)
            del pred, data
        numbers = {**metrics, **{'loss/'+k: v for k, v in losses.items()}}
        for k, v in numbers.items():
            if k not in results:
                results[k] = AverageMetric()
                if k in conf.median_metrics:
                    results[k+'_median'] = MedianMetric()
            results[k].update(v)
            if k in conf.median_metrics:
                results[k+'_median'].update(v)
    results = {k: results[k].compute() for k in results}
    return results


def filter_parameters(params, regexp):
    '''Filter trainable parameters based on regular expressions.'''
    # Examples of regexp:
    #     '.*(weight|bias)$'
    #     'cnn\.(enc0|enc1).*bias'
    def filter_fn(x):
        n, p = x
        match = re.search(regexp, n)
        if not match:
            p.requires_grad = False
        return match
    params = list(filter(filter_fn, params))
    assert len(params) > 0, regexp
    logger.info('Selected parameters:\n'+'\n'.join(n for n, p in params))
    return params


def pack_lr_parameters(params, base_lr, lr_scaling):
    '''Pack each group of parameters with the respective scaled learning rate.
    '''
    filters, scales = tuple(zip(*[
        (n, s) for s, names in lr_scaling for n in names]))
    scale2params = defaultdict(list)
    for n, p in params:
        scale = 1
        # TODO: use proper regexp rather than just this inclusion check
        is_match = [f in n for f in filters]
        if any(is_match):
            scale = scales[is_match.index(True)]
        scale2params[scale].append((n, p))
    logger.info('Parameters with scaled learning rate:\n%s',
                {s: [n for n, _ in ps] for s, ps in scale2params.items()
                 if s != 1})
    lr_params = [{'lr': scale*base_lr, 'params': [p for _, p in ps]}
                 for scale, ps in scale2params.items()]
    return lr_params


def training(rank, conf, output_dir, args):
    if args.restore:
        logger.info(f'Restoring from previous training of {args.experiment}')
        init_cp = get_last_checkpoint(args.experiment, allow_interrupted=False)
        logger.info(f'Restoring from checkpoint {init_cp.name}')
        init_cp = torch.load(str(init_cp), map_location='cpu')
        conf = OmegaConf.merge(OmegaConf.create(init_cp['conf']), conf)
        epoch = init_cp['epoch'] + 1

        # get the best loss or eval metric from the previous best checkpoint
        best_cp = get_best_checkpoint(args.experiment)
        best_cp = torch.load(str(best_cp), map_location='cpu')
        best_eval = best_cp['eval'][conf.train.best_key]
        del best_cp
    else:
        # we start a new, fresh training
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        epoch = 0
        best_eval = float('inf')
        if conf.train.load_experiment:
            logger.info(
                f'Will fine-tune from weights of {conf.train.load_experiment}')
            # the user has to make sure that the weights are compatible
            init_cp = get_last_checkpoint(conf.train.load_experiment)
            init_cp = torch.load(str(init_cp), map_location='cpu')
        else:
            init_cp = None

    OmegaConf.set_struct(conf, True)  # prevent access to unknown entries
    set_seed(conf.train.seed)
    if rank == 0:
        writer = SummaryWriter(log_dir=str(output_dir))

    data_conf = copy.deepcopy(conf.data)
    if args.distributed:
        logger.info(f'Training in distributed mode with {args.n_gpus} GPUs')
        assert torch.cuda.is_available()
        device = rank
        lock = Path(os.getcwd(),
                    f'distributed_lock_{os.getenv("LSB_JOBID", 0)}')
        assert not Path(lock).exists(), lock
        torch.distributed.init_process_group(
                backend='nccl', world_size=args.n_gpus, rank=device,
                init_method='file://'+str(lock))
        torch.cuda.set_device(device)

        # adjust batch size and num of workers since these are per GPU
        if 'batch_size' in data_conf:
            data_conf.batch_size = int(data_conf.batch_size / args.n_gpus)
        if 'train_batch_size' in data_conf:
            data_conf.train_batch_size = int(
                data_conf.train_batch_size / args.n_gpus)
        if 'num_workers' in data_conf:
            data_conf.num_workers = int(
                (data_conf.num_workers + args.n_gpus - 1) / args.n_gpus)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device {device}')

    dataset = get_dataset(data_conf.name)(data_conf)
    if args.overfit:
        # we train and eval with the same single training batch
        logger.info('Data in overfitting mode')
        assert not args.distributed
        train_loader = dataset.get_overfit_loader('train')
        val_loader = dataset.get_overfit_loader('val')
    else:
        train_loader = dataset.get_data_loader(
            'train', distributed=args.distributed)
        val_loader = dataset.get_data_loader('val')
    if rank == 0:
        logger.info(f'Training loader has {len(train_loader)} batches')
        logger.info(f'Validation loader has {len(val_loader)} batches')

    # interrupts are caught and delayed for graceful termination
    def sigint_handler(signal, frame):
        logger.info('Caught keyboard interrupt signal, will terminate')
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        stop = True
    stop = False
    signal.signal(signal.SIGINT, sigint_handler)

    model = get_model(conf.model.name)(conf.model).to(device)
    loss_fn, metrics_fn = model.loss, model.metrics
    if init_cp is not None:
        model.load_state_dict(init_cp['model'])
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device])
    if rank == 0:
        logger.info(f'Model: \n{model}')
    torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)

    optimizer_fn = {'sgd': torch.optim.SGD,
                    'adam': torch.optim.Adam,
                    'rmsprop': torch.optim.RMSprop}[conf.train.optimizer]
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if conf.train.opt_regexp:
        params = filter_parameters(params, conf.train.opt_regexp)
    all_params = [p for n, p in params]

    lr_params = pack_lr_parameters(
            params, conf.train.lr, conf.train.lr_scaling)
    optimizer = optimizer_fn(
            lr_params, lr=conf.train.lr, **conf.train.optimizer_options)
    def lr_fn(it):  # noqa: E306
        if conf.train.lr_schedule.type is None:
            return 1
        if conf.train.lr_schedule.type == 'exp':
            gam = 10**(-1/conf.train.lr_schedule.exp_div_10)
            return 1 if it < conf.train.lr_schedule.start else gam
        else:
            raise ValueError(conf.train.lr_schedule.type)
    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)
    if args.restore:
        optimizer.load_state_dict(init_cp['optimizer'])
        if 'lr_scheduler' in init_cp:
            lr_scheduler.load_state_dict(init_cp['lr_scheduler'])

    if rank == 0:
        logger.info('Starting training with configuration:\n%s',
                    OmegaConf.to_yaml(conf))
    losses_ = None

    while epoch < conf.train.epochs and not stop:
        if rank == 0:
            logger.info(f'Starting epoch {epoch}')
        set_seed(conf.train.seed + epoch)
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        if epoch > 0 and conf.train.dataset_callback_fn:
            getattr(train_loader.dataset, conf.train.dataset_callback_fn)(
                conf.train.seed + epoch)

        for it, data in enumerate(train_loader):
            tot_it = len(train_loader)*epoch + it

            model.train()
            optimizer.zero_grad()
            data = batch_to_device(data, device, non_blocking=True)
            pred = model(data)
            losses = loss_fn(pred, data)
            loss = torch.mean(losses['total'])

            do_backward = loss.requires_grad
            if args.distributed:
                do_backward = torch.tensor(do_backward).float().to(device)
                torch.distributed.all_reduce(
                        do_backward, torch.distributed.ReduceOp.PRODUCT)
                do_backward = do_backward > 0
            if do_backward:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                if conf.train.get('clip_grad', None):
                    if it % conf.train.log_every_iter == 0:
                        grads = [p.grad.data.abs().reshape(-1)
                                 for p in all_params if p.grad is not None]
                        ratio = (torch.cat(grads, 0) > conf.train.clip_grad)
                        ratio = ratio.float().mean().item()*100
                        if ratio > 25:
                            logger.warning(
                                f'More than {ratio:.1f}% of the parameters'
                                ' are larger than the clip value.')
                        del grads, ratio
                    torch.nn.utils.clip_grad_value_(
                            all_params, conf.train.clip_grad)
            else:
                if rank == 0:
                    logger.warning(f'Skip iteration {it} due to detach.')

            if it % conf.train.log_every_iter == 0:
                for k in sorted(losses.keys()):
                    if args.distributed:
                        losses[k] = losses[k].sum()
                        torch.distributed.reduce(losses[k], dst=0)
                        losses[k] /= (train_loader.batch_size * args.n_gpus)
                    losses[k] = torch.mean(losses[k]).item()
                if rank == 0:
                    str_losses = [f'{k} {v:.3E}' for k, v in losses.items()]
                    logger.info('[E {} | it {}] loss {{{}}}'.format(
                        epoch, it, ', '.join(str_losses)))
                    for k, v in losses.items():
                        writer.add_scalar('training/'+k, v, tot_it)
                    writer.add_scalar(
                        'training/lr', optimizer.param_groups[0]['lr'], tot_it)

            del pred, data, loss, losses

            if ((it % conf.train.eval_every_iter == 0) or stop
                    or it == (len(train_loader)-1)):
                with fork_rng(seed=conf.train.seed):
                    results = do_evaluation(
                        model, val_loader, device, loss_fn, metrics_fn,
                        conf.train, pbar=(rank == 0))
                if rank == 0:
                    str_results = [f'{k} {v:.3E}' for k, v in results.items()]
                    logger.info(f'[Validation] {{{", ".join(str_results)}}}')
                    for k, v in results.items():
                        writer.add_scalar('val/'+k, v, tot_it)
                torch.cuda.empty_cache()  # should be cleared at the first iter

            if stop:
                break

        if rank == 0:
            state = (model.module if args.distributed else model).state_dict()
            checkpoint = {
                'model': state,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'conf': OmegaConf.to_container(conf, resolve=True),
                'epoch': epoch,
                'losses': losses_,
                'eval': results,
            }
            cp_name = f'checkpoint_{epoch}' + ('_interrupted' if stop else '')
            logger.info(f'Saving checkpoint {cp_name}')
            cp_path = str(output_dir / (cp_name + '.tar'))
            torch.save(checkpoint, cp_path)
            if results[conf.train.best_key] < best_eval:
                best_eval = results[conf.train.best_key]
                logger.info(
                    f'New best checkpoint: {conf.train.best_key}={best_eval}')
                shutil.copy(cp_path, str(output_dir / 'checkpoint_best.tar'))
            delete_old_checkpoints(
                output_dir, conf.train.keep_last_checkpoints)
            del checkpoint

        epoch += 1

    logger.info(f'Finished training on process {rank}.')
    if rank == 0:
        writer.close()


def main_worker(rank, conf, output_dir, args):
    if rank == 0:
        with capture_outputs(output_dir / 'log.txt'):
            training(rank, conf, output_dir, args)
    else:
        training(rank, conf, output_dir, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--conf', type=str)
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('dotlist', nargs='*')
    args = parser.parse_args()

    logger.info(f'Starting experiment {args.experiment}')
    output_dir = Path(TRAINING_PATH, args.experiment)
    output_dir.mkdir(exist_ok=True, parents=True)

    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)
    if not args.restore:
        if conf.train.seed is None:
            conf.train.seed = torch.initial_seed() & (2**32 - 1)
        OmegaConf.save(conf, str(output_dir / 'config.yaml'))

    if args.distributed:
        args.n_gpus = torch.cuda.device_count()
        torch.multiprocessing.spawn(
            main_worker, nprocs=args.n_gpus,
            args=(conf, output_dir, args))
    else:
        main_worker(0, conf, output_dir, args)
