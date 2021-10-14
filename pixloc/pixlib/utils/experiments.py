"""
A set of utilities to manage and load checkpoints of training experiments.
"""

from pathlib import Path
import logging
import re
from omegaconf import OmegaConf
import torch
import os

from ...settings import TRAINING_PATH
from ..models import get_model

logger = logging.getLogger(__name__)


def list_checkpoints(dir_):
    """List all valid checkpoints in a given directory."""
    checkpoints = []
    for p in dir_.glob('checkpoint_*.tar'):
        numbers = re.findall(r'(\d+)', p.name)
        if len(numbers) == 0:
            continue
        assert len(numbers) == 1
        checkpoints.append((int(numbers[0]), p))
    return checkpoints


def get_last_checkpoint(exper, allow_interrupted=True):
    """Get the last saved checkpoint for a given experiment name."""
    ckpts = list_checkpoints(Path(TRAINING_PATH, exper))
    if not allow_interrupted:
        ckpts = [(n, p) for (n, p) in ckpts if '_interrupted' not in p.name]
    assert len(ckpts) > 0
    return sorted(ckpts)[-1][1]


def get_best_checkpoint(exper):
    """Get the checkpoint with the best loss, for a given experiment name."""
    p = Path(TRAINING_PATH, exper, 'checkpoint_best.tar')
    return p


def delete_old_checkpoints(dir_, num_keep):
    """Delete all but the num_keep last saved checkpoints."""
    ckpts = list_checkpoints(dir_)
    ckpts = sorted(ckpts)[::-1]
    kept = 0
    for ckpt in ckpts:
        if ('_interrupted' in str(ckpt[1]) and kept > 0) or kept >= num_keep:
            logger.info(f'Deleting checkpoint {ckpt[1].name}')
            ckpt[1].unlink()
        else:
            kept += 1


def load_experiment(exper, conf={}, get_last=False):
    """Load and return the model of a given experiment."""
    if get_last:
        ckpt = get_last_checkpoint(exper)
    else:
        ckpt = get_best_checkpoint(exper)
    logger.info(f'Loading checkpoint {ckpt.name}')
    ckpt = torch.load(str(ckpt), map_location='cpu')

    loaded_conf = OmegaConf.create(ckpt['conf'])
    OmegaConf.set_struct(loaded_conf, False)
    conf = OmegaConf.merge(loaded_conf.model, OmegaConf.create(conf))
    model = get_model(conf.name)(conf).eval()

    state_dict = ckpt['model']
    dict_params = set(state_dict.keys())
    model_params = set(map(lambda n: n[0], model.named_parameters()))
    diff = model_params - dict_params
    if len(diff) > 0:
        subs = os.path.commonprefix(list(diff)).rstrip('.')
        logger.warning(f'Missing {len(diff)} parameters in {subs}')
    model.load_state_dict(state_dict, strict=False)
    return model


def flexible_load(state_dict, model):
    """TODO: fix a probable nasty bug, and move to BaseModel."""
    dict_params = set(state_dict.keys())
    model_params = set(map(lambda n: n[0], model.named_parameters()))

    if dict_params == model_params:  # prefect fit
        logger.info('Loading all parameters of the checkpoint.')
        model.load_state_dict(state_dict, strict=True)
        return
    elif len(dict_params & model_params) == 0:  # perfect mismatch
        strip_prefix = lambda x: '.'.join(x.split('.')[:1]+x.split('.')[2:])
        state_dict = {strip_prefix(n): p for n, p in state_dict.items()}
        dict_params = set(state_dict.keys())
        if len(dict_params & model_params) == 0:
            raise ValueError('Could not manage to load the checkpoint with'
                             'parameters:' + '\n\t'.join(sorted(dict_params)))
    common_params = dict_params & model_params
    left_params = dict_params - model_params
    logger.info('Loading parameters:\n\t'+'\n\t'.join(sorted(common_params)))
    if len(left_params) > 0:
        logger.info('Could not load parameters:\n\t'
                    + '\n\t'.join(sorted(left_params)))
    model.load_state_dict(state_dict, strict=False)
