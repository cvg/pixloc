import torch


def masked_mean(x, mask, dim):
    mask = mask.float()
    return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)


def checkpointed(cls, do=True):
    '''Adapted from the DISK implementation of Michał Tyszkiewicz.'''
    assert issubclass(cls, torch.nn.Module)

    class Checkpointed(cls):
        def forward(self, *args, **kwargs):
            super_fwd = super(Checkpointed, self).forward
            if any((torch.is_tensor(a) and a.requires_grad) for a in args):
                return torch.utils.checkpoint.checkpoint(
                        super_fwd, *args, **kwargs)
            else:
                return super_fwd(*args, **kwargs)

    return Checkpointed if do else cls
