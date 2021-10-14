import torch
import numpy as np
import functools


def torchify(func):
    """Extends to NumPy arrays a function written for PyTorch tensors.

    Converts input arrays to tensors and output tensors back to arrays.
    Supports hybrid inputs where some are arrays and others are tensors:
    - in this case all tensors should have the same device and float dtype;
    - the output is not converted.

    No data copy: tensors and arrays share the same underlying storage.

    Warning: kwargs are currently not supported when using jit.
    """
    # TODO: switch to  @torch.jit.unused when is_scripting will work
    @torch.jit.ignore
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        device = None
        dtype = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device_ = arg.device
                if device is not None and device != device_:
                    raise ValueError(
                        'Two input tensors have different devices: '
                        f'{device} and {device_}')
                device = device_
                if torch.is_floating_point(arg):
                    dtype_ = arg.dtype
                    if dtype is not None and dtype != dtype_:
                        raise ValueError(
                            'Two input tensors have different float dtypes: '
                            f'{dtype} and {dtype_}')
                    dtype = dtype_

        args_converted = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg).to(device)
                if torch.is_floating_point(arg):
                    arg = arg.to(dtype)
            args_converted.append(arg)

        rets = func(*args_converted, **kwargs)

        def convert_back(ret):
            if isinstance(ret, torch.Tensor):
                if device is None:  # no input was torch.Tensor
                    ret = ret.cpu().numpy()
            return ret

        # TODO: handle nested struct with map tensor
        if not isinstance(rets, tuple):
            rets = convert_back(rets)
        else:
            rets = tuple(convert_back(ret) for ret in rets)
        return rets

    # BUG: is_scripting does not work in 1.6 so wrapped is always called
    if torch.jit.is_scripting():
        return func
    else:
        return wrapped
