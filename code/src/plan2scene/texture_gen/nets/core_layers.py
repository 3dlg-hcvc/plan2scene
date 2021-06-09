# Code adapted from https://github.com/madhawav/Rent3D-to-STK/blob/master/src/models/core_layers.py

import torch
import functools


def normalization(type):
    if type == 'batch2d':
        return functools.partial(torch.nn.BatchNorm2d)
    elif type == 'batch3d':
        return functools.partial(torch.nn.BatchNorm3d)
    elif type == 'inst2d':
        return functools.partial(torch.nn.InstanceNorm2d)
    elif type == 'inst3d':
        return functools.partial(torch.nn.InstanceNorm3d)
    elif type == 'spectral':
        return torch.nn.utils.spectral_norm
    elif type == 'none' or type is None:
        return functools.partial(torch.nn.Identity)
    else:
        raise NotImplementedError('Normalization {} is not implemented'.format(type))


def non_linearity(type):
    if type == 'relu':
        return functools.partial(torch.nn.ReLU)
    elif type == 'lrelu':
        return functools.partial(torch.nn.LeakyReLU, negative_slope=0.2)
    elif type == 'elu':
        return functools.partial(torch.nn.ELU)
    elif type == 'none' or type is None:
        return functools.partial(torch.nn.Identity)
    else:
        raise NotImplementedError('Nonlinearity {} is not implemented'.format(type))