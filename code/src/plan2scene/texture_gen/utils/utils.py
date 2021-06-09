# Code adapted from https://github.com/henzler/neuraltexture/blob/master/code/utils/utils.py

import numpy as np
import torch
import torchvision


def get_grid_coords_2d(y, x, coord_dim=-1):
    y, x = torch.meshgrid(y, x)
    coords = torch.stack([x, y], dim=coord_dim)
    return coords


def get_grid_coords_3d(z, y, x, coord_dim=-1):
    z, y, x = torch.meshgrid(z, y, x)
    coords = torch.stack([x, y, z], dim=coord_dim)
    return coords


def signed_to_unsigned(array):
    """
    Converts a signed tensor to unsigned.
    :param array:
    :return:
    """
    return (array + 1) / 2


def unsigned_to_signed(array):
    """
    Converts an unsigned tensor to signed.
    :param array:
    :return:
    """
    return (array - 0.5) / 0.5


def pytorch_to_numpy(array, is_batch=True, flip=True):
    array = array.detach().cpu().numpy()

    if flip:
        source = 1 if is_batch else 0
        dest = array.ndim - 1
        array = np.moveaxis(array, source, dest)

    return array


def numpy_to_pytorch(array, is_batch=False, flip=True):
    if flip:
        dest = 1 if is_batch else 0
        source = array.ndim - 1
        array = np.moveaxis(array, source, dest)

    array = torch.from_numpy(array)
    array = array.float()

    return array


def convert_to_int(array):
    array *= 255
    array[array > 255] = 255.0

    if type(array).__module__ == 'numpy':
        return array.astype(np.uint8)

    elif type(array).__module__ == 'torch':
        return array.byte()
    else:
        raise NotImplementedError


def convert_to_float(array):
    max_value = np.iinfo(array.dtype).max
    array[array > max_value] = max_value

    if type(array).__module__ == 'numpy':
        return array.astype(np.float32) / max_value

    elif type(array).__module__ == 'torch':
        return array.float() / max_value
    else:
        raise NotImplementedError


def metric_mse(output, target):
    return torch.nn.functional.mse_loss(output, target).mean().item()


def dict_to_keyvalue(params, prefix=''):
    hparams = {}

    for key, value in params.items():
        if isinstance(value, dict):
            if not prefix == '':
                new_prefix = '{}.{}'.format(prefix, key)
            else:
                new_prefix = key
            hparams.update(dict_to_keyvalue(value, prefix=new_prefix))
        else:
            if not prefix == '':
                key = '{}.{}'.format(prefix, key)
            hparams[key] = value

    return hparams


def dict_mean(dict_list):
    mean_dict = {}
    dict_item = dict_list[0]

    for key in dict_list[0].keys():
        if isinstance(dict_item[key], dict):
            for key2 in dict_item[key].keys():
                if not mean_dict.__contains__(key):
                    mean_dict[key] = {}
                mean_dict[key][key2] = sum(d[key][key2] for d in dict_list) / len(dict_list)
        else:
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
