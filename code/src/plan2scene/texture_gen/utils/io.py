# Code adapted from https://raw.githubusercontent.com/henzler/neuraltexture/master/code/utils/io.py
import logging
import numpy as np
import random
import torch
import yaml
from orderedattrdict.yamlutils import AttrDictYAMLLoader
import os
import os.path as osp
import torchvision


def load_config_train(config_path: str):
    """
    Load a neural texture synthesis config file for training purposes.
    :param config_path: Path to config file.
    :return: Loaded config
    """
    config = load_config(config_path)

    # set seed
    seed = config['train']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    param = _update_config_common(config, config_path)
    return param


def _update_config_common(config):
    """
    Load dependent configs.
    :param config: Neural texture synthesis config.
    :return: Updated config.
    """
    config['texture']['channels'] = 3  # RGB
    config['texture']['t'] = config.dim ** 2 * config.noise.octaves
    config['texture']['z'] = config.texture.e + config.texture.t
    config['system']['arch']['model_texture_encoder']['model_params']['z'] = config.texture.z
    shape_in = [[config.texture.e + config.noise.octaves * config.texture.channels * 2, config.image.image_res,
                 config.image.image_res]]

    shape_out = [[config.texture.channels, config.image.image_res, config.image.image_res]]

    config['system']['arch']['model_texture_decoder']['model_params'][
        'noise'] = config.noise.octaves * config.dim * config.texture.channels
    config['system']['arch']['model_texture_decoder']['model_params']['shape_in'] = shape_in
    config['system']['arch']['model_texture_decoder']['model_params']['shape_out'] = shape_out

    param = config

    if param.device == 'cuda' and not torch.cuda.is_available():
        raise Exception('No GPU found, please use "cpu" as device')

    return param


def load_conf_eval(config_path: str):
    """
    Load neural texture synthesis config for evaluation purposes.
    :param config_path: Path to saved config file.
    :return: Loaded config
    """
    config = load_config(config_path)

    conf = _update_config_common(config=config)
    conf["train"]["bs"] = 1
    return conf


def load_config(config_path: str):
    """
    Loaded neural texture config from disk.
    :param config_path: Path to config file.
    :return: Loaded config.
    """
    logging.info('Using PyTorch {}'.format(torch.__version__))
    logging.info('Load config: {}'.format(config_path))

    config = yaml.load(open(str(config_path)), Loader=AttrDictYAMLLoader)
    return config


def preview_images(images: torch.Tensor, ncol: int) -> torch.Tensor:
    """
    Preview images using a grid.
    :param images: Images to preview.
    :param ncol: Number of columns
    :return: Images concatenated into a grid.
    """
    images = images.clone()
    prev_image = torchvision.utils.make_grid(images, ncol)
    return prev_image


def preview_deltas(signed_delta_image: torch.Tensor) -> list:
    """
    Preview individual channels (RGB/HSV) of a delta image.
    :param signed_delta_image: Batch of images to preview [batch_size, channel_count, height, width]
    :return: List of tensors, one per each channel.
    """
    outputs = []
    for channel in range(signed_delta_image.shape[1]):
        pallet = signed_delta_image[:, channel:channel + 1, :, :].repeat([1, 2, 1, 1]).permute(1, 0, 2, 3)
        pallet[0, pallet[0] > 0] = 0.0
        pallet[0, :] = -pallet[0, :]
        pallet[1, pallet[1] < 0] = 0.0
        pallet *= 1
        pallet = torch.cat(
            [pallet[0:1].cpu(), torch.zeros((1, pallet.shape[1], pallet.shape[2], pallet.shape[3])), pallet[1:2].cpu()])
        pallet = pallet.permute(1, 0, 2, 3)
        outputs.append(pallet)

    return outputs