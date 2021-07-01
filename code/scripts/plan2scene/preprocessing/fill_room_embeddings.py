#!/usr/bin/python3
from plan2scene.common.house_parser import parse_houses, save_house_texture_embeddings, save_house_crops
from plan2scene.common.image_description import ImageDescription, ImageSource
from plan2scene.common.residence import Room, House
from plan2scene.config_manager import ConfigManager
import os
import os.path as osp

from plan2scene.crop_select.util import fill_texture_embeddings
from plan2scene.texture_gen.utils.io import load_conf_eval
from plan2scene.utils.io import load_image

import logging
from plan2scene.texture_gen.predictor import TextureGenPredictor

def process(conf: ConfigManager, houses: dict, output_path: str) -> None:
    """
    Parse surface crops of given houses. Then, compute texture embeddings.
    :param conf: Config Manager
    :param houses: Dictionary of houses.
    :param output_path: Path to save surface crops and texture embeddings.
    """
    if not osp.exists(osp.join(output_path, "surface_texture_embeddings")):
        os.mkdir(osp.join(output_path, "surface_texture_embeddings"))

    if not osp.exists(osp.join(output_path, "texture_crops")):
        os.mkdir(osp.join(output_path, "texture_crops"))

    predictor = TextureGenPredictor(
        conf=load_conf_eval(config_path=conf.texture_gen.texture_synth_conf),
        rgb_median_emb=conf.texture_gen.rgb_median_emb)
    predictor.load_checkpoint(checkpoint_path=conf.texture_gen.checkpoint_path)

    for i, (house_key, house) in enumerate(houses.items()):
        logging.info("[%d/%d] Processing %s" % (i, len(houses), house_key))
        fill_texture_embeddings(conf, house, predictor)
        save_house_texture_embeddings(house, save_path=osp.join(output_path, "surface_texture_embeddings", house_key + ".json"))
        save_house_crops(house, save_path=osp.join(output_path, "texture_crops", house_key))


if __name__ == "__main__":
    """
    Parse crop surface assignments. Then, compute texture embeddings of surfaces. 
    We pass each crop assigned to every surface of a house through the neural texture synthesis encoder and obtain a latent embedding.
    This latent embedding is assigned as a texture embedding of the considered surface.
    """
    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Compute texture embeddings of houses.")
    conf.add_args(parser)
    parser.add_argument("output_path",
                        help="Output path to save texture embeddings and assigned crops."
                             "Usually './data/processed/texture_gen/[split]/drop_[drop_fraction]'.")
    parser.add_argument("split", help="train/val/test")
    args = parser.parse_args()
    conf.process_args(args)

    output_path = args.output_path
    split = args.split

    if not osp.exists(output_path):
        os.makedirs(output_path)

    house_keys = conf.get_data_list(split)

    houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=split,
                                                                                                  house_key="{house_key}"),
                          photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=split,
                                                                                             drop_fraction=conf.drop_fraction,
                                                                                             house_key="{house_key}"))
    process(conf, houses, output_path)
