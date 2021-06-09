#!/bin/python3

from plan2scene.common.house_parser import parse_houses, load_house_texture_embeddings, save_house_crops, save_house_texture_embeddings
from plan2scene.common.image_description import ImageSource
from plan2scene.common.residence import House, Room
from plan2scene.config_manager import ConfigManager
import os.path as osp
import os
import logging
import torch

from plan2scene.crop_select.util import fill_textures
from plan2scene.texture_gen.predictor import TextureGenPredictor
from plan2scene.texture_gen.utils.io import load_conf_eval


def process_house(conf: ConfigManager, house: House, predictor: TextureGenPredictor) -> None:
    """
    Predicts textures for observed surface of a house using the mean embedding approach.
    :param conf: Config manager
    :param house: House applied with new textures
    :param predictor: Predictor used to synthesize textures
    """
    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        # Calculate the mean embs
        for surface in conf.surfaces:
            if len(room.surface_embeddings[surface]) > 0:
                room.surface_embeddings[surface] = {"prop": torch.mean(torch.cat(list(room.surface_embeddings[surface].values())), dim=0).unsqueeze(0)}
            else:
                room.surface_embeddings[surface] = {}

    fill_textures(conf, {house.house_key: house}, predictor=predictor, log=False, image_source=ImageSource.MEAN_EMB, skip_existing_textures=False)


if __name__ == "__main__":
    """
    Predict textures for observed surfaces using the NaiveSynth approach.
    """

    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Predict textures for observed surfaces using the naive synth approach.")
    conf.add_args(parser)
    parser.add_argument("output_path",
                        help="Output path to save the predicted texture embeddings and crops."
                             "Usually './data/processed/baselines/naivesynth/observed/[split]/drop_[drop_fraction]'.")
    parser.add_argument("texture_gen_path", help="Path to saved texture_gen embeddings and crops. "
                                                 "Usually './data/processed/baselines/naivesynth/texture_gen/[split]/drop_[drop_fraction]'.")
    parser.add_argument("split", help="val/test")
    args = parser.parse_args()
    conf.process_args(args, output_is_dir=True)

    output_path = args.output_path
    split = args.split
    texture_gen_path = args.texture_gen_path

    if not osp.exists(osp.join(output_path, "texture_crops")):
        os.mkdir(osp.join(output_path, "texture_crops"))

    if not osp.exists(osp.join(output_path, "surface_texture_embeddings")):
        os.mkdir(osp.join(output_path, "surface_texture_embeddings"))

        # Load houses
    house_keys = conf.get_data_list(split)
    houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=split,
                                                                                                  house_key="{house_key}"),
                          photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=split,
                                                                                             drop_fraction=conf.drop_fraction,
                                                                                             house_key="{house_key}"))

    # Load checkpoint
    predictor = TextureGenPredictor(
        conf=load_conf_eval(config_path=conf.texture_gen.texture_synth_conf),
        rgb_median_emb=conf.texture_gen.rgb_median_emb)
    predictor.load_checkpoint(checkpoint_path=conf.texture_gen.checkpoint_path)

    # Process houses
    for i, (house_key, house) in enumerate(houses.items()):
        assert isinstance(house, House)
        logging.info("[%d/%d] Processing %s" % (i, len(houses), house_key))
        load_house_texture_embeddings(house,
                                      osp.join(texture_gen_path, "surface_texture_embeddings", house_key + ".json"))
        process_house(conf, house, predictor)
        save_house_crops(house, osp.join(output_path, "texture_crops", house_key))
        save_house_texture_embeddings(house, osp.join(output_path, "surface_texture_embeddings", house_key + ".json"))
