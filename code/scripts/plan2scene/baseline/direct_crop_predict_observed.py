#!/usr/bin/python3
from plan2scene.common.house_parser import parse_houses, save_house_crops, map_surface_crops_to_houses, map_surface_crops_to_house
from plan2scene.common.image_description import ImageDescription, ImageSource
from plan2scene.common.residence import Room
from plan2scene.config_manager import ConfigManager
import logging
import os.path as osp
import os
from PIL import Image
import torch
import numpy as np

from plan2scene.utils.image_util import get_medoid_key
from plan2scene.utils.io import load_image


if __name__ == "__main__":
    """
    Predict textures for observed surfaces using the direct crop approach. We use the same strategy taken to choose the ground truth reference crop.
    """

    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Predict textures for observed surfaces using the direct crop approach.")
    conf.add_args(parser)
    parser.add_argument("output_path",
                        help="Output path to save the predicted crops."
                             "Usually './data/processed/baselines/direct_crop/observed/[split]/drop_[drop_fraction]'.")
    parser.add_argument("split", help="val/test")
    args = parser.parse_args()
    conf.process_args(args, output_is_dir=True)

    output_path = args.output_path
    split = args.split

    if not osp.exists(output_path):
        os.makedirs(output_path)

    if not osp.exists(osp.join(output_path, "texture_crops")):
        os.mkdir(osp.join(output_path, "texture_crops"))

    house_keys = conf.get_data_list(split)

    houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=split,
                                                                                                  house_key="{house_key}"),
                          photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=split,
                                                                                             drop_fraction=conf.drop_fraction,
                                                                                             house_key="{house_key}"))
    map_surface_crops_to_houses(conf, houses)

    for i, (house_key, house) in enumerate(houses.items()):
        logging.info("[%d/%d] Processing %s" % (i, len(houses), house_key))
        map_surface_crops_to_house(conf, house)
        for room_index, room in house.rooms.items():
            assert isinstance(room, Room)
            for surface in conf.surfaces:
                if len(room.surface_textures[surface]) > 0:
                    medoid_key = get_medoid_key(room.surface_textures[surface])
                    room.surface_textures[surface] = {"prop": room.surface_textures[surface][medoid_key]}
                else:
                    room.surface_textures[surface] = {}

        save_house_crops(house, save_path=osp.join(output_path, "texture_crops", house_key))
