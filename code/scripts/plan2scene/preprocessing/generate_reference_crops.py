#!/usr/bin/python3
from plan2scene.common.house_parser import parse_houses, save_house_crops, map_surface_crops_to_house
from plan2scene.common.residence import Room
from plan2scene.config_manager import ConfigManager
import logging
import os.path as osp
import os
from plan2scene.utils.image_util import get_medoid_key

if __name__ == "__main__":
    """
    Identifies the ground truth reference crop for each surface. These reference crops are used by the quantitative evaluations.
    """

    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Identify ground truth reference crop for each surface.")
    conf.add_args(parser)
    parser.add_argument("output_path",
                        help="Output path to create the ground truth reference crops. "
                             "Usually './data/processed/gt_reference/[split]'.")
    parser.add_argument("photoroom_path",
                        help="Path to photoroom.csv files. "
                             "Usually './data/input/photo_assignments/[split]'.")
    parser.add_argument("split", help="train/val/test")
    args = parser.parse_args()
    conf.process_args(args)

    output_path = args.output_path
    split = args.split
    photoroom_path = args.photoroom_path

    if not osp.exists(output_path):
        os.makedirs(output_path)

    if not osp.exists(osp.join(output_path, "texture_crops")):
        os.mkdir(osp.join(output_path, "texture_crops"))

    house_keys = conf.get_data_list(split)

    houses = parse_houses(conf, house_keys,
                          house_path_spec=conf.data_paths.arch_path_spec.format(split=split, house_key="{house_key}"),
                          photoroom_csv_path_spec=osp.join(photoroom_path, "{house_key}.photoroom.csv"))

    for i, (house_key, house) in enumerate(houses.items()):
        logging.info("[%d/%d] Processing %s" % (i, len(houses), house_key))
        map_surface_crops_to_house(conf, house)
        for room_index, room in house.rooms.items():
            assert isinstance(room, Room)
            for surface in conf.surfaces:
                if len(room.surface_textures[surface]) > 0:
                    medoid_key = get_medoid_key(room.surface_textures[surface])
                    room.surface_textures[surface] = {medoid_key: room.surface_textures[surface][medoid_key]}
        save_house_crops(house, save_path=osp.join(output_path, "texture_crops", house_key))
