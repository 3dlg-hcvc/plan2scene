from plan2scene.common.house_parser import parse_houses, load_houses_with_textures, map_surface_crops_to_houses, save_house_crops
from plan2scene.common.residence import Room, House
from plan2scene.config_manager import ConfigManager
import logging

from plan2scene.utils.image_util import get_medoid_key
import os.path as osp

import os
import numpy as np


def load_houses_with_crops(conf: ConfigManager, split: str, drop: str) -> dict:
    """
    Load houses populated with rectified surface crops extracted from photos.
    :param conf: Config manager
    :param split: Train/val/test split
    :param drop: Drop fraction
    :return: Dictionary of houses.
    """
    house_keys = conf.get_data_list(split)
    houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=split,
                                                                                                  house_key="{house_key}"),
                          photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=split,
                                                                                             drop_fraction=drop,
                                                                                             house_key="{house_key}"))
    map_surface_crops_to_houses(conf, houses)
    return houses


def collect_house_crops(conf: ConfigManager, house: House, house_rs_crops=None) -> dict:
    """
    Compute a nested dictionary for the mapping room_type->surface_type->list of crops.
    :param conf: Config Manager
    :param house: House to consider
    :param house_rs_crops: If provided, update this mapping.
    :return: Computed mapping
    """
    if house_rs_crops is None:
        house_rs_crops = {}
    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        room_type = frozenset(room.types)

        if room_type not in house_rs_crops:
            house_rs_crops[room_type] = {v: [] for v in conf.surfaces}

        for surface in conf.surfaces:
            if len(room.surface_textures[surface]) > 0:
                if "prop" in room.surface_textures[surface]:
                    house_rs_crops[room_type][surface].append(room.surface_textures[surface]["prop"])
                else:
                    house_rs_crops[room_type][surface].append(
                        room.surface_textures[surface][get_medoid_key({k: v for k, v in room.surface_textures[surface].items()})])

    return house_rs_crops


def collect_train_set_crops(conf: ConfigManager) -> dict:
    """
    Collect crops from train set so we can use them in predictions.
    :param conf: Config manager.
    :return: Nested dictionary for the mapping room_type->surface_type->list of crops.
    """
    train_houses = load_houses_with_crops(conf, "train", "0.0")
    train_rs_crops = {}  # {k:{v: [] for v in conf.surfaces} for k in conf.room_types}
    for i, (house_key, house) in enumerate(train_houses.items()):
        logging.info("[%d/%d] Processing TrainSet: %s" % (i, len(train_houses), house_key))
        collect_house_crops(conf, house, train_rs_crops)
    return train_rs_crops


def pick_texture(train_rs_crops: dict, train_s_crops: dict, house_rs_crops: dict, room_type, surface_type):
    """
    Pick a suitable texture given a room type and a surface type.
    :param train_rs_crops: Room type -> surface type -> textures mapping from train set
    :param train_s_crops: Surface type -> textures mapping from the train set
    :param house_rs_crops: Room type -> surface type -> textures mapping from the observed surfaces of the house
    :param room_type: Room type
    :param surface_type: Surface type
    :return: Selected texture
    """
    # First check whether we can pick a crop from the same house itself. For that, we need a similar RS surface with textures in the same house.
    if room_type in house_rs_crops and len(house_rs_crops[room_type][surface_type]) > 0:
        index = np.random.randint(len(house_rs_crops[room_type][surface_type]))
        return house_rs_crops[room_type][surface_type][index]

    elif room_type in train_rs_crops and len(train_rs_crops[room_type][surface_type]) > 0:
        # If we cant pick from the same house, try picking a similar RS surface from the train set.
        index = np.random.randint(len(train_rs_crops[room_type][surface_type]))
        return train_rs_crops[room_type][surface_type][index]
    else:
        # If we can't find a similar RS crop in the train set, just pick a similar S crop from the trainset
        index = np.random.randint(len(train_s_crops[surface_type]))
        return train_s_crops[surface_type][index]


if __name__ == "__main__":
    """
    Predict textures for unobserved surfaces using the direct crop approach.
    First, try to pick from another surface of the same house that has same RS. 
    If no match, try to pick from a surface in the training set that has the same RS.
    If no match, pick from a surface in the training set of same S.
    """
    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(
        description="Predict textures for unobserved surfaces using the direct crop approach. We pick a crop from a similar RS surface. First try from observed surfaces in the same house. If no match, pick from the training set, relaxing room type match if required.")
    conf.add_args(parser)
    parser.add_argument("output_path",
                        help="Output path to save the predicted crops."
                             "Usually './data/processed/baselines/direct_crop/all_surfaces/[split]/drop_[drop_fraction]'.")
    parser.add_argument("observed_surface_textures_path",
                        help="Path to directory containing predictions for observed surfaces"
                             "Usually './data/processed/baselines/direct_crop/observed/[split]/drop_[drop_fraction]/texture_crops'.")
    parser.add_argument("split", help="val/test")
    parser.add_argument("--multiprop", default="0", help="Multiprop count")

    args = parser.parse_args()
    conf.process_args(args, output_is_dir=True)

    observed_surface_textures_path = args.observed_surface_textures_path
    output_path = args.output_path
    split = args.split
    multiprop = int(args.multiprop)

    if not osp.exists(osp.join(output_path, "texture_crops")):
        os.mkdir(osp.join(output_path, "texture_crops"))

    # Load crops assigned to train houses
    logging.info("Computing statistics on train set...")

    # Statistics from train set
    train_rs_crops = collect_train_set_crops(conf)  # Mapping room_type -> surface_type -> textures
    train_s_crops = {"floor": [], "wall": [], "ceiling": []}  # Mapping surface_type -> textures
    for _, surfaces in train_rs_crops.items():
        for surface in surfaces:
            train_s_crops[surface].extend(surfaces[surface])

    # Load houses
    houses = load_houses_with_textures(conf, data_split=split, drop_fraction=conf.drop_fraction, textures_path=observed_surface_textures_path)

    key = "prop"  # Key used to denote predictions

    # Predict textures for unobserved surfaces of each house
    for i, (house_key, house) in enumerate(houses.items()):
        logging.info("[%d/%d] Processing %s" % (i, len(houses), house_key))

        # Compute statistics on already assigned textures to observed surfaces.
        house_rs_crops = collect_house_crops(conf, house, house_rs_crops=None)

        for room_index, room in house.rooms.items():
            assert isinstance(room, Room)
            room_type = frozenset(room.types)

            for prop_index in [x for x in range(0, multiprop)] + [-1]:  # -1 is the base case. It must be added to the end to ensure key is populated last.
                prop_key = key
                if prop_index >= 0:
                    prop_key = f"{key}_{prop_index}"

                for surface in conf.surfaces:
                    if key not in room.surface_textures[surface]:
                        # Surface needs a texture
                        room.surface_textures[surface][prop_key] = pick_texture(train_rs_crops, train_s_crops, house_rs_crops, room_type, surface)
                    else:
                        room.surface_textures[surface][prop_key] = room.surface_textures[surface][key]  # Multiprop

        save_house_crops(house, save_path=osp.join(output_path, "texture_crops", house_key))
