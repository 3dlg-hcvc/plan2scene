import logging
import os.path as osp
import os

from plan2scene.common.house_parser import map_surface_crops_to_houses, parse_houses, save_house_crops, load_houses_with_textures
from plan2scene.common.image_description import ImageDescription, ImageSource
from plan2scene.common.residence import Room, House
from plan2scene.config_manager import ConfigManager
from plan2scene.evaluation.metrics import CorrespondingPixelL1
from plan2scene.utils.image_util import get_medoid_key
from plan2scene.utils.io import load_image, load_smt_dataset
import torchvision.transforms as tfs
import torch
import torch.nn as nn
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


def house_rs_statistics(conf: ConfigManager, house: House, textures: dict, rs_closest_picks: dict = None):
    """
    Compute RS statistics on a house.
    :param conf: Config manager
    :param house: House considered
    :param textures: Dictionary of textures
    :param rs_closest_picks: Optional. A nested dictionary {room_type -> surface_type -> list of textures} which we update.
    :return: Updated / new nested dictionary showing the mapping room_type -> surface_type -> list of textures.
    """
    if rs_closest_picks is None:
        rs_closest_picks = {}

    for room_index, room in house.rooms.items():
        isinstance(room, Room)
        room_type = frozenset(room.types)
        if room_type not in rs_closest_picks:
            rs_closest_picks[room_type] = {v: [] for v in conf.surfaces}

        for surface in conf.surfaces:
            if len(room.surface_textures[surface]) > 0:
                rs_closest_picks[room_type][surface].append(
                    find_closest_match_key(conf, room.surface_textures[surface][get_medoid_key(room.surface_textures[surface])], textures))

    return rs_closest_picks


def compute_rs_statistics(conf: ConfigManager, textures: dict):
    """
    Compute RS statistics on the train set.
    :param conf: Config Manager
    :param textures: Dictionary of textures.
    :return: Nested dictionary depicting the mapping room type -> surface type -> list of textures.
    """
    # Compute statistics on the train set
    train_houses = load_houses_with_crops(conf, "train", "0.0")

    train_rs_closest_picks = {}  # {k:{v: [] for v in conf.surfaces} for k in conf.room_types}
    for i, (house_key, house) in enumerate(train_houses.items()):
        logging.info("[%d/%d] Processing train set: %s" % (i, len(train_houses), house_key))
        house_rs_statistics(conf, house, textures, train_rs_closest_picks)

    return train_rs_closest_picks


def find_closest_match_key(conf: ConfigManager, reference_crop: ImageDescription, textures: dict) -> str:
    """
    Find the most similar texture to the reference crop using pixel l1 loss.
    :param conf: Config Manager
    :param reference_crop:  Reference image
    :param textures: Dictionary of textures
    :return: Key of closest matching texture
    """
    with torch.no_grad():
        metric = CorrespondingPixelL1()
        min_loss = float("inf")
        min_texture_key = None

        for texture_key, texture in textures.items():
            loss = metric(reference_crop.image, texture)

            if loss <= min_loss:
                min_loss = loss
                min_texture_key = texture_key

        assert min_texture_key is not None
        return min_texture_key


def pick(items: list):
    """
    Pick the most frequent item in a list.
    :param items: List of items.
    :return: Most frequent item.
    """
    # Pick most frequent
    elements, counts = np.unique(items, return_counts=True)
    most_freq = elements[np.argmax(counts)]
    return most_freq


def pick_texture(train_rs_closest_picks: dict, train_s_closest_picks: dict, house_closest_picks: dict, room_type: frozenset, surface_type: str,
                 textures: dict) -> ImageDescription:
    """
    Pick a suitable texture for an unobserved surface.
    :param train_rs_closest_picks: RS statistics computed on the train set.
    :param train_s_closest_picks: S statistics computed on the train set.
    :param house_closest_picks: RS statistics computed on the observed surfaces of the house considered.
    :param room_type: Room type of the room containing the considered surface,
    :param surface_type: Surface type of the surface considered.
    :param textures: Textures dataset.
    :return: Predicted texture for the considered surface.
    """
    # First, try to select a same RS surface from the same house
    if room_type in house_closest_picks and len(house_closest_picks[room_type][surface_type]) > 0:
        return ImageDescription(textures[pick(house_closest_picks[room_type][surface_type])], ImageSource.RETRIEVE_UNOBSERVED)

    # If no match, try to pick from a same RS surface from the training set
    if room_type in train_rs_closest_picks and len(train_rs_closest_picks[room_type][surface_type]) > 0:
        return ImageDescription(textures[pick(train_rs_closest_picks[room_type][surface_type])], ImageSource.RETRIEVE_UNOBSERVED)

    # If no match, relax room type matching, pick a same S surface from the training set
    if room_type not in train_rs_closest_picks or len(train_rs_closest_picks[room_type][surface_type]) == 0:
        # Search in entire pool of surface typed textures if no match can be done on RS.
        return ImageDescription(textures[pick(train_s_closest_picks[surface_type])], ImageSource.RETRIEVE_UNOBSERVED)

    assert False


if __name__ == "__main__":
    """
    Predict textures for unobserved surfaces using the retrieve approach.
    """

    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Predict textures for unobserved surfaces using the retrieve approach.")
    conf.add_args(parser)
    parser.add_argument("output_path",
                        help="Output path to save the predicted houses with textures."
                             "Usually './data/processed/baselines/retrieve/observed/[split]/drop_[drop_fraction]'.")
    parser.add_argument("observed_surface_textures_path",
                        help="Path to directory containing predictions for observed surfaces"
                             "Usually './data/processed/baselines/retrieve/observed/[split]/drop_[drop_fraction]'.")
    parser.add_argument("split", help="val/test")
    parser.add_argument("smt_path", help="Path to substance mapped textures dataset")
    parser.add_argument("--multiprop", default="0", help="Multiprop count")
    parser.add_argument("--texture-size", default=128, type=int)

    args = parser.parse_args()
    conf.process_args(args, output_is_dir=True)

    observed_surface_textures_path = args.observed_surface_textures_path
    output_path = args.output_path
    split = args.split
    multiprop = int(args.multiprop)
    smt_path = args.smt_path
    key = "prop"
    texture_size = args.texture_size

    if not osp.exists(osp.join(output_path, "texture_crops")):
        os.mkdir(osp.join(output_path, "texture_crops"))

    # Load textures
    logging.info("Loading SMT dataset...")
    textures = load_smt_dataset(smt_path=smt_path, image_size=texture_size)
    logging.info("{count} textures loaded".format(count=len(textures)))

    # Compute statistics on train set
    train_rs_closest_picks = compute_rs_statistics(conf, textures)
    train_s_closest_picks = {"floor": [], "wall": [], "ceiling": []}
    for _, surfaces in train_rs_closest_picks.items():
        for surface in surfaces:
            train_s_closest_picks[surface].extend(surfaces[surface])

    # Load house keys
    house_keys = conf.get_data_list(split)

    # Load houses which will get populated with textures
    houses = load_houses_with_textures(conf, data_split=split, drop_fraction=conf.drop_fraction, textures_path=observed_surface_textures_path)

    # Predict textures for unobserved surfaces
    for i, (house_key, house) in enumerate(houses.items()):
        logging.info("[%d/%d] Processing %s" % (i, len(houses), house_key))
        assert isinstance(house, House)
        house_closest_picks = house_rs_statistics(conf, house, textures)
        for room_index, room in house.rooms.items():
            assert isinstance(room, Room)

            room_type = frozenset(room.types)

            # Multiprop for FID calculation
            for prop_index in [x for x in range(0, multiprop)] + [-1]:  # -1 is the base case. It must be added to the end to ensure key is populated last.
                prop_key = key
                if prop_index >= 0:
                    prop_key = f"{key}_{prop_index}"

                for surface in conf.surfaces:
                    if key not in room.surface_textures[surface]:
                        # Surface is unobserved and needs prediction
                        room.surface_textures[surface][prop_key] = pick_texture(train_rs_closest_picks, train_s_closest_picks, house_closest_picks, room_type,
                                                                                surface,
                                                                                textures)

        save_house_crops(house, save_path=osp.join(output_path, "texture_crops", house_key))
