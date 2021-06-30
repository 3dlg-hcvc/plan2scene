import logging

from arch_parser.preferred_format import PreferredFormat
from plan2scene.common.image_description import ImageSource, ImageDescription
from plan2scene.common.residence import House, Room
from plan2scene.config_manager import ConfigManager
from plan2scene.utils.io import load_image
from arch_parser.parser import parse_house_json_file
import json
import os
import os.path as osp
from PIL import Image
import shutil
import torch
import numpy as np

"""
Utility methods to parse and serialize houses.
"""


def parse_houses(conf: ConfigManager, house_keys, house_path_spec: str, photoroom_csv_path_spec: str) -> dict:
    """
    Parse house arch_jsons into houses
    :param conf: Config Manager
    :param house_keys: Keys of houses to be parsed
    :param house_path_spec: Spec of path to a house
    :param photoroom_csv_path_spec: Spec of path to photoroom csv file
    :return: mapping from house_key to parsed house
    """
    results = {}
    for house_key in house_keys:
        arch_house = parse_house_json_file(house_path_spec.format(house_key=house_key),
                                           photoroom_csv_path_spec.format(house_key=house_key))
        results[house_key] = House.from_arch_house(arch_house, surfaces=conf.surfaces)

    return results


def map_surface_crops_to_house(conf: ConfigManager, house: House) -> None:
    """
    For each room of the given house, load the rectified crops belonging to assigned photos as textures.
    :param conf: Config Manager
    :param house: House processed
    """
    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        for photo in room.photos:
            for surface in conf.surfaces:
                surface_instances = [i for i in range(conf.texture_gen.masks_per_surface[surface])]
                for surface_instance in surface_instances:
                    for crop_instance in range(conf.texture_gen.crops_per_mask):
                        candidate_key = "%s_%d_crop%d" % (photo, surface_instance, crop_instance)
                        if osp.exists(osp.join(conf.data_paths.rectified_crops_path, surface, candidate_key + ".png")):
                            image = load_image(
                                osp.join(conf.data_paths.rectified_crops_path, surface, candidate_key + ".png"))
                            image = Image.fromarray(np.array(image))  # Drop any file system links
                            room.surface_textures[surface][candidate_key] = ImageDescription(image, ImageSource.DIRECT_CROP)


def map_surface_crops_to_houses(conf: ConfigManager, houses: dict) -> None:
    """
    For each room of every house, load the rectified crops belonging to assigned photos as textures.
    :param conf: Config Manager
    :param houses: House processed
    """
    for i, (house_key, house) in enumerate(houses.items()):
        map_surface_crops_to_house(conf, house)


def save_arch(conf: ConfigManager, house: House, arch_path: str, texture_both_sides_of_walls) -> None:
    """
    Saves a house as an arch.json file or a scene.json file.
    :param conf: Config Manager
    :param house: House to save
    :param arch_path: Save path. We determine save format based on extension specified here.
    :param texture_both_sides_of_walls: Both sides of all walls are textured, including walls with only one interior side. The interior side texture is copied to exterior side.
    :return:
    """
    save_format = PreferredFormat.NONE
    if arch_path.endswith(".arch.json"):
        save_format = PreferredFormat.ARCH_JSON
    elif arch_path.endswith(".scene.json"):
        save_format = PreferredFormat.SCENE_JSON
    else:
        save_format = house.preferred_format
        arch_path += save_format.extension

    if save_format == PreferredFormat.ARCH_JSON:
        house_json = house.to_arch_json(texture_both_sides_of_walls=texture_both_sides_of_walls)
    elif save_format == PreferredFormat.SCENE_JSON:
        house_json = house.to_scene_json(texture_both_sides_of_walls=texture_both_sides_of_walls)
    else:
        assert False, "Save format unspecified"

    with open(arch_path, "w") as f:
        json.dump(house_json, f, indent=4)


def save_house_crops(house: House, save_path: str) -> None:
    """
    Save textures/crops of a house to disk
    :param house: House to save
    :param save_path: Save path
    """
    assert not osp.exists(save_path), f"Already saved. Please delete {save_path}"

    if not osp.exists(save_path):
        os.mkdir(save_path)

    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        if not osp.exists(osp.join(save_path, room.room_id)):
            os.mkdir(osp.join(save_path, room.room_id))

        for surface, crop_map in room.surface_textures.items():
            if not osp.exists(osp.join(save_path, room.room_id, surface)):
                os.mkdir(osp.join(save_path, room.room_id, surface))
            for crop_name, crop_img_description in crop_map.items():
                assert isinstance(crop_img_description, ImageDescription)
                crop_img_description.save(osp.join(save_path, room.room_id, surface, crop_name + ".png"))


def save_house_texture_embeddings(house: House, save_path: str = None) -> dict:
    """
    Saves texture embeddings of a house to disk
    :param house: House to save.
    :param save_path: Optional. Path to save.
    :return: Content to save as a dictionary.
    """
    assert not osp.exists(save_path), f"Already saved. Please delete {save_path}"
    room_id_embeddings_map = {}
    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        room_id = room.room_id
        room_id_embeddings_map[room_id] = {
            "texture_embeddings": {surf: {emb_key: emb.tolist() for emb_key, emb in emb_map.items()} for surf, emb_map
                                   in room.surface_embeddings.items()},
            "texture_loss": {surf: {emb_key: emb_loss.item() for emb_key, emb_loss in emb_loss_map.items()} for
                             surf, emb_loss_map
                             in room.surface_losses.items()}
        }
    result = {
        "house_key": house.house_key,
        "rooms": room_id_embeddings_map
    }
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(result, f, indent=4)
    return result


def load_house_crops(conf: ConfigManager, house: House, save_path: str, exclude_prior_path: str = None, restrict_prior_path: str = None) -> None:
    """
    Load texture crops of a house from disk
    :param house: House to assign loaded texture crops
    :param save_path: Save location of texture crops
    :param exclude_prior_path: Specify prior prediction save path here to exclude surfaces that has a prior prediction.
    :param restrict_prior_path: Specify prior prediction save path here to only load surfaces that has a prior prediction.
    """
    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        # import pdb; pdb.set_trace()
        assert osp.exists(osp.join(save_path, room.room_id)), "Not exist %s" % osp.join(save_path, room.room_id)

        for surface in conf.surfaces:
            # import pdb; pdb.set_trace()
            if osp.exists(osp.join(save_path, room.room_id, surface)):
                if exclude_prior_path is not None:  # Exclude prior predictions
                    assert osp.exists(osp.join(exclude_prior_path, room.room_id, surface)), "Prior path doesn't have a matching room surface."
                    prior_files = os.listdir(osp.join(exclude_prior_path, room.room_id, surface))
                    prior_files = [a for a in prior_files if a.endswith(".png")]
                    assert len(prior_files) <= 1, "Multiple prior predictions."
                    if len(prior_files) == 1:
                        assert "prop" in prior_files[0], "Invalid prior prediction"

                    if len(prior_files) > 0:
                        # We have prior predictions. Therefore, exclude this surface.
                        continue

                if restrict_prior_path is not None:  # Restrict to prior predictions
                    assert osp.exists(osp.join(restrict_prior_path, room.room_id, surface)), "Prior path doesn't have a matching room surface."
                    prior_files = os.listdir(osp.join(restrict_prior_path, room.room_id, surface))
                    prior_files = [a for a in prior_files if a.endswith(".png")]
                    assert len(prior_files) <= 1, "Multiple prior predictions."
                    if len(prior_files) == 1:
                        assert "prop" in prior_files[0], "Invalid prior prediction"
                    if len(prior_files) == 0:
                        # We don't have prior predictions. Therefore, exclude this surface.
                        continue

                crop_files = os.listdir(osp.join(save_path, room.room_id, surface))
                crop_files = [a for a in crop_files if a.endswith(".png")]
                for crop_file in crop_files:
                    crop_name = os.path.splitext(crop_file)[0]
                    crop_image_description = ImageDescription.parse_image(osp.join(save_path, room.room_id, surface, crop_file))
                    room.surface_textures[surface][crop_name] = crop_image_description


def load_house_texture_embeddings(house: House, save_path: str) -> None:
    """
    Load texture embeddings (and losses) of a house from disk
    :param house: House to assignn loaded embeddings
    :param save_path: Save location of texture embeddings
    """
    with open(save_path) as f:
        embedding_json = json.load(f)

    assert embedding_json["house_key"] == house.house_key
    room_id_embeddings_map = embedding_json["rooms"]

    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        room_id = room.room_id
        room_json = room_id_embeddings_map[room_id]
        for surface in room_json["texture_embeddings"]:
            room.surface_embeddings[surface] = {k: torch.tensor(v, dtype=torch.float32) for k, v in
                                                room_json["texture_embeddings"][surface].items()}
        for surface in room_json["texture_loss"]:
            room.surface_losses[surface] = {k: torch.tensor(v, dtype=torch.float32) for k, v in
                                            room_json["texture_loss"][surface].items()}


def load_houses_with_textures(conf: ConfigManager, data_split: str, drop_fraction: str, textures_path: str) -> dict:
    """
    Load houses for the specified data split and drop setting. Then, assign textures to them.
    :param conf: Config Manager
    :param data_split: Train/val/test.
    :param drop_fraction: Photo unobserve setting.
    :param textures_path: Path containing saved textures.
    :return: Dictionary of houses
    """
    house_keys = conf.get_data_list(data_split)
    houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=data_split,
                                                                                                  house_key="{house_key}"),
                          photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=data_split,
                                                                                             drop_fraction=drop_fraction,
                                                                                             house_key="{house_key}"))
    for i, (house_key, house) in enumerate(houses.items()):
        load_house_crops(conf, house,
                         osp.join(textures_path, house_key))

    return houses


def load_houses_with_embeddings(conf: ConfigManager, data_split: str, drop_fraction: str, embeddings_path: str):
    """
    Load houses for the specified data split and drop setting. Then, assign embeddings to them.
    :param conf: Config Manager
    :param data_split: Train/val/test.
    :param drop_fraction: Photo unobserve setting.
    :param embeddings_path: Path containing saved embedding files.
    :return: Dictionary of houses
    """
    # Load houses
    house_keys = conf.get_data_list(data_split)
    houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=data_split,
                                                                                                  house_key="{house_key}"),
                          photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=data_split,
                                                                                             drop_fraction=drop_fraction,
                                                                                             house_key="{house_key}"))

    for i, (house_key, house) in enumerate(houses.items()):
        load_house_texture_embeddings(house,
                                      osp.join(embeddings_path, house_key + ".json"))

    return houses
