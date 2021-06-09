import logging
import os.path as osp
import os
from plan2scene.common.house_parser import map_surface_crops_to_houses, parse_houses, save_house_crops
from plan2scene.common.image_description import ImageDescription, ImageSource
from plan2scene.common.residence import Room, House
from plan2scene.config_manager import ConfigManager
from plan2scene.evaluation.metrics import CorrespondingPixelL1
from plan2scene.utils.image_util import get_medoid_key
from plan2scene.utils.io import load_image, load_smt_dataset
import torchvision.transforms as tfs
import torch
import torch.nn as nn


def find_closest_match(conf: ConfigManager, reference_crop: ImageDescription, textures: dict) -> ImageDescription:
    """
    Find the most similar texture to the reference crop using pixel l1 loss.
    :param conf: Config Manager
    :param reference_crop:  Reference image
    :param textures: Dictionary of textures
    :return: Selected texture
    """
    with torch.no_grad():
        metric = CorrespondingPixelL1()
        min_loss = float("inf")
        min_texture = None

        for texture_key, texture in textures.items():
            loss = metric(reference_crop.image, texture)

            if loss <= min_loss:
                min_loss = loss
                min_texture = texture

        assert min_texture is not None
        return ImageDescription(min_texture, ImageSource.RETRIEVE_OBSERVED)


if __name__ == "__main__":
    """
    Predict textures for observed surfaces using the retrieve approach.
    """

    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Predict textures for observed surfaces using the retrieve approach.")
    conf.add_args(parser)
    parser.add_argument("output_path",
                        help="Output path to save houses with predicted textures."
                             "Usually './data/processed/baselines/retrieve/observed/[split]/drop_[drop_fraction]'.")
    parser.add_argument("split", help="val/test")
    parser.add_argument("smt_path", help="Path to substance mapped textures dataset")
    parser.add_argument("--multiprop", default="0", help="Multiprop count")
    parser.add_argument("--texture-size", default=128, type=int)

    args = parser.parse_args()
    conf.process_args(args, output_is_dir=True)

    output_path = args.output_path
    split = args.split
    multiprop = int(args.multiprop)
    smt_path = args.smt_path
    key = "prop"
    smt_image_size = args.texture_size

    if not osp.exists(osp.join(output_path, "texture_crops")):
        os.mkdir(osp.join(output_path, "texture_crops"))

    # Load textures
    logging.info("Loading SMT dataset...")
    textures = load_smt_dataset(smt_path=smt_path, image_size=smt_image_size)
    logging.info("{count} textures loaded".format(count=len(textures)))

    # Load house keys
    house_keys = conf.get_data_list(split)

    # Load houses and rectified surface crops
    crop_houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=split,
                                                                                                       house_key="{house_key}"),
                               photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=split,
                                                                                                  drop_fraction=conf.drop_fraction,
                                                                                                  house_key="{house_key}"))
    map_surface_crops_to_houses(conf, crop_houses)

    # Load houses which will get populated with textures
    output_houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=split,
                                                                                                         house_key="{house_key}"),
                                 photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=split,
                                                                                                    drop_fraction=conf.drop_fraction,
                                                                                                    house_key="{house_key}"))

    # Predict textures for observed surfaces
    for i, (house_key, output_house) in enumerate(output_houses.items()):
        logging.info("[%d/%d] Processing %s" % (i, len(output_houses), house_key))
        crop_house = crop_houses[house_key]
        assert isinstance(output_house, House)
        assert isinstance(crop_house, House)

        for room_index, output_room in output_house.rooms.items():
            crop_room = crop_house.rooms[room_index]
            assert isinstance(output_room, Room)
            assert isinstance(crop_room, Room)

            # Multiprop for FID calculation
            for prop_index in [x for x in range(0, multiprop)] + [-1]:
                prop_key = key
                if prop_index >= 0:
                    prop_key = f"{key}_{prop_index}"

                for surface in conf.surfaces:
                    if len(crop_room.surface_textures[surface]) > 0:
                        # Make a prediction using crops assigned to the surface
                        medoid_crop = crop_room.surface_textures[surface][get_medoid_key(crop_room.surface_textures[surface])]
                        output_room.surface_textures[surface][prop_key] = find_closest_match(conf, medoid_crop, textures)

        save_house_crops(output_house, save_path=osp.join(output_path, "texture_crops", house_key))
