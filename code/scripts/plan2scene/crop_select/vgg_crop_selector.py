import torch

from plan2scene.common.image_description import ImageSource
from plan2scene.common.residence import Room, House
from plan2scene.config_manager import ConfigManager
import os
import os.path as osp
import logging

from plan2scene.common.house_parser import load_house_texture_embeddings, parse_houses, save_house_crops, \
    save_house_texture_embeddings
from plan2scene.crop_select.util import fill_textures
from plan2scene.texture_gen.predictor import TextureGenPredictor
from plan2scene.texture_gen.utils.io import load_conf_eval


def get_least_key(kv):
    """
    Given a dictionary, returns the key with minimum value.
    :param kv: Dictionary considered.
    :return: Key with the minimum value.
    """
    min_k = None
    min_v = None
    for k, v in kv.items():
        if min_v is None or v.item() < min_v:
            min_k = k
            min_v = v.item()

    return min_k


def process(conf: ConfigManager, house: House, predictor: TextureGenPredictor) -> None:
    """
    Assigns the least VGG loss crop for each surface of the house.
    :param conf: ConfigManager
    :param house: House to update
    :param predictor: Predictor used to synthesize textures
    """
    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        # Calculate the least VGG loss embeddings
        for surface in room.surface_embeddings:
            least_key = get_least_key(room.surface_losses[surface])
            if least_key is not None:
                room.surface_embeddings[surface] = {"prop": room.surface_embeddings[surface][least_key]}
                room.surface_losses[surface] = {"prop": room.surface_losses[surface][least_key]}
            else:
                room.surface_embeddings[surface] = {}
                room.surface_losses[surface] = {}

    fill_textures(conf, {house.house_key: house}, predictor=predictor, log=False, image_source=ImageSource.VGG_CROP_SELECT, skip_existing_textures=False)


if __name__ == "__main__":
    """Selects least VGG loss embeddings for each room surface, from multiple texture embeddings assigned to the surface."""
    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(
        description="Selects least VGG loss embeddings for each room surface, from multiple texture embeddings assigned to the surface.")
    conf.add_args(parser)
    parser.add_argument("output_path", help="Output path to save selected embeddings and synthesized texture crops. "
                                            "Usually './data/processed/vgg_crop_select/[split]/drop_[drop_fraction]'.")
    parser.add_argument("texture_gen_path", help="Path to saved texture_gen embeddings and crops. "
                                                 "Usually './data/processed/texture_gen/[split]/drop_[drop_fraction]'.")
    parser.add_argument("split", help="train/val/test")
    args = parser.parse_args()
    conf.process_args(args)

    output_path = args.output_path
    texture_gen_path = args.texture_gen_path
    split = args.split

    if not osp.exists(output_path):
        os.makedirs(output_path)

    if not osp.exists(osp.join(output_path, "texture_crops")):
        os.mkdir(osp.join(output_path, "texture_crops"))

    if not osp.exists(osp.join(output_path, "surface_texture_embeddings")):
        os.mkdir(osp.join(output_path, "surface_texture_embeddings"))

    # Load checkpoint
    predictor = TextureGenPredictor(
        conf=load_conf_eval(config_path=conf.texture_gen.texture_synth_conf),
        rgb_median_emb=conf.texture_gen.rgb_median_emb)
    predictor.load_checkpoint(checkpoint_path=conf.texture_gen.checkpoint_path)

    # Load houses
    house_keys = conf.get_data_list(split)
    houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=split,
                                                                                                  house_key="{house_key}"),
                          photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=split,
                                                                                             drop_fraction=conf.drop_fraction,
                                                                                             house_key="{house_key}"))

    # Load texture embeddings
    for i, (house_key, house) in enumerate(houses.items()):
        logging.info("[%d/%d] Processing %s" % (i, len(houses), house_key))
        load_house_texture_embeddings(house,
                                      osp.join(texture_gen_path, "surface_texture_embeddings", house_key + ".json"))
        process(conf, house, predictor=predictor)

        save_house_crops(house, osp.join(output_path, "texture_crops", house_key))
        save_house_texture_embeddings(house, osp.join(output_path, "surface_texture_embeddings", house_key + ".json"))
