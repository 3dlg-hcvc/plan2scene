import torch

from plan2scene.common.image_description import ImageSource
from plan2scene.common.residence import Room, House
from plan2scene.config_manager import ConfigManager
import os
import os.path as osp
import logging

from plan2scene.common.house_parser import load_house_texture_embeddings, parse_houses, save_house_crops, \
    save_house_texture_embeddings
from plan2scene.crop_select.util import fill_textures, vgg_crop_select
from plan2scene.texture_gen.predictor import TextureGenPredictor
from plan2scene.texture_gen.utils.io import load_conf_eval


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
        vgg_crop_select(conf, house, predictor=predictor)

        save_house_crops(house, osp.join(output_path, "texture_crops", house_key))
        save_house_texture_embeddings(house, osp.join(output_path, "surface_texture_embeddings", house_key + ".json"))
