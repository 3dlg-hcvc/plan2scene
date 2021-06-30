#!/usr/bin/python3


from plan2scene.common.house_parser import load_house_texture_embeddings, load_house_crops, parse_houses, save_arch
from plan2scene.config_manager import ConfigManager
import os
import os.path as osp
import logging

if __name__ == "__main__":
    """
    Embed textures to arch.json files/scene.json files.
    """

    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(
        description="Generates texture embedded arch.json files.")
    conf.add_args(parser)
    parser.add_argument("output_path", help="Output path to save texture embedded arch.json files. "
                                            "Usually './data/processed/vgg_crop_select/[split]/drop_[drop_fraction]/archs'.")
    parser.add_argument("texture_crops_path", help="Path to saved texture crops. "
                                                   "Usually './data/processed/vgg_crop_select/[split]/drop_[drop_fraction]/tileable_texture_crops'.")
    parser.add_argument("--texture-internal-walls-only", action="store_true", default=False, help="Specify flag to ommit textures on external side of perimeter walls.")
    parser.add_argument("split", help="train/val/test")
    args = parser.parse_args()
    conf.process_args(args)

    output_path = args.output_path
    texture_crops_path = args.texture_crops_path
    texture_internal_walls_only = args.texture_internal_walls_only
    split = args.split

    if not osp.exists(output_path):
        os.mkdir(output_path)

    # Load houses
    house_keys = conf.get_data_list(split)
    houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=split,
                                                                                                  house_key="{house_key}"),
                          photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=split,
                                                                                             drop_fraction=conf.drop_fraction,
                                                                                             house_key="{house_key}"))

    # Load textures
    for i, (house_key, house) in enumerate(houses.items()):
        logging.info("[%d/%d] Processing %s" % (i, len(houses), house_key))
        load_house_crops(conf, house,
                         osp.join(texture_crops_path, house_key))
        save_arch(conf, house, osp.join(output_path, house_key), texture_both_sides_of_walls=not texture_internal_walls_only)
