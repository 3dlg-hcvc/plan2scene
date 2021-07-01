from plan2scene.common.house_parser import parse_houses, load_house_crops, save_house_crops
from plan2scene.config_manager import ConfigManager
import os.path as osp
import os
import logging

from plan2scene.utils.tile_util import seam_correct_house

if __name__ == "__main__":
    """
    Correct seams of textures of a house so the textures can be tiled in a seamless manner.
    """

    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(
        description="Correct seams of textures of a house so the textures can be tiled in a seamless manner.")
    conf.add_args(parser)
    parser.add_argument("output_path", help="Output path to save seam corrected textures."
                                            "Usually './data/processed/vgg_crop_select/[split]/drop_[drop_fraction]/tileable_texture_crops'.")
    parser.add_argument("texture_crops_path", help="Path to saved textures."
                                                   "Usually './data/processed/vgg_crop_select/[split]/drop_[drop_fraction]/texture_crops'.")
    parser.add_argument("split", help="train/val/test")

    args = parser.parse_args()
    conf.process_args(args)

    output_path = args.output_path
    texture_crops_path = args.texture_crops_path
    split = args.split
    embark_texture_synthesis_path = conf.seam_correct_config.texture_synthesis_path
    seam_mask_path = conf.seam_correct_config.seam_mask_path

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
        seam_correct_house(house, embark_texture_synthesis_path, seam_mask_path)
        save_house_crops(house, osp.join(output_path, house_key))
