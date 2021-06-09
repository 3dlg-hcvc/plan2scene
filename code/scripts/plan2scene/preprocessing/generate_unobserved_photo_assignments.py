#!/usr/bin/python3
import logging

from plan2scene.config_manager import ConfigManager
import pandas as pd
import os.path as osp
import os
import json

if __name__ == "__main__":
    """
    Generate photoroom.csv files by simulating photo un-observations.
    """
    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Generate photoroom.csv files by simulating photo un-observations.")
    conf.add_args(parser)
    parser.add_argument("output_path", help="Path to generate simulated un-observed photoroom.csv files. "
                                            "Usually ./data/processed/photo_assignments/[split]/")
    parser.add_argument("input_path", help="Path to directory containing photoroom.csv files. "
                                           "Usually ./data/input/photo_assignments/[split]/drop_0.0")
    parser.add_argument("unobserved_list_path",
                        help="Path to unobserved_photos.json. Usually './conf/plan2scene/unobserved_photos.json'")
    parser.add_argument("split", help="train/val/test")
    args = parser.parse_args()
    conf.process_args(args, output_is_dir=True)

    output_path = args.output_path
    input_path = args.input_path
    unobserved_list_path = args.unobserved_list_path
    split = args.split

    # Work

    with open(unobserved_list_path) as f:
        frac_unobserved_photos_list = json.load(f)

    for frac, unobserved_photos_list in frac_unobserved_photos_list.items():
        unobserved_photos = set(unobserved_photos_list)
        logging.info("Processing unobserved fraction %s" % frac)
        data_list = conf.get_data_list(split)

        if not osp.exists(osp.join(output_path, "drop_%s" % frac)):
            os.mkdir(osp.join(output_path, "drop_%s" % frac))

        for i, house_key in enumerate(data_list):
            logging.info("[%d/%d] Processing %s" % (i, len(data_list), house_key))
            input_df = pd.read_csv(osp.join(input_path, house_key + ".photoroom.csv"))
            remain_df = input_df[~(input_df.photo.isin(unobserved_photos))]
            remain_df.to_csv(osp.join(output_path, "drop_%s" % frac, house_key + ".photoroom.csv"), index=False)