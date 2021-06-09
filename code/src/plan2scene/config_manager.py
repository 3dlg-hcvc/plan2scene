import json
import argparse
import logging
import os
import os.path as osp
import torch
import numpy as np
import random
import shutil

from config_parser import parse_config

"""
Config manager makes various configurations available to all modules.
"""


class ConfigManager:
    def __init__(self):
        """
        Initialize config manager.
        """
        self.args = None  # Contains command line arguments passed in.
        self.surfaces = None  # List of surfaces of a room.
        self.room_types = None  # List of supported room types.
        self.data_paths = None  # Configuration of paths to useful data.
        self.house_gen = None  # Configuration used to parse house architectures.
        self.texture_gen = None  # Configuration used to synthesize textures for observed surfaces.
        self.texture_prop = None  # Configuration used to propagate textures for unobserved surfaces.
        self.metrics = None  # Configuration of different metrics used.
        self.num_workers = None  # Number of workers used by data loders.
        self.output_path = None  # Output path to store results of the script.
        self.seed = None  # Random seed used.
        self.drop_fraction = None  # Fraction of surfaces synthetically unobserved.
        self.render_config = None  # Configuration used to render houses using scene toolkit.
        self.seam_correct_config = None  # Configuration used for seam correction of textures.

    def setup_seed(self, seed) -> None:
        """
        Update random seed.
        :param seed: New seed.
        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        logging.info("Using seed: %d" % seed)

    def load_default_args(self) -> None:
        """
        Load default arguments. Useful for loading plan2scene in a jupyter notebook.
        """
        parser = argparse.ArgumentParser()
        self.add_args(parser)
        args, _ = parser.parse_known_args()
        self.process_args(args)

    def copy_config_to_output(self) -> None:
        """
        Copy texture genration and texture propagation configuration to the output directory.
        """
        if not osp.exists(osp.join(self.output_path, "conf")):
            os.makedirs(osp.join(self.output_path, "conf"))
        shutil.copyfile(self.args.texture_prop, osp.join(self.output_path, "conf", "texture_prop.json"))
        shutil.copyfile(self.args.texture_gen, osp.join(self.output_path, "conf", "texture_gen.json"))

    def process_args(self, args, output_is_dir=False) -> None:
        """
        Process command line arguments.
        :param args: Command line arguments.
        :param output_is_dir: Specify true to create a directory at the output path. A log fill will be created automatically in this directory.
        """
        self.args = args
        if "output_path" in self.args.__dict__:
            if isinstance(self.args.output_path, str):
                self.output_path = self.args.output_path
            else:
                self.output_path = self.args.output_path[0]
            if output_is_dir:
                if not osp.exists(self.output_path):
                    os.makedirs(self.output_path)
                logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(osp.join(self.output_path, "log.out"))])
            else:
                logging.basicConfig(level=logging.getLevelName(args.log_level))
        else:
            logging.basicConfig(level=logging.getLevelName(args.log_level))

        self.setup_seed(int(args.seed))
        self.num_workers = args.num_workers
        self.data_paths = parse_config(args.data_paths)
        self.house_gen = parse_config(args.house_gen)
        self.texture_gen = parse_config(args.texture_gen)
        self.texture_prop = parse_config(args.texture_prop)
        self.metrics = parse_config(args.metric_conf)
        self.render_config = parse_config(args.render_config)
        self.seam_correct_config = parse_config(args.seam_correct_config)

        self.surfaces = parse_config(osp.join(args.labels_path, "surfaces.json"))
        self.room_types = parse_config(osp.join(args.labels_path, "room_types.json"))

        self.texture_prop.node_embedding_dim = len(self.room_types) + 3 * (self.texture_gen.combined_emb_dim + 1)
        self.texture_prop.node_target_dim = 3 * (self.texture_gen.combined_emb_dim)

        self.drop_fraction = args.drop
        logging.info("Args: %s" % str(self.args))

    def get_data_list(self, split: str) -> list:
        """
        Return a train/val/test data list.
        :param split: Data split.
        :return: Data list.
        """
        results = []
        with open(self.data_paths.data_list_path_spec.format(split=split)) as f:
            for line in f:
                results.append(line.strip())
        return results

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """
        Add optional arguments to an argument parser.
        :param parser: Argument parser.
        """
        parser.add_argument("--seed", help="Default seed value to use.", default=12415, type=int)
        parser.add_argument("--data-paths", help="Path to data_paths.json file",
                            default="./conf/plan2scene/data_paths.json")
        parser.add_argument("--house-gen", help="Path to house_gen.json file",
                            default="./conf/plan2scene/house_gen.json")
        parser.add_argument("--metric-conf", help="Path to metric.json.", default="./conf/plan2scene/metric.json")
        parser.add_argument("--texture-gen", help="Path to texture_gen.json file",
                            default="./conf/plan2scene/texture_gen.json")
        parser.add_argument("--texture-prop", help="Path to texture_prop.json file",
                            default="./conf/plan2scene/texture_prop_conf/default.json")
        parser.add_argument("--render-config", help="Path to ./conf/plan2scene/render.json file",
                            default="./conf/plan2scene/render.json")
        parser.add_argument("--seam-correct-config", help="Path to ./conf/plan2scene/seam_correct.json", default="./conf/plan2scene/seam_correct.json")

        parser.add_argument("--labels-path", help="Path to directory which contains surfaces.json and room_types.json",
                            default="./conf/plan2scene/labels")
        parser.add_argument("-l", "--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                            default="INFO",
                            help="Set the log level")

        parser.add_argument("--drop", help="Drop fraction used", default="0.0")
        parser.add_argument("--num-workers", default=4, type=int, help="Number of workers used by a data loader.")
