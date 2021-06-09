#!/bin/python3
import logging

from torch.utils.tensorboard import SummaryWriter

from plan2scene.config_manager import ConfigManager
from config_parser import parse_config
import os.path as osp
import os
import argparse
import shutil

from plan2scene.evaluation.metric_impl.substance_classifier.trainer.substance_classifier_trainer import SubstanceClassifierTrainer

if __name__ == "__main__":
    """
    Train the substance classifier used by the SUBS metric.
    """
    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Train the Substance Classifier used by the SUBS metric.")
    conf.add_args(parser)
    parser.add_argument("output_path", type=str, nargs=1, help="Output path to save train progress.")
    parser.add_argument("conf_path", type=str, nargs=1, help="Path to train configuration.")
    parser.add_argument("--save-model-interval", type=int, default=10, help="Epoch interval between periodic saving of checkpoints.")
    parser.add_argument("--preview-results", default=False, action="store_true",
                        help="Specify true to preview correct classifications and miss classifications.")
    parser.add_argument("--textures-dataset-path", type=str, default=None, help="Override the textures dataset path specified in train config file.")
    parser.add_argument("--os-dataset-path", type=str, default=None, help="Override the OpenSurfaces dataset path specified in train config file.")

    args = parser.parse_args()
    conf.process_args(args, output_is_dir=True)

    preview_results = args.preview_results
    save_model_interval = args.save_model_interval
    output_path = conf.output_path
    conf_path = args.conf_path[0]

    if osp.exists(output_path):
        logging.warning("Output path already exists!")

    system_conf = parse_config(conf_path)
    if not osp.exists(osp.join(output_path, "conf")):
        os.mkdir(osp.join(output_path, "conf"))
    shutil.copyfile(conf_path, osp.join(output_path, "conf", "substance_classifier_conf.json"))

    if args.textures_dataset_path is not None:  # Override path to textures dataset
        system_conf.datasets.textures = args.textures_dataset_path

    if args.os_dataset_path is not None:  # Override path to opensurfaces dataset
        system_conf.datasets.os = args.os_dataset_path

    summary_writter = SummaryWriter(osp.join(output_path, "tensorboard"))

    trainer = SubstanceClassifierTrainer(conf=conf, system_conf=system_conf, output_path=output_path, summary_writer=summary_writter,
                                         save_model_interval=save_model_interval, preview_results=preview_results)
    trainer.setup()
    trainer.train()
