#!/bin/python3

import argparse
import shutil
import os
import os.path as osp
import sys
from torch.utils.tensorboard import SummaryWriter
import logging

from plan2scene.texture_gen.trainer.texture_gen_trainer import TextureGenTrainer
from plan2scene.texture_gen.utils.io import load_config_train

if __name__ == "__main__":
    """
    Train the texture synthesis stage of Plan2Scene.
    """

    parser = argparse.ArgumentParser(description="Train the texture synthesis stage of Plan2Scene.")
    parser.add_argument("output_path", type=str, nargs=1, help="Output path to save train logs and checkpoints.")
    parser.add_argument("texture_synth_config_path", type=str, nargs=1, help="Path to texture synthesis config file.")
    parser.add_argument("--save-model-interval", type=int, default=50, help="Epoch interval between periodic checkpoint saves.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Override path to textures dataset.")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    texture_synth_config_path = args.texture_synth_config_path[0]
    output_path = args.output_path[0]
    save_model_interval = args.save_model_interval

    if osp.exists(output_path):
        logging.error("Output path already exist")
        sys.exit(1)

    # Setup outputs
    summary_writter = SummaryWriter(osp.join(output_path, "tensorboard"))
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.StreamHandler(),
        logging.FileHandler(osp.join(output_path, "train.log"))])

    # Load train params
    train_params = load_config_train(config_path=texture_synth_config_path)
    # Overrides
    if dataset_path is not None:
        train_params.dataset.path = dataset_path
    logging.info("Output Path: %s" % output_path)
    shutil.copyfile(texture_synth_config_path, osp.join(output_path, "config.yml"))

    # Setup trainer
    trainer = TextureGenTrainer(train_params=train_params, output_path=output_path, summary_writer=summary_writter,
                                save_model_interval=save_model_interval)
    trainer.setup()
    trainer.train()
