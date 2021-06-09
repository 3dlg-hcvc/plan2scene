#!/bin/python3
import argparse
import logging
import sys

from torch.utils.tensorboard import SummaryWriter
import os.path as osp

from plan2scene.config_manager import ConfigManager
from plan2scene.texture_prop.trainer.texture_prop_trainer import TexturePropTrainer

if __name__ == "__main__":
    """
    Train the texture propagation GNN.
    """
    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Train the texture propagation GNN.")
    conf.add_args(parser)
    parser.add_argument("output_path", type=str, nargs=1, help="Output path to save checkpoints and train logs.")
    parser.add_argument("--save-model-interval", type=int, default=1, help="Epoch interval between periodically saved checkpoints.")
    parser.add_argument("--eval-interval", type=int, default=20, help="Epoch interval between slow detailed evaluations.")

    args = parser.parse_args()

    if osp.exists(args.output_path[0]):
        logging.error("Output path already exist")
        sys.exit(1)

    conf.process_args(args, output_is_dir=True)
    conf.copy_config_to_output()

    output_path = conf.output_path
    save_model_interval = args.save_model_interval
    eval_interval = args.eval_interval

    # Setup summary writer
    summary_writer = SummaryWriter(osp.join(output_path, "tensorboard_sp"))

    trainer = TexturePropTrainer(conf=conf, output_path=output_path, system_conf=conf.texture_prop, save_model_interval=save_model_interval,
                                 deep_eval_interval=eval_interval, summary_writer=summary_writer)
    trainer.setup()
    trainer.train()
