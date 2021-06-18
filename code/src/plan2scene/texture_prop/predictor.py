from torch_geometric.data import Batch

from config_parser import Config
from plan2scene.config_manager import ConfigManager
from plan2scene.texture_prop.utils import get_network
import torch
import logging


class TexturePropPredictor:
    """
    Predicts texture embeddings for surfaces using the texture propagation network.
    """

    def __init__(self, conf: ConfigManager, system_conf: Config):
        """
        Initialize.
        :param conf: Config manager.
        :param system_conf: Texture propagation network configuration.
        """
        self.conf = conf
        self.net = get_network(conf, system_conf.network_arch).to(system_conf.device)

    def load_checkpoint(self, checkpoint_path) -> None:
        """
        Loads a checkpoint.
        :param checkpoint_path: Path to checkpoint.
        """
        ckpt = torch.load(checkpoint_path)
        logging.info("Loaded: %s" % (checkpoint_path))
        logging.info(self.net.load_state_dict(ckpt["model_state_dict"]))

    def predict(self, batch: Batch, training: bool = False):
        """
        Make predictions for a batch.
        :param batch: Batch of graphs used as input.
        :param training: Specify true to set the network to train mode. Otherwise, the network will be in eval mode.
        :return: Node embedding predictions for the batch of input graphs.
        """
        if training:
            self.net.train()
        else:
            self.net.eval()

        return self.net(batch.to(self.conf.texture_prop.device))
