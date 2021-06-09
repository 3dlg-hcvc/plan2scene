import abc

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from config_parser import Config
from plan2scene.common.trainer.save_reason import SaveReason
from plan2scene.config_manager import ConfigManager
import logging


class AbstractTrainer(abc.ABC):
    """
    Abstract trainer for substance classifier
    """

    def __init__(self, conf: ConfigManager, system_conf: Config, output_path: str, summary_writer: SummaryWriter, save_model_interval: int):
        """
        Initialize abstract trainer
        :param conf: Config Manager
        :param system_conf: Configuration of substance classifier.
        :param output_path: Path to save train outputs.
        :param summary_writer: Summary writer.
        :param save_model_interval: Model save interval.
        """
        self._conf = conf
        self._system_conf = system_conf
        self._summary_writer = summary_writer
        self._save_model_interval = save_model_interval
        self._output_path = output_path

        self._train_dataset = None
        self._val_dataset = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._max_epoch = None

        self._net = None
        self._crit = None
        self._optim = None
        self._device = system_conf.device
        self.epoch_stats = {}

    @property
    def max_epoch(self) -> int:
        return self._max_epoch

    @property
    def device(self) -> str:
        return self._device

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    @property
    def val_dataset(self) -> Dataset:
        return self._val_dataset

    @property
    def val_dataloader(self) -> DataLoader:
        return self._val_dataloader

    def setup(self) -> None:
        """
        Setup trainer.
        """
        # Setup dataset
        self._setup_dataset()
        logging.info("Train Data: %d" % len(self.train_dataset))
        logging.info("Val Data: %d" % len(self.val_dataset))

        self._net = self._setup_network().to(self.device)
        logging.info("Network: %s" % str(self.net))

        self._crit = self._setup_crit()
        logging.info("Criterion: %s" % str(self.crit))

        self._optim = self._setup_optim()
        logging.info("Optimizer: %s" % str(self.optim))

        self._max_epoch = self.system_conf.train.max_epoch
        logging.info("Max Epoch: %s" % self._max_epoch)

    @property
    def net(self):
        return self._net

    @abc.abstractmethod
    def _setup_optim(self):
        pass

    @property
    def optim(self):
        return self._optim

    @property
    def crit(self):
        return self._crit

    @abc.abstractmethod
    def _setup_crit(self):
        pass

    @abc.abstractmethod
    def _setup_network(self):
        pass

    @abc.abstractmethod
    def _setup_dataset(self):
        pass

    @property
    def output_path(self):
        return self._output_path

    @property
    def conf(self):
        return self._conf

    @property
    def system_conf(self):
        return self._system_conf

    @property
    def summary_writer(self):
        return self._summary_writer

    @abc.abstractmethod
    def _train_epoch(self, epoch: int):
        pass

    @abc.abstractmethod
    def _eval_epoch(self, epoch: int):
        pass

    @abc.abstractmethod
    def _report(self, epoch: int, train_stats, val_stats):
        pass

    @abc.abstractmethod
    def _save_checkpoint(self, epoch: int, save_reason: SaveReason, train_stats, val_stats):
        pass

    def train(self) -> None:
        """
        Start training.
        """
        for epoch in range(1, self.max_epoch + 1):
            self.net.train()
            train_stats = self._train_epoch(epoch)

            self.net.eval()
            val_stats = self._eval_epoch(epoch)

            self.epoch_stats[epoch] = {
                "epoch": epoch,
                "train_stats": train_stats,
                "val_stats": val_stats
            }

            self._report(epoch, train_stats, val_stats)

            if epoch % self._save_model_interval == 0:
                self._save_checkpoint(epoch, SaveReason.INTERVAL, train_stats, val_stats)
