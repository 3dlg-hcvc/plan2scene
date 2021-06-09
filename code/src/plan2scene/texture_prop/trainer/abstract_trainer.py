from torch.utils.tensorboard import SummaryWriter

from config_parser import Config
from plan2scene.common.trainer.epoch_summary import EpochSummary
from plan2scene.common.trainer.save_reason import SaveReason
from plan2scene.config_manager import ConfigManager
import abc
import logging
import os.path as osp
import os


class AbstractTrainer(abc.ABC):
    """
    Abstract trainer for texture propagation network
    """
    def __init__(self, conf: ConfigManager, system_conf: Config, output_path: str, summary_writer: SummaryWriter, deep_eval_interval: int, save_model_interval: int):
        """
        Initialize trainer.
        :param conf: Config Manager.
        :param system_conf: Propagation network configuration.
        :param output_path: Path to save training results.
        :param summary_writer: Summary writer used for logging.
        :param deep_eval_interval: Epoch interval between detail evaluations.
        :param save_model_interval: Epoch interval between model checkpoint saves.
        """
        self._conf = conf
        self._system_conf = system_conf
        self._output_path = output_path
        self._summary_writer = summary_writer
        self._save_model_interval = save_model_interval
        self._deep_eval_interval = deep_eval_interval

        self._train_dataset = None
        self._train_dataloader = None
        self._val_dataset = None
        self._val_dataloader = None

        self._net = None
        self._crit = None
        self._optim = None
        self._max_epoch = None
        self._metrics = None
        self._epoch_stats = {}

    @property
    def num_workers(self):
        return self._conf.num_workers

    @abc.abstractmethod
    def _setup_datasets(self):
        pass

    def setup(self) -> None:
        """
        Setup trainer for training.
        """
        self.conf.setup_seed(self.system_conf.train.seed)

        self._setup_datasets()
        logging.info("Train Data: %d" % (len(self.train_dataset)))
        logging.info("Val Data: %d" % (len(self.val_dataset)))

        self._net = self._setup_network().to(self.device)
        logging.info("Network: %s" % str(self.net))

        self._crit = self._setup_crit()
        logging.info("Criterion: %s" % str(self.crit))

        self._optim = self._setup_optim()
        logging.info("Optimizer: %s" % str(self.optim))

        self._max_epoch = self.system_conf.train.max_epoch

        self._metrics = self._setup_metrics()
        logging.info("Metrics: %s", str(self.metrics))

        self._setup_extra()

        if not osp.exists(self.output_path):
            os.mkdir(self.output_path)

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

            deep_eval_results = None
            if epoch % self._deep_eval_interval == 0:
                deep_eval_results = self._deep_eval_epoch(epoch)

            self._report(epoch, train_stats, val_stats, deep_eval_results)

            if epoch % self._save_model_interval == 0:
                self._save_checkpoint(epoch, SaveReason.INTERVAL, train_stats, val_stats)

    @abc.abstractmethod
    def _save_checkpoint(self, epoch: int, reason: SaveReason, train_stats, val_stats):
        pass

    @abc.abstractmethod
    def _deep_eval_epoch(self, epoch):
        pass

    @abc.abstractmethod
    def _report(self, epoch: int, train_stats: EpochSummary, val_stats: EpochSummary, deep_eval_results:list):
        pass

    @abc.abstractmethod
    def _setup_extra(self):
        pass

    @property
    def epoch_stats(self):
        return self._epoch_stats

    @abc.abstractmethod
    def _eval_epoch(self, epoch: int):
        pass

    @abc.abstractmethod
    def _train_epoch(self, epoch: int):
        pass

    @property
    def metrics(self):
        return self._metrics

    @abc.abstractmethod
    def _setup_metrics(self):
        pass

    @property
    def max_epoch(self):
        return self._max_epoch

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
    def _setup_network(self):
        pass

    @abc.abstractmethod
    def _setup_crit(self):
        pass

    @property
    def net(self):
        return self._net

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def val_dataloader(self):
        return self._val_dataloader

    @property
    def conf(self):
        return self._conf

    @property
    def system_conf(self):
        return self._system_conf

    @property
    def output_path(self):
        return self._output_path

    @property
    def device(self) -> str:
        """
        Device to train on (E.g. cuda, cpu)
        :return: Device as string
        """
        return self._system_conf.device

    @property
    def summary_writer(self):
        return self._summary_writer
