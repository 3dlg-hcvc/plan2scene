from torch.utils.tensorboard import SummaryWriter

from plan2scene.common.trainer.epoch_summary import EpochSummary
import logging
import abc

from plan2scene.common.trainer.save_reason import SaveReason


class AbstractTrainer(abc.ABC):
    """
    Abstract trainer for modified neural texture synthesis approach.
    """

    def __init__(self, train_params: dict, output_path: str, summary_writer: SummaryWriter, save_model_interval: int):
        """
        Initializes trainer.
        :param train_params: Configuration of the model.
        :param output_path: Directory to save outputs.
        :param summary_writer: Summary writer.
        :param save_model_interval: Checkpoint save interval.
        """
        self._train_params = train_params
        self._output_path = output_path
        self._save_model_interval = save_model_interval
        self._summary_writer = summary_writer

        self._predictor = None
        self._train_dataset = None
        self._train_dataloader = None
        self._val_dataset = None
        self._val_dataloader = None
        self._optimizer = None
        self._device = None
        self._epoch_stats = {}
        self._max_epoch = 0

        self._best_checkpoint_metric_value = None

        pass

    @abc.abstractmethod
    def _setup_predictor(self):
        pass

    @abc.abstractmethod
    def _setup_datasets(self):
        pass

    @property
    def epoch_stats(self):
        return self._epoch_stats

    def setup(self) -> None:
        """
        Setup the trainer.
        """
        self._device = self.train_params.device

        self._setup_predictor()
        logging.info("Predictor: %s" % self._predictor)
        self._setup_datasets()
        logging.info("Train Data: %d" % (len(self.train_dataset)))
        logging.info("Val Data: %d" % (len(self.val_dataset)))

        self._setup_loss_function()
        self._setup_optimizer()
        self._max_epoch = self.train_params.train.epochs

    @property
    def device(self):
        return self._device

    @property
    def max_epoch(self):
        return self._max_epoch

    def train(self) -> None:
        """
        Start training
        """
        for epoch in range(1, self.max_epoch + 1):
            train_stats = self._train_epoch(epoch)
            val_stats = self._eval_epoch(epoch)

            self.epoch_stats[epoch] = {
                "epoch": epoch,
                "train_stats": train_stats,
                "val_stats": val_stats
            }

            self._report(epoch, train_stats, val_stats)

            checkpoint_metric_value = self._get_checkpoint_metric_value(val_stats)
            if self._best_checkpoint_metric_value is None or checkpoint_metric_value < self._best_checkpoint_metric_value:
                self._best_checkpoint_metric_value = checkpoint_metric_value
                self._save_checkpoint(epoch, SaveReason.BEST_MODEL, train_stats, val_stats)

            if epoch % self.save_model_interval == 0:
                self._save_checkpoint(epoch, SaveReason.INTERVAL, train_stats, val_stats)

    @abc.abstractmethod
    def _get_checkpoint_metric_value(self, eval_stats: EpochSummary) -> float:
        """
        Returns the overall goodness value given the evaluation results on the validation set.
        This value is used to identify the best checkpoints for the purpose of saving them.
        :param eval_stats: Evaluation stats on the validation set for the current epoch.
        :return: Overall goodness value considering eval_stats
        """
        pass

    @abc.abstractmethod
    def _save_checkpoint(self, epoch: int, reason: SaveReason, train_stats, val_stats):
        pass

    @abc.abstractmethod
    def _report(self, epoch: int, train_stats: EpochSummary, val_stats: EpochSummary) -> None:
        """
        Write evaluation results to disk/tensorboard.
        :param epoch: Epoch
        :param train_stats: Train set evaluation
        :param val_stats: Validation set evaluation
        """
        pass

    @abc.abstractmethod
    def _eval_epoch(self, epoch) -> EpochSummary:
        pass

    @abc.abstractmethod
    def _train_epoch(self, epoch) -> EpochSummary:
        pass

    @abc.abstractmethod
    def _setup_optimizer(self):
        pass

    @abc.abstractmethod
    def _setup_loss_function(self):
        pass

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def predictor(self):
        return self._predictor

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
    def train_params(self):
        return self._train_params

    @property
    def output_path(self):
        return self._output_path

    @property
    def save_model_interval(self):
        return self._save_model_interval

    @property
    def summary_writer(self):
        return self._summary_writer
