from plan2scene.common.trainer.epoch_summary import EpochSummary


class SubstanceClassifierEpochSummary(EpochSummary):
    """
    Epoch summary for substance classifier.
    """
    def __init__(self, epoch_loss: float, passed_count: int, epoch_batch_count: int, epoch_entry_count: int, mistakes: list, correct_predictions: list,
                 per_substance_correct_count_map: dict, per_substance_count_map: dict):
        """
        Initialize substance classifier epoch summary.
        :param epoch_loss: Epoch loss
        :param passed_count: Passed example count.
        :param epoch_batch_count: Batch count.
        :param epoch_entry_count: Total example count.
        :param mistakes: List of mistakes.
        :param correct_predictions: List of correct predictions.
        :param per_substance_correct_count_map: Dictionary mapping from substance to correct prediction count
        :param per_substance_count_map: Dictionary mapping from substance to total substance count.
        """
        super().__init__(epoch_loss=epoch_loss, epoch_batch_count=epoch_batch_count, epoch_entry_count=epoch_entry_count)
        self._mistakes = mistakes
        self._correct_predictions = correct_predictions
        self._passed_count = passed_count
        self._per_substance_correct_count_map = per_substance_correct_count_map
        self._per_substance_count_map = per_substance_count_map

    @property
    def per_substance_correct_count_map(self) -> dict:
        """
        Return dictionary mapping from substance to correct prediction count
        :return: Dictionary mapping from substance to correct prediction count
        """
        return self._per_substance_correct_count_map

    @property
    def per_substance_count_map(self) -> dict:
        """
        Returns a dictionary mapping from substance to total substance count.
        :return: Dictionary mapping from substance to total substance count.
        """
        return self._per_substance_count_map

    @property
    def correct_predictions(self) -> list:
        """
        Returns list of correct predictions.
        :return: List of correct predictions.
        """
        return self._correct_predictions

    @property
    def mistakes(self) -> list:
        """
        Returns list of mistakes.
        :return: List of mistakes.
        """
        return self._mistakes

    @property
    def passed_count(self) -> int:
        """
        Returns the count of examples that were correctly classified.
        :return: Correct prediction count.
        """
        return self._passed_count

    @passed_count.setter
    def passed_count(self, val) -> None:
        """
        Set the count of examples that were correctly classified.
        """
        assert isinstance(val, int)
        self._passed_count = val
