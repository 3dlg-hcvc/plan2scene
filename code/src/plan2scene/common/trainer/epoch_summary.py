class EpochSummary:
    """
    Evaluation summary for an epoch
    """

    def __init__(self, epoch_loss: float, epoch_batch_count: int, epoch_entry_count: int):
        """
        Initalize epoch summary
        :param epoch_loss: Loss reported for the epoch
        :param epoch_batch_count: Number of batches in the epoch
        :param epoch_entry_count: Number of dataset entries in the epoch
        """
        assert isinstance(epoch_loss, float)
        assert isinstance(epoch_batch_count, int)
        assert isinstance(epoch_entry_count, int)

        self.epoch_loss = epoch_loss
        self.epoch_entry_count = epoch_entry_count
        self.epoch_batch_count = epoch_batch_count

    @property
    def epoch_loss(self) -> float:
        """
        Retrieve epoch loss
        :return: Epoch loss
        """
        return self._epoch_loss

    @epoch_loss.setter
    def epoch_loss(self, value: float):
        """
        Set epoch loss
        """
        assert isinstance(value, float)
        self._epoch_loss = value

    @property
    def epoch_batch_count(self) -> int:
        """
        Retrieve batch count
        :return: batch count
        """
        return self._epoch_batch_count

    @epoch_batch_count.setter
    def epoch_batch_count(self, value: int):
        """
        Set batch count
        """
        assert isinstance(value, int)
        self._epoch_batch_count = value

    @property
    def epoch_entry_count(self) -> int:
        """
        Retrieve number of dataset entries considered for the epoch
        """
        return self._epoch_entry_count

    @epoch_entry_count.setter
    def epoch_entry_count(self, value: int):
        """
        Set the number of entries for the epoch
        """
        assert isinstance(value, int)
        self._epoch_entry_count = value
