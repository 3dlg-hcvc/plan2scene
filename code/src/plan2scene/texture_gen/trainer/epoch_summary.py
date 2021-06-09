from plan2scene.common.trainer.epoch_summary import EpochSummary


class TextureGenEpochSummary(EpochSummary):
    """
    Epoch evaluation results of modified neural texture synthesis.
    """

    def __init__(self, epoch_style_loss: float, epoch_substance_loss: float, epoch_substance_passed: int, *args, **kwargs):
        """
        Initializes TextureGenEpochSummary.
        :param epoch_style_loss: VGG style loss for epoch
        :param epoch_substance_loss:  Substance loss for epoch
        :param epoch_substance_passed: Number of crops that were correctly classified.
        """
        super().__init__(*args, **kwargs)
        assert isinstance(epoch_style_loss, float)
        assert isinstance(epoch_substance_loss, float)
        assert isinstance(epoch_substance_passed, int)

        self._epoch_style_loss = epoch_style_loss
        self._epoch_substance_loss = epoch_substance_loss
        self._epoch_substance_passed = epoch_substance_passed

    @property
    def epoch_style_loss(self) -> float:
        """
        Return VGG style loss computed for an epoch.
        :return: VGG style loss computed for an epoch.
        """
        return self._epoch_style_loss

    @epoch_style_loss.setter
    def epoch_style_loss(self, value: float) -> None:
        """
        Set VGG style loss computed for an epoch.
        :param value: VGG style loss computed for an epoch.
        """
        assert isinstance(value, float)
        self._epoch_style_loss = value

    @property
    def epoch_substance_loss(self) -> float:
        """
        Return substance classification loss.
        :return: Substance classification loss
        """
        return self._epoch_substance_loss

    @epoch_substance_loss.setter
    def epoch_substance_loss(self, value: float) -> None:
        """
        Set substance classification loss
        :param value: Substance classification loss
        """
        assert isinstance(value, float)
        self._epoch_substance_loss = value

    @property
    def epoch_substance_passed(self) -> int:
        """
        Returns count of entries that received correct substance prediction.
        :return: Count of entries that received correct substance prediction.
        """
        return self._epoch_substance_passed

    @epoch_substance_passed.setter
    def epoch_substance_passed(self, value: int) -> None:
        """
        Set the count of entries that received correct substance prediction.
        :param value: Count of entries that received correct substance prediction.
        """
        assert isinstance(value, int)
        self._epoch_substance_passed = value
