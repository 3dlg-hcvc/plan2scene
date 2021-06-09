from PIL import Image
from abc import ABC, abstractmethod
import torch

from plan2scene.common.image_description import ImageDescription
from plan2scene.evaluation.metrics import AbstractPairedMetric, AbstractUnpairedMetric

"""
A matcher pairs a predicted texture with a ground truth reference crop (if available).
The matcher informs the evaluator whether to include or exclude a prediction from the evaluation. 
    (E.g. textures predicted for surfaces without ground truth crops should be excluded.)
"""


class AbstractMatcher(ABC):
    @abstractmethod
    def __call__(self, pred: Image.Image, gt_textures: dict) -> tuple:
        """
        Invokes matcher.
        :param pred: Predicted texture for a surface.
        :param gt_textures: Dictionary containing ground truth references for the surface.
        :return: Tuple of (Metric value, Matched ground truth image, True if the texture should be included in the evaluation.)
        """
        pass


class PairedMatcher(AbstractMatcher):
    """
    Wrapper for metrics that require a pair of inputs (prediction and ground truth).
    """

    def __init__(self, metric: AbstractPairedMetric):
        """
        Initializes paired matcher.
        :param metric: Metric used to evaluate.
        """
        super().__init__()
        self.metric = metric

    def __repr__(self):
        return str(self.metric)

    def __call__(self, pred: Image.Image, gt_textures: dict) -> tuple:
        """
        Evaluates prediction (crop) against the gt reference in gt_textures.
        :param pred: Prediction
        :param gt_textures: Dictionary containing the gt texture. Should only contain a single ground truth texture.
        :return: If success, return Tuple of metric result, ground truth texture, True. Else return None, None, False.
        """
        pred_texture = pred
        if isinstance(pred_texture, ImageDescription):
            pred_texture = pred_texture.image
        with torch.no_grad():
            if len(gt_textures) == 0:
                return None, None, False

            assert len(gt_textures) == 1
            gt_texture = list(gt_textures.values())[0]
            if isinstance(gt_texture, ImageDescription):
                gt_texture = gt_texture.image
            loss = self.metric(pred_texture, gt_texture)

            return loss, gt_texture, True


class UnpairedMatcher(AbstractMatcher):
    """
    Wrapper for metrics that only require the prediction to evaluate. (E.g. TILE)
    """

    def __init__(self, metric: AbstractUnpairedMetric):
        """
        Initializes unpaired matcher.
        :param metric: Metric used
        """
        super().__init__()
        self.metric = metric

    def __repr__(self):
        return str(self.metric)

    def __call__(self, pred: Image, gt_textures: dict):
        """
        Evaluates prediction (crop).
        :param pred: Prediction
        :param gt_textures: Dictionary containing the gt texture. Not used.
        :return: return Tuple of metric result, None, True.
        """
        pred_texture = pred
        if isinstance(pred_texture, ImageDescription):
            pred_texture = pred_texture.image
        with torch.no_grad():
            loss = self.metric(pred_texture)
            return loss, None, True
