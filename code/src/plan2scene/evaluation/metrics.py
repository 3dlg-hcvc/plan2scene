from abc import ABC, abstractmethod
from PIL import Image
import torchvision.transforms as tfs
import torch
import torch.nn as nn

from config_parser import Config
from plan2scene.evaluation.metric_impl.substance_classifier.classifier import SubstanceClassifier


class AbstractPairedMetric(ABC):
    """
    Paired metric which uses a prediction ground truth pair.
    """

    @abstractmethod
    def __call__(self, pred_texture: Image.Image, gt_texture: Image.Image) -> float:
        """
        Invokes paired metric.
        :param pred_texture: Predicted texture.
        :param gt_texture: Ground truth image.
        :return: Computed score.
        """
        pass


class AbstractUnpairedMetric(ABC):
    """
    Unpaired metric which only uses the prediction for evaluation. (Doesn't require a ground truth)
    """

    @abstractmethod
    def __call__(self, pred_texture: Image.Image) -> float:
        """
        Invokes the metric.
        :param pred_texture: Predicted texture
        :return: Computed score.
        """
        pass


class FreqHistL1(AbstractPairedMetric):
    """
    FREQ metric.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "Freq"

    def __call__(self, pred_texture: Image.Image, gt_texture: Image.Image) -> float:
        """
        Evaluates the predicted texture using the FREQ metric.
        :param pred_texture: Predicted texture.
        :param gt_texture: Ground truth reference image.
        :return: Metric loss.
        """
        from plan2scene.evaluation.metric_impl.freq_hist import compute_freq_hist_l1
        return compute_freq_hist_l1(pred=pred_texture, gt=gt_texture)


class TileabilityMean(AbstractUnpairedMetric):
    """
    TILE metric.
    """

    def __init__(self, metric_param: Config):
        """
        Initializes metric.
        :param metric_param: Specify configuration of the metric.
        """
        from plan2scene.evaluation.metric_impl.tileability_mean_metric import get_gaussian
        self.kernel_size = metric_param.kernel_size
        self.std = metric_param.gaussian_std
        self.gaus = get_gaussian(sig=self.std, size=self.kernel_size)

    def __repr__(self):
        return "Tile (%d)" % self.kernel_size

    def __call__(self, pred_texture: Image.Image) -> float:
        """
        Evaluates predicted texture using the TILE metric.
        :param pred_texture:
        :return: Metric loss.
        """
        from plan2scene.evaluation.metric_impl.tileability_mean_metric import compute_mean_tileability
        score = compute_mean_tileability(img=pred_texture, gaus=self.gaus)
        return score


class HSLHistL1(AbstractPairedMetric):
    """
    COLOR metric.
    """

    def __init__(self):
        """
        Initialize COLOR metric.
        """
        from plan2scene.evaluation.metric_impl.color_hist import generate_bins
        self.bins = generate_bins()
        pass

    def __repr__(self):
        return "Color"

    def __call__(self, pred_texture: Image.Image, gt_texture: Image.Image) -> float:
        """
        Evaluates predicted texture using the COLOR metric.
        :param pred_texture: Predicted texture.
        :param gt_texture: Ground truth reference image.
        :return: Metric loss.
        """
        from plan2scene.evaluation.metric_impl.color_hist import hsl_hist_l1
        return hsl_hist_l1(pred=pred_texture, gt=gt_texture, bins=self.bins)


class ClassificationError(AbstractPairedMetric):
    """
    SUBS metric.
    """

    def __init__(self, classifier: SubstanceClassifier):
        """
        Initializes SUBS metric.
        :param classifier: Substance classifier used for evaluation purpose.
        """
        self.classifier = classifier

    def __repr__(self):
        return str(self.classifier)

    def __call__(self, pred_texture, gt_texture):
        """
        Evaluates predicted texture using the SUBS metric.
        :param pred_texture: Predicted texture.
        :param gt_texture: Ground truth texture.
        :return: Metric loss.
        """
        pred_class = self.classifier.predict(pred_texture)
        gt_class = self.classifier.predict(gt_texture)
        if pred_class == gt_class:
            return 0
        else:
            return 1


class CorrespondingPixelL1:
    """
    Measures Pixel L1 Loss between two images. Used by the retrieve baseline.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "Corresponding Pixel L1"

    def __call__(self, pred_texture: Image.Image, gt_texture: Image.Image) -> float:
        """
        Measures corresponding pixel L1 distance between predicted texture and the ground truth image.
        :param pred_texture: Predicted texture.
        :param gt_texture: Ground truth image.
        :return: Metric loss
        """
        pred_tf_unsigned = tfs.ToTensor()(pred_texture.convert("RGB")).unsqueeze(0)
        gt_tf_unsigned = tfs.ToTensor()(gt_texture).unsqueeze(0)

        return nn.functional.l1_loss(pred_tf_unsigned, gt_tf_unsigned).item()
