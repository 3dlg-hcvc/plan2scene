from plan2scene.config_manager import ConfigManager
import torch
from plan2scene.evaluation.metric_impl.substance_classifier.util import get_model
from torchvision import transforms as tfs
import logging

from config_parser import parse_config, Config


def load_checkpoint(checkpoint_path, model) -> None:
    """
    Load checkpoint into the model.
    :param checkpoint_path: Saved path of the checkpoint.
    :param model: Model to load the checkpoint into.
    """
    ckpt = torch.load(checkpoint_path)
    logging.info("Loading checkpoint: %s" % checkpoint_path)
    logging.info(model.load_state_dict(ckpt["model_params"]))


class SubstanceClassifier:
    """
    Substance classification network
    """

    def __init__(self, classifier_conf: Config, params_path: str = None, checkpoint_path: str = None, device: str = None):
        """
        Initialize the substance classification network
        :param classifier_conf: Configuration of the substance classification metric.
        :param params_path: Optional. Path to configuration of the classification network.
        :param checkpoint_path: Optional. Path to checkpoints of the classification network.
        :param device: Optional. Device to use.
        """
        if params_path is None:
            params_path = classifier_conf.conf_path
        if checkpoint_path is None:
            checkpoint_path = classifier_conf.checkpoint_path

        self.params = parse_config(params_path)
        if device is None:
            device = self.params.device
        self.device = device

        self.transforms = tfs.Compose([tfs.ToTensor(),
                                       tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        self.model = get_model(self.params.arch, self.params.substances).to(device)

        load_checkpoint(checkpoint_path, self.model)
        self.model.eval()

    def __repr__(self):
        return "Subs"

    def predict(self, image):
        """
        Predict substance of a given image.
        :param image: Image to make the prediction on.
        :return: Predicted substance of the image.
        """
        with torch.no_grad():
            image_tf = self.transforms(image.convert("RGB")).unsqueeze(0).to(self.device)
            output = self.model(image_tf)
            _, pred = torch.max(output, dim=1)
            return self.params.substances[pred.item()]
