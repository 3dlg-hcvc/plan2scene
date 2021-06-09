from enum import Enum
import os.path as osp
from PIL import Image
import json

from plan2scene.utils.io import load_image


class ImageSource(Enum):
    """
    Image source describes the origin of a texture/crop loaded into a house.
    """
    DIRECT_CROP = ("Crop", None)  # The image is a crop extracted from an image
    NEURAL_SYNTH = ("NeuralSynth", None)  # The image is a synthesized texture
    VGG_CROP_SELECT = ("VGGCropSelect", True)  # The image is synthesized based on VGG Crop Selects suggestion (Output for observed surfaces)
    GNN_PROP = ("GNNProp", False)  # The image is a synthesized based on GNNs suggestions (Output for unobserved surfaces)
    MULTI_PROP = ("MultiProp", None)  # The image is synthesized by multiprop script.
    RETRIEVE_OBSERVED = ("RetrieveObserved", True)  # The image is chosen by the retrieve baseline for an observed surface
    RETRIEVE_UNOBSERVED = ("RetrieveUnobserved", False)  # This image is chosen by the retrieve baseline for an unobserved surface
    MEAN_EMB = ("MeanEmb", True)  # Image synthesized from a mean embedding crop
    RS_MEAN_EMB = ("RSMeanEmb", True)  # Image synthesized from a room type and surface type conditioned mean embedding crop

    def __init__(self, name, observed):
        self._name = name
        self._observed = observed

    @property
    def name(self) -> str:
        return self._name

    @property
    def observed(self) -> bool:
        """
        Return true if the crop/texture is for an observed surface.
        Return False if the crop/texture is for an unobserved surface.
        Return None if unspecified.
        """
        return self._observed

    @classmethod
    def parse(cls, _str):
        """
        Parse from string.
        :param _str: String to parse.
        :return: ImageSource
        """
        for member_key, member_value in ImageSource.__members__.items():
            if _str == member_value.name:
                return member_value
        assert False


class ImageDescription:
    """
    Wrapper on a PIL image containing additional information such as its source.
    """
    def __init__(self, image: Image.Image, source: ImageSource):
        """
        Initialze image description.
        :param image: PIL Image
        :param source: Image source
        """
        assert isinstance(source, ImageSource)
        assert isinstance(image, Image.Image)
        self._image = image
        self._source = source

    @property
    def image(self) -> Image.Image:
        """
        Return PIL Image
        :return: Image
        """
        return self._image

    @image.setter
    def image(self, value: Image.Image) -> None:
        """
        Set PIL Image
        :param value: Image
        """
        assert isinstance(value, Image.Image)
        self._image = value

    @property
    def source(self) -> ImageSource:
        """
        Return source of the image.
        :return: image source
        """
        return self._source

    def save(self, path: str) -> None:
        """
        Save image to disk
        :param path: Save path
        """
        self.image.save(path)
        crop_name = osp.splitext(path)[0]
        with open(crop_name + ".desc.json", "w") as f:
            json.dump({
                "image_source": self.source.name
            }, f)

    @classmethod
    def parse_image(cls, image_path: str):
        """
        Loads an image description form the disk.
        :param image_path: Saved path.
        :return: Loaded image description
        """
        crop_image = load_image(image_path)
        crop_name = osp.splitext(image_path)[0]
        source = ImageSource.DIRECT_CROP
        if osp.exists(crop_name + ".desc.json"):
            with open(crop_name + ".desc.json") as f:
                desc_json = json.load(f)
            source = ImageSource.parse(desc_json["image_source"])
        return ImageDescription(crop_image, source)


