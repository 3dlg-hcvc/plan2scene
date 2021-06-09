from torch.utils.data import Dataset
from torchvision import transforms as tfs
import os
from PIL import Image
import os.path as osp
from plan2scene.utils.io import load_image

from plan2scene.texture_gen.custom_transforms.hsv_transforms import ToHSV
from plan2scene.texture_gen.custom_transforms.random_crop import RandomCropAndDropAlpha
import numpy as np


class ImageDataset(Dataset):
    """
    Dataset of images used to train the modified neural texture synthesis stage.
    """

    def __init__(self, data_path: str, image_res: tuple, resample_count: int, scale_factor: float, substances: list):
        """
        Initializes dataset.
        :param data_path: Path to dataset image.
        :param image_res: Size of output crops in the format (width, height)
        :param resample_count: Number of times to re-sample an image.
        :param scale_factor: Scale factor used to determine the random crop size.
        :param substances: List of substance labels.
        """
        self.data_path = data_path
        self.image_res = image_res
        self.resample_count = resample_count
        self.scale_factor = scale_factor
        self.substances = substances

        transforms = [
            RandomCropAndDropAlpha((self.image_res[0] * self.scale_factor,
                                    self.image_res[1] * self.scale_factor), 100000),
            tfs.Resize((self.image_res[0], self.image_res[1])),
            tfs.RandomHorizontalFlip(p=0.5),
            tfs.RandomVerticalFlip(p=0.5),
        ]

        self.transforms = tfs.Compose(transforms)

        self.samples = self._get_samples()

    def _get_samples(self) -> list:
        """
        Load dataset entries.
        :return: list of dataset entries
        """
        samples = []
        types = {'.jpeg', '.jpg', '.png'}

        files = os.listdir(self.data_path)
        files = [a for a in files if osp.splitext(a)[1] in types]
        for file in files:
            img = load_image(osp.join(self.data_path, file))
            if img.width > self.image_res[0] * self.scale_factor and img.height > self.image_res[1] * self.scale_factor:
                samples.append(osp.join(self.data_path, file))

        samples.sort()
        samples = samples * self.resample_count
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """
        Return dataset entry.
        :param idx: Index
        :return: tuple(RGB tensor, file path, substance label index, HSV tensor)
        """
        filepath = self.samples[idx]
        substance = -1
        if self.substances is not None:
            substance = filepath.split("/")[-1].split("_")[0]
            substance = self.substances.index(substance)

        image = Image.open(filepath)
        image.load()
        if image.mode not in ["RGB", "RGBA"]:
            image = image.convert("RGB")

        image_transformed = self.transforms(image)
        image_tensor = tfs.ToTensor()(image_transformed)
        image_hsv = ToHSV()(image_transformed)
        image_hsv_tensor = tfs.ToTensor()(image_hsv)
        return image_tensor, filepath, substance, image_hsv_tensor
