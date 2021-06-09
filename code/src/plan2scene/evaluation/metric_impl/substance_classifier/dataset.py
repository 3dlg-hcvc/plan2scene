import logging

from torchvision import transforms
import os
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def map_label(label, mapping):
    if label in mapping:
        return mapping[label]
    return label


class SubstanceCropDataset(Dataset):
    """
    Dataset used to train the substance classifier. Feeds crops from the textures dataset and the open-surfaces dataset.
    """
    def __init__(self, os_dataset_path, texture_dataset_path, label_mapping, substances, train: bool):
        """
        Initializes the dataset.
        :param os_dataset_path: Path to the open-surfaces dataset.
        :param texture_dataset_path: Path to the textures dataset.
        :param label_mapping: Label mapping used to merge different labels of two dataset.
        :param substances: List of substance labels.
        :param train: Specify true if the dataset is used for training. Specify false if the dataset is used for validation.
        """
        val_multicrop = False
        self.entries = []
        self.train = train
        self.substances = substances
        additional_transforms = []
        if train:
            additional_transforms.append(transforms.RandomHorizontalFlip())
            additional_transforms.append(transforms.RandomVerticalFlip())
        self.transforms = transforms.Compose([
            transforms.Compose(additional_transforms),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        if os_dataset_path is not None:
            os_count = 0
            for file in os.listdir(os_dataset_path):
                if not train and not osp.splitext(file)[0].endswith("_crop0"): # Validation uses crop0 only
                    continue
                if ".png" not in file:
                    continue
                substance_part = file.split("_")[0]
                entry = {
                    "file_path": osp.join(os_dataset_path, file),
                    "substance": map_label(substance_part, label_mapping),
                    "source": "os"
                }
                self.entries.append(entry)
                os_count += 1
            logging.info("Loaded {count} images from OpenSurfaces.".format(count=os_count))

        if texture_dataset_path is not None:
            tex_count = 0
            for file in os.listdir(texture_dataset_path):
                if not train and not osp.splitext(file)[0].endswith("_crop0"):
                    continue # Validation should include crop0 only.
                if ".png" not in file:
                    continue

                substance_part = file.split("_")[0]
                entry = {
                    "file_path": osp.join(texture_dataset_path, file),
                    "substance": map_label(substance_part, label_mapping),
                    "source": "textures"
                }
                self.entries.append(entry)
                tex_count += 1
        logging.info("Loaded {count} images from Textures dataset.".format(count=tex_count))

        np.random.shuffle(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        substance = entry["substance"]
        substance_id = self.substances.index(substance)

        img = Image.open(entry["file_path"])
        img_transformed = self.transforms(img)
        return img_transformed, substance_id, entry