#!/bin/python3
import sys
import os
import os.path as osp
from PIL import Image
from torchvision import transforms
import argparse
import logging

from plan2scene.config_manager import ConfigManager
from plan2scene.texture_gen.custom_transforms.random_crop import RandomResizedCropAndDropAlpha

if __name__ == "__main__":
    """
    This script is used to prepare rectified surface crops from OpenSurfaces dataset, which we use to train the substance classifier.
    """

    parser = argparse.ArgumentParser(description="Extract rectified surface crops from the OpenSurfaces dataset.")
    parser.add_argument("output_path", type=str, help="Output directory to save extracted crops.")
    parser.add_argument("input_path", type=str, help="Directory containing rectified surface masks from OpenSurfaces dataset.")
    conf = ConfigManager()
    conf.add_args(parser)
    args = parser.parse_args()
    conf.process_args(args)

    output_path = args.output_path
    input_path = args.input_path

    # Configuration used
    crop_count = 10
    crop_size = (85, 85)
    output_size = (128, 128)
    image_scaleup = 1
    second_crop_min_scale = 0.25

    if osp.exists(output_path):
        logging.error("Output directory already exist")
        sys.exit(1)

    if not osp.exists(output_path):
        os.makedirs(output_path)

    image_file_paths = [osp.join(input_path, a) for a in os.listdir(input_path)]
    logging.info("Found {count} files.".format(count = len(image_file_paths)))

    with open(osp.join(output_path, "index.html"), "w") as f:
        for image_file_path in image_file_paths:
            img_name = image_file_path.split("/")[-1]
            img = Image.open(image_file_path)
            index = 0
            for i in range(crop_count):
                crop = RandomResizedCropAndDropAlpha(crop_size,100, ratio=(1.0,1.0))(img)
                if crop is not None:
                    crop = transforms.RandomResizedCrop(size=(crop_size),ratio=(1.0,1.0), scale=(second_crop_min_scale,1.0))(crop)
                    crop = crop.resize(output_size)
                    logging.info("Saved {file}.".format(file=osp.join(output_path, img_name.split(".")[0] + "_crop%d.png" % index)))
                    crop.save(osp.join(output_path, img_name.split(".")[0] + "_crop%d.png" % index))
                    f.write("<div style='float:left; margin:5px;'><img src='%s'/><br><small>%s</small></div>" % (img_name.split(".")[0] + "_crop%d.png" % index, img_name))
                    index += 1
            f.flush()