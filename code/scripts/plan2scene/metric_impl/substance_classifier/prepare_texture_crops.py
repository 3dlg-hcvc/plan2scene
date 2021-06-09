#!/bin/python3

import sys
import os
import os.path as osp
from PIL import Image
import argparse
import logging

from plan2scene.config_manager import ConfigManager
from plan2scene.texture_gen.custom_transforms.random_crop import RandomResizedCropAndDropAlpha

if __name__ == "__main__":
    """
    This script is used to prepare texture crops (from texture dataset) used to train the substance classifier.
    """

    parser = argparse.ArgumentParser(description="Extract rectified surface crops from the opensurfaces dataset, to train the substance classifier.")
    parser.add_argument("output_path", type=str, help="Output directory to save texture crops.")
    parser.add_argument("input_path", type=str, help="Directory containing textures.")
    parser.add_argument("--crops-per-image", type=int, default=20)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--output-size", type=int, default=128)
    parser.add_argument("--attempt-count", type=int, default=100)
    conf = ConfigManager()
    conf.add_args(parser)
    args = parser.parse_args()
    conf.process_args(args)

    output_path = args.output_path
    input_path = args.input_path

    # Configuration used
    crop_count = args.crops_per_image
    crop_size = (args.crop_size, args.crop_size)
    output_size = (args.output_size, args.output_size)
    attempt_count = args.attempt_count

    if osp.exists(output_path):
        logging.error("Output directory already exist")
        sys.exit(1)

    if not osp.exists(output_path):
        os.makedirs(output_path)

    image_file_paths = [osp.join(input_path, a) for a in os.listdir(input_path)]
    image_file_paths = [a for a in image_file_paths if osp.splitext(a)[1] in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]]
    logging.info("Found {count} files.".format(count=len(image_file_paths)))

    with open(osp.join(output_path, "index.html"), "w") as f:
        for image_file_path in image_file_paths:
            img_name = image_file_path.split("/")[-1]
            img = Image.open(image_file_path)
            index = 0
            for i in range(crop_count):
                crop = RandomResizedCropAndDropAlpha(crop_size, attempt_count, ratio=(1.0, 1.0))(img)
                if crop is not None:
                    crop = crop.resize(output_size)
                    crop.save(osp.join(output_path, img_name.split(".")[0] + "_crop%d.png" % index))
                    logging.info("Saved {file}.".format(file=osp.join(output_path, img_name.split(".")[0] + "_crop%d.png" % index)))
                    f.write("<div style='float:left; margin:5px;'><img src='%s'/><br><small>%s</small></div>" % (
                    img_name.split(".")[0] + "_crop%d.png" % index, img_name))
                    index += 1
            f.flush()
