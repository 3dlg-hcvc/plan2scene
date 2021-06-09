from plan2scene.config_manager import ConfigManager
import os
import os.path as osp
import logging
import torch
from plan2scene.texture_gen.custom_transforms.random_crop import RandomCropAndDropAlpha
from plan2scene.texture_gen.utils.io import load_conf_eval
import torchvision.transforms as tfs
from PIL import Image

if __name__ == "__main__":
    """
    Extract rectified surface crops from rectified surface masks.
    """

    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Extract rectified surface crops from rectified surface masks.")
    conf.add_args(parser)
    parser.add_argument("output_path",
                        help="Output directory to create the rectified crops. Usually './data/processed/rectified_crops/[surface]")
    parser.add_argument("rectified_surfaces_path",
                        help="Path to directory with rectified surfaces. Usually './data/input/rectified_masks/[surface].")
    args = parser.parse_args()
    conf.process_args(args,output_is_dir=True)

    # Work
    output_path = args.output_path
    rectified_surfaces_path = args.rectified_surfaces_path

    param = load_conf_eval(conf.texture_gen.texture_synth_conf)
    crop_size = param.image.image_res
    crop_count = conf.texture_gen.crops_per_mask

    with open(osp.join(output_path, "index.htm"), "w") as f:
        f.write("<table>\n")
        f.write("<tr><th>#</th><th>File</th><th>Rectified Surface</th>")
        for c in range(crop_count):
            f.write("<th>Crop %d</th>" % (c))
        f.write("</tr>\n")

        rectified_surface_files = [a for a in os.listdir(rectified_surfaces_path) if "image" not in a]
        for i, file in enumerate(rectified_surface_files):
            f.write("<tr><td>%d</td>" % (i))
            f.write("<td>%s</td>" % file)

            img = Image.open(osp.join(rectified_surfaces_path, file))
            img.save(osp.join(output_path, file.replace(".png", "_input.png")))
            f.write("<td><img src='%s' style='width:350px;' /></td>" % (file.replace(".png", "_input.png")))

            j = 0
            for _ in range(crop_count):
                logging.info("[%d/%d] Cropping %s: (%d/%d)" % (i, len(rectified_surface_files), file, j, crop_count))
                img = Image.open(osp.join(rectified_surfaces_path, file))
                transforms = tfs.Compose([
                    tfs.Lambda(lambda a: a.resize(
                        (a.width * conf.texture_gen.input_scale_up, a.height * conf.texture_gen.input_scale_up))),
                    RandomCropAndDropAlpha((param.image.image_res * param.image.scale_factor,
                                            param.image.image_res * param.image.scale_factor), 1000),
                ])

                crop = transforms(img)
                if crop is None:
                    logging.info("Unable to extract crop %d from %s" % (j, file))
                    continue
                crop = tfs.Resize((param.image.image_res, param.image.image_res))(crop)
                crop.save(osp.join(output_path, file.replace(".png", "_crop%d.png" % (j))))
                f.write("<td><img src='%s'/></td>" % (file.replace(".png", "_crop%d.png" % (j))))
                j += 1

            f.write("</tr>")
            f.flush()
        f.write("</table>\n")


