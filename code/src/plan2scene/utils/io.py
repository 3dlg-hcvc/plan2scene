from PIL import Image
import os.path as osp
import os
import logging

def load_smt_dataset(smt_path: str, image_size):
    """
    Load substance mapped textures dataset
    :param smt_path: Path to SMT dataset
    :return: Dictionary mapping from texture key to texture    """

    textures = {}
    # for substance_type in os.listdir(smt_path):
    #     if osp.isdir(osp.join(smt_path, substance_type)):
    for texture_name in os.listdir(osp.join(smt_path)):
        if texture_name.endswith(".jpg"):
            tex = load_image(osp.join(smt_path, texture_name))
            tex = tex.resize((image_size, image_size))
            textures[osp.join(smt_path, texture_name)] = tex
                    # textures.append(tex)
    return textures


def load_image(image_path: str) -> Image.Image:
    """
    Loads an image as RGB/RGBA PIL Image
    :param image_path: Path to Image
    :return: loaded image
    """
    sample_image = Image.open(image_path)
    if sample_image.mode not in ["RGB", "RGBA"]:
        sample_image = sample_image.convert("RGB")

    sample_image.load()
    return sample_image

def fraction_str(numerator, denominator) -> str:
    """
    Formats a fraction to the format %.5f [numerator/deniminator]
    :param numerator:
    :param denominator:
    :return: Formatted string
    """
    if denominator > 0:
        return "%.5f [%.5f/%d]" % (float(numerator)/denominator, numerator, denominator)
    else:
        return "No Data"