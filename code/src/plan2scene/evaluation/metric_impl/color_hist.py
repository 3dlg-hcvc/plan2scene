import numpy as np
from PIL import Image
import colorsys

"""
Methods implementing the COLOR metric.
"""


def generate_bins() -> list:
    """
    Generate color bins.
    :return: List of bins
    """
    h_bins = [(x / 10.0, (x + 1) / 10.0) for x in range(0, 10)]
    h_bins[-1] = (h_bins[-1][0], 1.1)
    s_bins = [(0.0, 0.333), (0.333, 0.666), (0.666, 1.1)]
    l_bins = [(0.0, 0.333), (0.333, 0.666), (0.666, 1.1)]
    bins = []
    for h_bin in h_bins:
        for s_bin in s_bins:
            for l_bin in l_bins:
                bins.append((h_bin, s_bin, l_bin))
    return bins


def to_hsl(img: Image.Image) -> np.ndarray:
    """
    Converts an image to HSL format and return as a numpy array.
    :param img: Image to convert
    :return: Converted image.
    """
    img_rgb = img.convert("RGB")
    img_rgb_np = np.array(img_rgb) / 255.0
    rgb_hls = np.vectorize(colorsys.rgb_to_hls)
    np_h, np_l, np_s = rgb_hls(r=img_rgb_np[:, :, 0], g=img_rgb_np[:, :, 1], b=img_rgb_np[:, :, 2])
    img_hsl_np = np.concatenate([np.expand_dims(np_h, 2), np.expand_dims(np_s, 2), np.expand_dims(np_l, 2)], axis=2)
    return img_hsl_np


def compute_histogram(img_hsl_np: np.ndarray, bins: list) -> np.ndarray:
    """
    Computes HSL histogram for a given image in HSL color space.
    :param img_hsl_np: Image in HSL color space.
    :param bins: Bin specification.
    :return: Histogram
    """
    freqs = []
    for h_bound, s_bound, l_bound in bins:
        count = np.logical_and(
            np.logical_and(
                np.logical_and(img_hsl_np[:, :, 0] >= h_bound[0], img_hsl_np[:, :, 0] < h_bound[1]),
                np.logical_and(img_hsl_np[:, :, 1] >= s_bound[0], img_hsl_np[:, :, 1] < s_bound[1])),
            np.logical_and(img_hsl_np[:, :, 2] >= l_bound[0], img_hsl_np[:, :, 2] < l_bound[1])).sum()
        freqs.append(count)
    return np.array(freqs)


def hsl_hist_l1(pred: Image, gt:Image, bins:list):
    """
    Compute COLOR metric between prediction and ground truth using the specified bins.
    :param pred: Predicted texture
    :param gt: Ground truth texture
    :param bins: Bin specification returned from generate_bins
    :return: Metric value.
    """
    pred_hist = compute_histogram(to_hsl(pred), bins)
    gt_hist = compute_histogram(to_hsl(gt), bins)
    dif = gt_hist - pred_hist
    score = np.sum(np.abs(dif)) / (2.0 * pred.width * pred.height)
    return score
