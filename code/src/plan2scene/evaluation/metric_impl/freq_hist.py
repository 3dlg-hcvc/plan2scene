from scipy import ndimage
from moisan2011 import per
import numpy as np
from PIL import Image


def getPSD1D(psd2D: np.ndarray) -> np.ndarray:
    # Source: https://medium.com/tangibit-studios/2d-spectrum-characterization-e288f255cc59
    h = psd2D.shape[0]
    w = psd2D.shape[1]
    wc = w // 2
    hc = h // 2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r = np.hypot(X - wc, Y - hc).astype(np.int)

    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = ndimage.mean(psd2D, r, index=np.arange(0, wc))

    # Set DC to zero
    psd1D[0] = 0
    return psd1D


def compute_histogram(texture: Image.Image) -> np.ndarray:
    """
    Compute frequency histogram for a given image.
    :param texture: Image to compute the histogram.
    :return: Frequency histogram computed on texture.
    """
    img, _ = per(np.array(texture.convert("L")))

    np_img = np.abs(img).astype(np.float32)
    np_img_c = np_img[:, :]
    f = np.fft.fft2(np_img_c, norm=None) / (texture.width * texture.height)
    f_shift = np.fft.fftshift(f)

    hist = getPSD1D(np.abs(f_shift))
    return hist


def compute_freq_hist_l1(pred: Image.Image, gt: Image.Image):
    """
    Compute FREQ metric between given predicted image and the ground truth image.
    :param pred: Prediction image
    :param gt: Ground truth image
    :return: FREQ metric.
    """
    pred_hist = compute_histogram(pred)
    gt_hist = compute_histogram(gt)

    dif_hist = gt_hist - pred_hist
    total_abs_difference_pos = np.mean(np.abs(dif_hist))
    return total_abs_difference_pos
