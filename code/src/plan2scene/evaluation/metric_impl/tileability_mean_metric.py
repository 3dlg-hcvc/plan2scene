import numpy as np
from PIL import Image

"""
Utility methods used to compute the TILE metric.
"""


def image_fft(image_np: np.ndarray) -> np.ndarray:
    """
    Computes FFT of an image
    :param image_np: Image to compute fft.
    :return: FFT of the image.
    """
    image_fft_shift = np.fft.fftshift(np.fft.fft2(image_np))
    return image_fft_shift


def image_ifft(image_fft: np.ndarray) -> np.ndarray:
    """
    Compute inverse FFT of an image.
    :param image_fft: FFT image
    :return: Invert FFT image.
    """
    return np.abs(np.fft.ifft2(np.fft.ifftshift(image_fft)))


def get_gaussian(sig: float, size: int) -> np.ndarray:
    """
    Creates gaussian kernel with side length 'size' and a sigma of 'sig'.
    Source: https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    l = size
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    kernel = kernel / np.sum(kernel)
    kernel[size // 2, size // 2] = 0
    kernel = np.abs(kernel)

    return kernel


def compute_mean_tileability(img: Image.Image, gaus: np.ndarray):
    """
    Computes TILE metric.
    :param img: Predicted texture.
    :param gaus: Gaussian kernel
    :return: Metric value
    """
    img_gray = img.convert("L")

    img_fft = image_fft(np.array(img_gray))
    image_fft_blured = img_fft * gaus
    score = np.linalg.norm(np.reshape(image_fft_blured, (image_fft_blured.shape[0] * image_fft_blured.shape[1],)),
                           ord=2)
    return score
