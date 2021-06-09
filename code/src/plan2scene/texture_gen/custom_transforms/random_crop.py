import numbers
from torchvision import transforms
from PIL import Image


class RandomCropAndDropAlpha(object):
    """
    Torchvision transform module that takes a random (opaque) crop from an alpha masked image.
    Returns the crop after dropping the alpha channel.
    """

    def __init__(self, size, iter_count: int):
        """
        Initializes transform.
        :param size: Size of the crop.
        :param iter_count: Maximum number of attempts made to take an opaque crop.
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.iter_count = iter_count

    def __call__(self, img: Image) -> Image.Image:
        """
        Obtains a random opaque crop from an alpha masked image.
        :param img: Image considered.
        :return: Crop if success or None.
        """
        if img.width < self.size[0] or img.height < self.size[1]:
            return None

        if len(img.getextrema()) == 3:
            # RGB
            i, j, h, w = transforms.RandomCrop.get_params(img, self.size)
            crop = img.crop((j, i, w + j, i + h))
            return crop

        for i in range(self.iter_count):
            i, j, h, w = transforms.RandomCrop.get_params(img, self.size)
            crop = img.crop((j, i, w + j, i + h))
            if crop.getextrema()[3][0] == 255:
                return crop.convert("RGB")


class RandomResizedCropAndDropAlpha(object):
    """
    Takes an opaque random resized crop and drop the alpha channel.
    """

    def __init__(self, size, iter_count, scale: tuple = (0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        """
        Initialize transform.
        :param size: Size of the output crop.
        :param iter_count: Max number of attempts to make an opaque crop.
        :param scale: Range of size of the origin size cropped.
        :param ratio: Range of aspect ratio of the origin aspect ratio cropped.
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.iter_count = iter_count
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        """
        Takes an opaque random resized crop and drop the alpha channel.
        :param img: Input image considered.
        :return: RGB crop.
        """
        if img.width < self.size[0] or img.height < self.size[1]:
            return None
        if len(img.getextrema()) == 3:
            # RGB
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, self.scale, self.ratio)
            crop = img.crop((j, i, w + j, i + h))
            crop = crop.resize(self.size)
            return crop

        for i in range(self.iter_count):
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, self.scale, self.ratio)
            crop = img.crop((j, i, w + j, i + h))
            crop = crop.resize(self.size)
            if crop.getextrema()[3][0] == 255:
                return crop.convert("RGB")
