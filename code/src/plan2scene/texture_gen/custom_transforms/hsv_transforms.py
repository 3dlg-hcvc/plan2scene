class ToHSV(object):
    """
    TorchVision transform module that converts an image to HSV format.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        return img.convert("HSV")
