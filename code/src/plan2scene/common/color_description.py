from enum import Enum


class ColorSpace(Enum):
    HSV = "HSV",
    RGB = "RGB"


class Color:
    """
    Describes a color and its color space.
    """
    def __init__(self, color_space: ColorSpace, components: list):
        """
        Initialize color.
        :param color_space: Color space of the color.
        :param components: Components of the color.
        """
        assert isinstance(color_space, ColorSpace)
        assert isinstance(components, list)
        self._color_space = color_space
        self._components = components

    @property
    def color_space(self) -> ColorSpace:
        """
        Retrieve colorspace of the color.
        :return: Colorspace of the color.
        """
        return self._color_space

    @property
    def components(self) -> list:
        """
        Retrieve color components.
        :return: Color components.
        """
        return self._components

    def to_dict(self):
        return {
            "color_space": self.color_space.name,
            "components": self.components
        }
