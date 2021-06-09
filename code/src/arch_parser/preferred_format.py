from enum import Enum


class PreferredFormat(Enum):
    """
    Preferred format to save an house.json file.
    """
    NONE = ("NoPreference", None)
    ARCH_JSON = ("ArchJson", ".arch.json")
    SCENE_JSON = ("SceneJson", ".scene.json")

    def __init__(self, name, extension):
        self._name = name
        self._extension = extension

    @property
    def name(self):
        return self._name

    @property
    def extension(self):
        return self._extension
