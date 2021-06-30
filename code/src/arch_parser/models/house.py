from arch_parser.models.cad_model import CADModel
from arch_parser.models.room import Room
from arch_parser.models.wall import Wall
from PIL import Image, ImageDraw

from collections.abc import Iterable
from arch_parser.models.wall_room_assignment import WallRoomAssignment
from arch_parser.preferred_format import PreferredFormat


class House:
    """
    A house
    """

    def __init__(self, house_key):
        assert isinstance(house_key, str)

        self._rooms = {}  # room_id: room
        self._house_key = house_key  # From the file name
        self._door_connected_room_pairs = []  # Duplicates edges. The edges to outside world are not duplicated.
        self._arch_extra_metadata = {}
        self._scene_extra_metadata = {}
        self._preferred_format = PreferredFormat.NONE
        self._cad_models = []

    @property
    def cad_models(self):
        return self._cad_models

    @cad_models.setter
    def cad_models(self, value):
        assert isinstance(value, Iterable)
        value = list(value)
        for cad_model in value:
            assert isinstance(cad_model, CADModel)

        self._cad_models = value

    @property
    def preferred_format(self):
        return self._preferred_format

    @preferred_format.setter
    def preferred_format(self, value):
        assert isinstance(value, PreferredFormat)
        self._preferred_format = value

    @property
    def scene_extra_metadata(self):
        return self._scene_extra_metadata

    @property
    def arch_extra_metadata(self):
        return self._arch_extra_metadata

    @property
    def house_key(self):
        return self._house_key

    @property
    def rooms(self) -> dict:
        return self._rooms

    @property
    def door_connected_room_pairs(self):
        return self._door_connected_room_pairs

    def __str__(self):
        return "House: %s" % self._house_key

    def compute_bounds(self):
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        for room_id, room in self.rooms.items():
            assert isinstance(room, Room)
            for wall_assignment in room.walls:
                assert isinstance(wall_assignment, WallRoomAssignment)
                wall = wall_assignment.wall
                for (x, _, y) in [wall.p1, wall.p2]:
                    min_x = min(x, min_x)
                    max_x = max(x, max_x)
                    min_y = min(y, min_y)
                    max_y = max(y, max_y)

        return min_x, min_y, max_x, max_y

    def sketch_house(self, image_size=400, margin=10, wall_color=(255, 0, 0), focused_color=(0, 255, 0), focused_room_id=None) -> Image.Image:
        """
        Generates a top-down sketch of the house.
        :param focused_color: Color of highlighted room walls
        :param wall_color: Default color of walls
        :param margin: Margin to image boundary
        :param image_size: Size of image
        :param focused_room_id: Room id of room to highlight
        :return: PIL Image
        """
        content_size = image_size - margin * 2
        bounds = self.compute_bounds()
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        max_size = max(width, height)
        scale_multiplier = content_size / max_size

        offset_x = -bounds[0]
        offset_y = -bounds[1]

        img = Image.new("RGBA", (image_size, image_size))
        img_draw = ImageDraw.Draw(img)

        def scale(p):
            x = (p[0] + offset_x) * scale_multiplier + margin
            y = (p[2] + offset_y) * scale_multiplier + margin
            return x, y

        for room_id, room in self.rooms.items():
            assert isinstance(room, Room)
            for wall_assignment in room.walls:
                assert isinstance(wall_assignment, WallRoomAssignment)
                wall = wall_assignment.wall
                x1, y1 = scale(wall.p1)
                x2, y2 = scale(wall.p2)
                color = wall_color
                if focused_room_id == room_id:
                    color = focused_color
                img_draw.line(((x1, y1), (x2, y2)), fill=color)

        return img
