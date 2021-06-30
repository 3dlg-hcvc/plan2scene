from PIL import Image

from arch_parser.models.floor import Floor
from arch_parser.models.ceiling import Ceiling
from arch_parser.models.wall_room_assignment import WallRoomAssignment


class Room:
    """
    A room of a house.
    """

    def __init__(self, house_key, room_id):
        assert isinstance(room_id, str)
        assert isinstance(house_key, str)

        self._types = []
        self._walls = []
        self._floor = None
        self._ceiling = None
        self._room_id = room_id
        self._photos = []
        self._house_key = house_key

        self._floor_texture = None
        self._wall_texture = None
        self._ceiling_texture = None

    def assign_texture(self, surface:str, texture:Image.Image):
        assert surface in ["floor", "wall", "ceiling"]
        assert isinstance(texture, Image.Image)

        if surface == "floor":
            self.floor_texture = texture
        elif surface == "wall":
            self.wall_texture = texture
        elif surface == "ceiling":
            self.ceiling_texture = texture


    @property
    def textures(self):
        return {"floor": self.floor_texture, "wall": self.wall_texture, "ceiling": self.ceiling_texture}

    @property
    def floor(self):
        return self._floor

    @floor.setter
    def floor(self, value):
        assert isinstance(value, Floor)
        self._floor = value

    @property
    def ceiling(self):
        return self._ceiling

    @ceiling.setter
    def ceiling(self, value):
        assert isinstance(value, Ceiling)
        self._ceiling = value

    @property
    def floor_texture(self):
        return self._floor_texture

    @floor_texture.setter
    def floor_texture(self, value: Image):
        assert value is None or isinstance(value, Image.Image)
        self._floor_texture = value

    @property
    def wall_texture(self):
        return self._wall_texture

    @wall_texture.setter
    def wall_texture(self, value: Image):
        assert value is None or isinstance(value, Image.Image)
        self._wall_texture = value

    @property
    def ceiling_texture(self):
        return self._ceiling_texture

    @ceiling_texture.setter
    def ceiling_texture(self, value: Image.Image):
        assert value is None or isinstance(value, Image.Image)
        self._ceiling_texture = value

    @property
    def photos(self):
        return self._photos

    @property
    def house_key(self):
        return self._house_key

    @property
    def room_id(self):
        return self._room_id

    @property
    def types(self):
        return self._types

    @property
    def walls(self):
        """
        Return list of wall assignments.
        """
        return self._walls

    def get_centroid(self):
        total_x, total_y = 0.0, 0.0
        count = 0
        for wall in self._walls:
            total_x += wall.p1[0] + wall.p2[0]
            total_y += wall.p1[1] + wall.p2[1]
            count += 2
        return total_x / count, total_y / count

    def add_wall(self, wall: WallRoomAssignment):
        assert isinstance(wall, WallRoomAssignment)
        self._walls.append(wall)

    def get_polyline(self):
        points = []
        for i in range(len(self._floor.points)):
            points.append(self._floor.points[i])
        return points

    def __str__(self):
        return "Room: %s" % (str(self._walls))

    def reverse(self):
        self._walls.reverse()