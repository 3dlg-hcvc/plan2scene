import arch_parser.models.house
import arch_parser.models.room
from arch_parser.preferred_format import PreferredFormat
from arch_parser.serializer import serialize_arch_json, serialize_scene_json
from PIL import Image


class Room(object):
    """
    Abstraction of room used by Plan2Scene
    """

    def __init__(self, house_key: str, room_index: int, surfaces: list):
        """
        Initialize room.
        :param house_key: House key.
        :param room_index: Room index
        :param surfaces: List of surfaces of the room.
        """
        assert isinstance(house_key, str)
        assert isinstance(room_index, int)

        self._house_key = house_key
        self._room_index = room_index

        self._arch_room = None
        self._types = []

        self._photos = []

        # Dictionary from crop path to embedding. (Computed for each surface type)
        self._surface_embeddings = {surf: {} for surf in surfaces}

        # Dictionary of textures. (Computed for each surface type)
        self._surface_textures = {surf: {} for surf in surfaces}

        # Dictionary storing VGG Style loss of generated crop from each embedding, compared to the gt crop used to \
        # derive that embedding. (Computed per each surface)
        self._surface_losses = {surf: {} for surf in surfaces}

    def flush(self, flush_key: str) -> None:
        """
        Updates arch.json house using textures corresponding to the flush_key
        :param flush_key: Key specified to textures that should be applied to surfaces
        """
        assert isinstance(self._arch_room, arch_parser.models.room.Room)
        for surface in self.surface_textures:
            if flush_key in self.surface_textures[surface]:
                self._arch_room.assign_texture(surface, self.surface_textures[surface][flush_key].image)
            else:
                self._arch_room.textures[surface] = None

    @property
    def room_id(self):
        """
        Return room id
        :return: Room id
        """
        return self._arch_room.room_id

    @classmethod
    def from_arch_room(cls, house_key: str, room_index: int, arch_room: arch_parser.models.room.Room, surfaces: list):
        """
        Parses room object from an arch_room.
        :param house_key: House key
        :param room_index: Room index
        :param arch_room: Arch room
        :param surfaces: List of surfaces
        :return: Room
        """
        assert isinstance(house_key, str)
        assert isinstance(arch_room, arch_parser.models.room.Room)
        assert isinstance(room_index, int)
        assert isinstance(surfaces, list)

        room = Room(house_key=house_key, room_index=room_index, surfaces=surfaces)
        room._arch_room = arch_room

        room._types = arch_room.types
        room._photos = arch_room.photos
        return room

    @property
    def room_index(self) -> int:
        """
        Return room index
        :return: room inde
        """
        return self._room_index

    @property
    def photos(self) -> list:
        """
        Return list of photos assigned to the room.
        :return: List of photos.
        """
        return self._photos

    @property
    def house_key(self) -> str:
        """
        Return house key.
        :return:
        """
        return self._house_key

    @property
    def types(self) -> list:
        """
        Return the list of room types
        :return: room types
        """
        return self._types

    @property
    def surface_embeddings(self) -> dict:
        """
        Returns a dictionary mapping from surface type to a dictionary of embeddings.
        surface_type -> {texture_key -> embedding tensor}
        """
        return self._surface_embeddings

    @property
    def surface_textures(self) -> dict:
        """
        Returns a dictionary mapping from surface type to a dictionary of textures.
        surface_type -> {texture_key -> texture image}
        """
        return self._surface_textures

    @property
    def surface_losses(self) -> dict:
        """
        Returns a dictionary mapping from surface type to a dictionary of vgg style losses.
        surface_type -> {texture_key -> vgg style loss between input image and synthesized texture}
        """
        return self._surface_losses


class House(object):
    """
    Abstraction of house used by Plan2Scene.
    """

    def __init__(self, house_key: str):
        """
        Initalize a house
        :param house_key: House key
        """
        assert isinstance(house_key, str)
        self._house_key = house_key
        self._rooms = {}  # Mapping from room_index to Room
        self._arch_house = None

        # Tuples of room indices. Each edge is represented twice (on both directions), \
        # except for edges linking to outside world (represented only once).
        self._door_connected_room_pairs = []

    def sketch_house(self, *args, **kwargs) -> Image.Image:
        """
        Preview house as a 2D drawing.
        :return: Sketch of the house.
        """
        return self._arch_house.sketch_house(*args, **kwargs)

    @property
    def preferred_format(self) -> PreferredFormat:
        """
        Return the preferred format to save the house.
        :return: Preferred format to save the house.
        """
        return self._arch_house.preferred_format

    @property
    def house_key(self) -> str:
        """
        Return house key
        :return: house key
        """
        return self._house_key

    @property
    def rooms(self) -> dict:
        """
        Returns a dictionary mapping from the room_index to room.
        :return: Dictionary mapping from the room_index to room.
        """
        return self._rooms

    @property
    def door_connected_room_pairs(self) -> list:
        """
        Returns room pairs connected by doors.
        Each entry is a tuple with two room indices that are connected.
        Connections among rooms are represented twice (once in each direction).
        Connections to outside world are represented only once. The second tuple element in this case is -1.
        :return: List of room pairs connected by doors.
        """
        return self._door_connected_room_pairs

    @classmethod
    def from_arch_house(cls, arch_house: arch_parser.models.house.House, surfaces: list):
        """
        Wraps an arch.json house.
        :param arch_house: House to wrap
        :param surfaces: List of surfaces of the house.
        :return: Wrapped house
        """
        room_id_room_index_map = {}
        house = House(arch_house.house_key)
        house._arch_house = arch_house

        for room_index, (room_id, arch_room) in enumerate(arch_house.rooms.items()):
            room = Room.from_arch_room(house_key=arch_house.house_key, room_index=room_index,
                                       arch_room=arch_room, surfaces=surfaces)
            house._rooms[room_index] = room
            room_id_room_index_map[room_id] = room_index

        # Generate RDR
        for k1, _, k2 in arch_house.door_connected_room_pairs:
            if k2 is None:
                house.door_connected_room_pairs.append((room_id_room_index_map[k1], -1))
            else:
                house.door_connected_room_pairs.append((room_id_room_index_map[k1], room_id_room_index_map[k2]))
            # print(room_key_room_index_map[k1], k2)
        return house

    def flush(self, flush_key: str) -> None:
        """
        Update textures of the house using textures having the specified flush key.
        :param flush_key: Key of textures applied to surfaces.
        """
        for room_index, room in self.rooms.items():
            room.flush(flush_key=flush_key)

    def to_arch_json(self, texture_both_sides_of_walls: bool, flush_key: str = "prop") -> dict:
        """
        Returns a dictionary describing the house in arch.json format.
        :param texture_both_sides_of_walls: Both sides of all walls are textured, including walls with only one interior side. The interior side texture is copied to exterior side.
        :param flush_key: Key used to specify the textures.
        :return: Dictionary in arch.json format
        """
        self.flush(flush_key)
        return serialize_arch_json(self._arch_house, texture_both_sides_of_walls)

    def to_scene_json(self, texture_both_sides_of_walls: bool, flush_key: str = "prop") -> dict:
        """
        Returns a dictionary describing the house in scene.json format.
        :param texture_both_sides_of_walls: Both sides of all walls are textured, including walls with only one interior side. The interior side texture is copied to exterior side.
        :param flush_key: Key used to specify the textures.
        :return: Dictionary in scene.json format.
        """
        self.flush(flush_key)
        return serialize_scene_json(self._arch_house, texture_both_sides_of_walls)
