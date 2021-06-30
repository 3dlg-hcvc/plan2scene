from arch_parser.models.wall import Wall


class WallRoomAssignment:
    """
    Assignment of a wall to a room
    """

    def __init__(self, wall: Wall, inner_surface_index: int):
        """
        Describes assignment of a wall to a room.
        :param wall: Wall assigned to the room
        :param inner_surface_index: Index of interior surface of the wall to the room.
        """
        self._wall = wall
        self._inner_surface_index = inner_surface_index

    @property
    def wall(self) -> Wall:
        return self._wall

    @property
    def inner_wall_index(self) -> int:
        return self._inner_surface_index
