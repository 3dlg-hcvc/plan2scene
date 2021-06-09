class Wall:
    """
    A wall of a room.
    """
    def __init__(self, wall_id: str, p1: tuple, p2: tuple, extra_args):
        assert isinstance(p1, tuple) and len(p1) == 3 and all([isinstance(a, float) for a in p1])
        assert isinstance(p2, tuple) and len(p2) == 3 and all([isinstance(a, float) for a in p2])
        assert isinstance(wall_id, str)
        self._wall_id = wall_id
        self._p1 = p1
        self._p2 = p2
        self._holes = []
        self._extra_args = extra_args

    @property
    def extra_args(self):
        return self._extra_args

    @property
    def wall_id(self) -> str:
        return self._wall_id

    @property
    def p1(self) -> tuple:
        return self._p1

    @property
    def p2(self) -> tuple:
        return self._p2

    @property
    def holes(self) -> list:
        return self._holes

    def __hash__(self):
        return hash(self.__repr__())

    def reverse(self):
        self._p1, self._p2 = self._p2, self._p1
        for hole in self.holes:
            hole.reverse()

    def __str__(self):
        return "Wall %s: (%f, %f, %f) >> (%f, %f, %f)" % (self.wall_id, self.p1[0], self.p1[1], self.p1[2], self.p2[0], self.p2[1], self.p2[2])