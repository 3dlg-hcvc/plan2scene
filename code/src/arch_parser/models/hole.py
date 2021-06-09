class Hole:
    """
    A hole which could contain a door or a window. A hole is attached to a wall.
    """

    def __init__(self, hole_id, hole_type, start, end, min_height, max_height, extra_args):
        assert isinstance(hole_type, str) and hole_type in ["Window", "Door"]
        assert isinstance(start, float)
        assert isinstance(end, float)
        assert isinstance(hole_id, str)
        assert isinstance(min_height, float)
        assert isinstance(max_height, float)

        self._hole_type = hole_type
        self._start = start
        self._end = end
        self._hole_id = hole_id
        self._min_height = min_height
        self._max_height = max_height
        self._extra_args = extra_args

    @property
    def extra_args(self):
        return self._extra_args

    @property
    def min_height(self):
        return self._min_height

    @min_height.setter
    def min_height(self, value):
        assert isinstance(value, float)
        self._min_height = value

    @property
    def max_height(self):
        return self._max_height

    @max_height.setter
    def max_height(self, value):
        assert isinstance(value, float)
        self._max_height = value

    @property
    def hole_type(self) -> str:
        return self._hole_type

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def hole_id(self) -> str:
        return self._hole_id

    def reverse(self):
        self._start = 1 - self._start
        self._end = 1 - self._end

    def __str__(self):
        return "H %s: %s : %f >> %f" % (self.hole_id, self.hole_type, self.start, self.end)
