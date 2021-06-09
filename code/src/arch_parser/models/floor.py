class Floor:
    """
    Floor of a room
    """
    def __init__(self, floor_id, points, extra_args):
        assert isinstance(floor_id, str)
        assert isinstance(points, list) and all(
            [(isinstance(a, tuple) and len(a) == 3 and all([isinstance(b, float) for b in a])) for a in points])
        self._id = floor_id
        self._points = points
        self._extra_args = extra_args

    @property
    def id(self):
        return self._id

    @property
    def points(self):
        return self._points

    @property
    def extra_args(self):
        return self._extra_args
