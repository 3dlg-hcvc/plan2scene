class CADModel:
    def __init__(self, model_id: str, index: int, parent_index: int, transform: dict):
        assert isinstance(model_id, str)
        assert isinstance(index, int)
        assert isinstance(parent_index, int)
        assert isinstance(transform, dict)

        self._model_id = model_id
        self._index = index
        self._parent_index = parent_index
        self._transform = transform

    @property
    def model_id(self):
        return self._model_id

    @property
    def index(self):
        return self._index

    @property
    def parent_index(self):
        return self._parent_index

    @property
    def transform(self):
        return self._transform