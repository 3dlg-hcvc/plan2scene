from torch import Tensor


class TextureGenPredictorResult:
    """
    Results returned from predict call of TextureGen predictor.
    """

    def __init__(self, image_out: Tensor, combined_emb: Tensor, substance_out: Tensor, extra: dict):
        self._image_out = image_out
        self._combined_emb = combined_emb
        self._substance_out = substance_out
        self._extra = extra

    @property
    def image_out(self) -> Tensor:
        """
        Texture prediction
        """
        return self._image_out

    @property
    def combined_emb(self) -> Tensor:
        """
        Latent embedding from encoder
        """
        return self._combined_emb

    @property
    def substance_out(self) -> Tensor:
        """
        Substance prediction
        """
        return self._substance_out

    @property
    def extra(self) -> dict:
        return self._extra
