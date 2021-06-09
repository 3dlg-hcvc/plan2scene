import torch

from plan2scene.texture_gen.custom_ops.noise import Noise
from plan2scene.texture_gen.nets.neural_texture.encoder import ResNet
from plan2scene.texture_gen.nets.neural_texture.mlp import MLP
import plan2scene.texture_gen.utils.neural_texture_helper as utils_nt
from torch import nn


class TextureGen(nn.Module):
    """
    Modified Neural Texture Synthesis Module.
    """

    def __init__(self, param):
        """
        Initializes module.
        :param param: Parameters specified.
        """
        super().__init__()
        self.param = param
        self.image_res = param.image.image_res
        self.encoder = ResNet(param, param.system.arch.model_texture_encoder.model_params)
        self.decoder = MLP(param, param.system.arch.model_texture_decoder.model_params)
        self.substance_layer = None
        if param.system.arch.model_substance_classifier.model_params.available:
            self.substance_layer = nn.Linear(
                in_features=param.system.arch.model_texture_encoder.model_params.bottleneck_size,
                out_features=len(param.dataset.substances))

        self.noise_sampler = Noise()

    def forward(self, image_gt: torch.Tensor, position: torch.Tensor, seed: torch.Tensor, weights_bottleneck: torch.Tensor = None) -> tuple:
        """
        Forward pass. Uses weights_bottleneck if specified. Otherwise uses image_gt.
        :param image_gt: Conditioned image.
        :param position: Position field.
        :param seed: Seed tensor.
        :param weights_bottleneck: Optional. Bottleneck layer embeddings.
        :return: tuple (synthesized texture, bottleneck embeddings, substance layer output)
        """

        if weights_bottleneck is None:
            weights, weights_bottleneck = self.encoder(image_gt)
        else:
            assert image_gt is None
            weights = self.encoder.fc_final(weights_bottleneck)

        weights = weights.unsqueeze(-1).unsqueeze(-1)
        bs, _, w_h, w_w = weights.size()
        _, _, h, w = position.size()

        transform_coeff, z_encoding = torch.split(weights, [self.param.texture.t, self.param.texture.e], dim=1)

        substance_output = None
        if self.substance_layer is not None:
            substance_output = self.substance_layer(weights_bottleneck)

        if z_encoding.shape[2] == 1:
            z_encoding = z_encoding.view(bs, self.param.texture.e, 1, 1)
            z_encoding = z_encoding.expand(bs, self.param.texture.e, self.image_res, self.image_res)

        position = position.unsqueeze(1).expand(bs, self.param.noise.octaves, self.param.dim, h, w)
        position = position.permute(0, 1, 3, 4, 2)

        position = utils_nt.transform_coord(position, transform_coeff, self.param.dim)

        # multiply with 2**i to initiate octaves
        octave_factor = torch.arange(0, self.param.noise.octaves, device=self.param.device)
        octave_factor = octave_factor.reshape(1, self.param.noise.octaves, 1, 1, 1)
        octave_factor = octave_factor.expand(1, self.param.noise.octaves, 1, 1, self.param.dim)
        octave_factor = torch.pow(2, octave_factor)
        position = position * octave_factor

        # position
        position = position.unsqueeze(2).expand(bs, self.param.noise.octaves, self.param.texture.channels, h, w,
                                                self.param.dim)
        seed = seed.unsqueeze(-1).unsqueeze(-1).expand(bs, self.param.noise.octaves, self.param.texture.channels, h, w)
        position = position.reshape(bs * self.param.noise.octaves * self.param.texture.channels * h * w, self.param.dim)
        seed = seed.reshape(bs * self.param.noise.octaves * self.param.texture.channels * h * w)

        noise = self.noise_sampler(position, seed).to(self.param.device)
        noise = noise.reshape(-1, bs, self.param.noise.octaves, self.param.texture.channels, h, w)
        noise = noise.permute(1, 0, 2, 3, 4, 5)
        noise = noise.reshape(bs, self.param.noise.octaves * self.param.texture.channels * 2,
                              self.param.image.image_res, self.param.image.image_res)

        input_mlp = torch.cat([z_encoding, noise], dim=1)
        image_out = self.decoder(input_mlp)
        image_out = torch.tanh(image_out)

        return image_out, weights_bottleneck, substance_output
