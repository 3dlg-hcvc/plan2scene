# Code adapted from https://github.com/henzler/neuraltexture/blob/master/code/custom_ops/noise/noise.py

from torch import nn
from torch.autograd import Function
import plan2scene.texture_gen.utils.neural_texture_helper as utils_nt
import noise_cuda
import torch
import numpy as np
from torch.autograd import gradcheck


class NoiseFunction(Function):
    @staticmethod
    def forward(ctx, position, seed):
        ctx.save_for_backward(position, seed)
        noise = noise_cuda.forward(position, seed)
        return noise

    @staticmethod
    def backward(ctx, grad_noise):
        position, seed = ctx.saved_tensors
        d_position_bilinear = noise_cuda.backward(position, seed)

        d_position = torch.stack([torch.zeros_like(d_position_bilinear), d_position_bilinear], dim=0)

        return grad_noise.unsqueeze(2) * d_position, None


class Noise(nn.Module):
    def __init__(self):
        super(Noise, self).__init__()

    def forward(self, position, seed):
        noise = NoiseFunction.apply(position.contiguous(), seed.contiguous())
        return noise
