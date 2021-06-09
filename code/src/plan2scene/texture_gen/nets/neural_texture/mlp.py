# Code adapted from https://github.com/henzler/neuraltexture/blob/master/code/models/neural_texture/mlp.py

import torch
from torch import nn
from plan2scene.texture_gen.nets.core_modules.standard_block import Conv2dBlock


class MLP(nn.Module):
    def __init__(self, param, model_param):

        super(MLP, self).__init__()
        self.param = param
        self.n_featutres = model_param.n_max_features
        self.encoding = model_param.encoding
        self.noise = model_param.noise

        self.nf_out = model_param.shape_out[0][0]
        self.nf_in = model_param.shape_in[0][0]
        self.n_blocks = model_param.n_blocks
        self.bias = model_param.bias

        self.first_conv = Conv2dBlock(self.nf_in, self.n_featutres, 1, 1, 0, None, model_param.non_linearity, model_param.dropout_ratio, bias=self.bias)

        self.res_blocks = nn.ModuleList()

        for idx in range(self.n_blocks):
            block_i = Conv2dBlock(self.n_featutres, self.n_featutres, 1, 1, 0, None, model_param.non_linearity, model_param.dropout_ratio, bias=self.bias)
            self.res_blocks.append(block_i)

        self.last_conv = Conv2dBlock(self.n_featutres, self.nf_out, 1, 1, 0, None, None, model_param.dropout_ratio, bias=self.bias)

    def forward(self, input):

        input_z = self.first_conv(input)
        output = input_z
        for idx, block in enumerate(self.res_blocks):
            output = block(output)

        output = self.last_conv(output)

        return output