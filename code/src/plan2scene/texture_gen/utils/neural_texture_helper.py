# Code adapted from https://github.com/henzler/neuraltexture/blob/master/code/utils/neural_texture_helper.py

import torch
import kornia
from plan2scene.texture_gen.nets.vgg import vgg19
import plan2scene.texture_gen.utils.utils as util


class VGGFeatures(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

        vgg_pretrained_features = vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(4):  # relu_1_1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):  # relu_2_1
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):  # relu_3_1
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):  # relu_4_1
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):  # relu_5_1
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):

        ## normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        device = 'cuda' if x.is_cuda else 'cpu'
        mean = torch.as_tensor(mean, device=device).view(1, 3, 1, 1)
        std = torch.as_tensor(std, device=device).view(1, 3, 1, 1)
        x = x.sub(mean)
        x = x.div(std)

        # get features
        h1 = self.slice1(x)
        h_relu1_1 = h1
        h2 = self.slice2(h1)
        h_relu2_1 = h2
        h3 = self.slice3(h2)
        h_relu3_1 = h3
        h4 = self.slice4(h3)
        h_relu4_1 = h4
        h5 = self.slice5(h4)
        h_relu5_1 = h5

        return [h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1]


class GramMatrix(torch.nn.Module):

    def forward(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        gram_matrix = torch.bmm(features, features.transpose(1, 2))

        gram_matrix.div_(h * w)
        return gram_matrix


def get_position(size, dim, device, batch_size):
    height, width = size
    aspect_ratio = width / height
    position = kornia.utils.create_meshgrid(height, width, device=torch.device(device)).permute(0, 3, 1, 2)
    position[:, 1] = -position[:, 1] * aspect_ratio  # flip y axis

    if dim == 1:
        x, y = torch.split(position, 1, dim=1)
        position = x
    if dim == 3:

        x, y = torch.split(position, 1, dim=1)

        z = torch.ones_like(x) * torch.rand(1, device=device) * 2 - 1

        a = torch.randint(0, 3, (1,)).item()
        if a == 0:
            xyz = [x, y, z]
        elif a == 1:
            xyz = [z, x, y]
        else:
            xyz = [x, z, y]

        position = torch.cat(xyz, dim=1)

    position = position.expand(batch_size, dim, height, width)

    return position


def transform_coord(coord, t_coeff, dim):
    device = 'cuda' if coord.is_cuda else 'cpu'
    identity_matrix = torch.nn.init.eye_(torch.empty(dim, dim, device=device))

    bs, octaves, h, w, dim = coord.size()

    inter = (t_coeff.shape[2] != 1)
    if inter:
        t_coeff = t_coeff.reshape(bs, octaves, dim, dim, h, w)
        t_coeff = t_coeff.permute(0, 1, 4, 5, 2, 3)
    else:
        t_coeff = t_coeff.reshape(bs, octaves, dim, dim).unsqueeze(2).unsqueeze(2)

        t_coeff = t_coeff.expand(bs, octaves, h, w, dim, dim)
    t_coeff = t_coeff.reshape(bs * octaves, h, w, dim, dim)

    transform_matrix = identity_matrix.expand(bs * octaves, dim, dim)
    transform_matrix = transform_matrix.unsqueeze(1).unsqueeze(1)
    transform_matrix = transform_matrix.expand(bs * octaves, h, w, dim, dim)

    transform_matrix = transform_matrix + t_coeff
    transform_matrix = transform_matrix.reshape(h * w * bs * octaves, dim, dim)

    coord = coord.reshape(h * w * bs * octaves, dim, 1)
    coord_transformed = torch.bmm(transform_matrix, coord).squeeze(2)
    coord_transformed = coord_transformed.reshape(bs, octaves, h, w, dim)

    return coord_transformed


def get_loss_no_reduce(image_gt, image_out, param, vgg_features, gram_matrix, criterion):
    vgg_features.eval()
    gram_matrix.eval()

    loss_style = torch.zeros((image_gt.shape[0]), device=param.device)
    vgg_features_out = vgg_features(util.signed_to_unsigned(image_out))
    vgg_features_gt = vgg_features(util.signed_to_unsigned(image_gt))

    gram_matrices_gt = list(map(gram_matrix, vgg_features_gt))
    gram_matrices_out = list(map(gram_matrix, vgg_features_out))

    for gram_matrix_gt, gram_matrix_out in zip(gram_matrices_gt, gram_matrices_out):
        loss_style += param.system.loss_params.style_weight * criterion(gram_matrix_out, gram_matrix_gt).view(
            image_gt.shape[0], -1).mean(dim=1)

    return loss_style


def get_loss(image_gt, image_out, param, vgg_features, gram_matrix, criterion):
    # Switching the VGG Network and Gram Matrix to EVAL mode.
    vgg_features.eval()
    gram_matrix.eval()

    loss_style = torch.tensor(0.0, device=param.device)
    vgg_features_out = vgg_features(util.signed_to_unsigned(image_out))
    vgg_features_gt = vgg_features(util.signed_to_unsigned(image_gt))

    gram_matrices_gt = list(map(gram_matrix, vgg_features_gt))
    gram_matrices_out = list(map(gram_matrix, vgg_features_out))

    for gram_matrix_gt, gram_matrix_out in zip(gram_matrices_gt, gram_matrices_out):
        loss_style += param.system.loss_params.style_weight * criterion(gram_matrix_out, gram_matrix_gt)

    return loss_style


