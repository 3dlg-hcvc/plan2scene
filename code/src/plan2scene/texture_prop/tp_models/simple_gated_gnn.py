import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv, GatedGraphConv
from plan2scene.config_manager import ConfigManager


def generate_extended_linear(count: int):
    """
    Returns a helper method which generates a sequential network of linear layers.
    :param count: Length of the chain
    :return: Method that can generate a chain of linear layers.
    """

    def generate_linear(input_dim: int, body_dim: int, output_dim: int):
        """
        Generates a sequential network of linear layers, having the specified input dim, hidden layer dim and output dim.
        :param input_dim: Input dimensions of the chain
        :param body_dim: Hidden layer dimensions of the chain
        :param output_dim: Output dimensions of the chain
        :return: Sequential network of linear layers.
        """
        layers = []
        for i in range(count):
            if i > 0:
                layers.append(nn.ReLU())
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = body_dim
            if i == count - 1:
                out_dim = output_dim
            else:
                out_dim = body_dim
            layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    return generate_linear


class SimpleGatedGNN(torch.nn.Module):
    """
    Neural network used for texture propagation.
    """

    def __init__(self, conf: ConfigManager, gated_layer_count: int, linear_count: int = 1, linear_layer_multiplier: int = 1):
        """
        Initialize network.
        :param conf: Config manager
        :param gated_layer_count: Number of layers of the gated graph convolution operator from https://arxiv.org/abs/1511.05493.
        :param linear_count: Number of linear layers at the front and back of the GNN.
        :param linear_layer_multiplier: Multiplier on width of linear layers.
        """
        super(SimpleGatedGNN, self).__init__()
        self.conf = conf
        linear_layer = generate_extended_linear(linear_count)

        self.linear1 = linear_layer(conf.texture_prop.node_embedding_dim,
                                    conf.texture_prop.node_embedding_dim * linear_layer_multiplier,
                                    conf.texture_prop.node_embedding_dim * linear_layer_multiplier)
        self.conv1 = GatedGraphConv(out_channels=conf.texture_prop.node_embedding_dim * linear_layer_multiplier,
                                    num_layers=gated_layer_count)

        self.linear2 = linear_layer(conf.texture_prop.node_embedding_dim * linear_layer_multiplier,
                                    conf.texture_prop.node_embedding_dim * linear_layer_multiplier,
                                    conf.texture_prop.node_target_dim)

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass. Returns a tensor of embeddings. Each entry of the batch represent texture embeddings predicted for a room.
        :param data: Batch of input data.
        :return: tensor [batch_size, surface_count, embedding dim]
        """
        bs, _ = data.x.shape
        x, edge_index = data.x, data.edge_index

        x = self.linear1(x)
        x = F.relu(x)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.linear2(x)
        return x.view(bs, len(self.conf.surfaces), self.conf.texture_gen.combined_emb_dim)
