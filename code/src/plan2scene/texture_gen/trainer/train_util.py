from torch import nn
from torch import optim
from torch import nn


def get_loss(loss_params):
    """
    Returns loss function.
    :param loss_params: Configuration of loss function.
    :return: Loss function
    """
    if loss_params.kind == "mse":
        return nn.MSELoss()
    elif loss_params.kind == "cross_entropy":
        return nn.CrossEntropyLoss()
    assert False, "Unsupported loss"


def get_optim(optimizer_params, model_parameters):
    """
    Returns optimizer.
    :param optimizer_params: Parameters used to configure the optimizer.
    :param model_parameters: Parameters of the network.
    :return:
    """
    if optimizer_params.kind == "adam":
        return optim.Adam(model_parameters, lr=optimizer_params.lr,
                          weight_decay=optimizer_params.weight_decay)
    assert False, "Unsupported optimizer"
