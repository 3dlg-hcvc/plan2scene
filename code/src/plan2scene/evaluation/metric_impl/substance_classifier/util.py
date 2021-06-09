from config_parser import Config
from plan2scene.common.trainer.save_reason import SaveReason

import os
import os.path as osp
import torch
from torchvision import models
from torch import nn, optim
import logging


def save_checkpoint(output_path: str, model, optim, reason: SaveReason, epoch: int, val_total_correct, val_total):
    """
    Save model checkpoint.
    :param output_path: Checkpoint save path.
    :param model: Network to save.
    :param optim: Optimizer to save.
    :param reason: Save reason.
    :param epoch: Epoch.
    :param val_total_correct: Correct classification count on validation set.
    :param val_total: Total size of validation set.
    :return:
    """
    state = {"model_params": model.state_dict(), "optim_params": optim.state_dict(), "epoch": epoch}
    if reason == SaveReason.INTERVAL:
        if not osp.exists(osp.join(output_path, "checkpoints")):
            os.makedirs(osp.join(output_path, "checkpoints"))

        torch.save(state, osp.join(output_path, "checkpoints", "correct_%d_total_%d_epoch_%d.ckpt") % (
            val_total_correct, val_total, epoch))
    elif reason == SaveReason.BEST_MODEL:
        if not osp.exists(osp.join(output_path, "best_models")):
            os.makedirs(osp.join(output_path, "best_models"))

        torch.save(state, osp.join(output_path, "best_models", "correct_%d_total_%d_epoch_%d.ckpt") % (
            val_total_correct, val_total, epoch))


def get_crit(crit_name: str, weight_loss_classes: bool, train_dataset, labels: list):
    """
    Get criterion used to train the substance classifier.
    :param crit_name: Criterion used.
    :param weight_loss_classes: Specify true to weight classed based on the class distribution.
    :param train_dataset: Train dataset used to compute class distribution.
    :param labels: Class labels.
    :return:
    """
    weight = None
    if weight_loss_classes:
        substances = [a["substance"] for a in train_dataset.entries]
        counts = [substances.count(a) for a in labels]
        weight = torch.tensor([1 / (a + 1) for a in counts])
        weight = weight / weight.norm()
        weight = weight.cuda()
        logging.info("Criterion Weights: %s" % (str(weight)))

    if crit_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight)
    assert False


def get_model(arch_param: Config, labels: list):
    """
    Return network architecture.
    :param arch_param: Parameters of the architecture
    :param labels: List of labels
    :return: Model
    """
    if arch_param.backbone == "resnet50":
        model = models.resnet50(pretrained=True)
    elif arch_param.backbone == "resnet18":
        model = models.resnet18(pretrained=True)
    elif arch_param.backbone == "resnet34":
        model = models.resnet34(pretrained=True)
    elif arch_param.backbone == "vgg16":
        model = models.vgg16(pretrained=True)

    if arch_param.backbone == "vgg16":
        if arch_param.freeze:
            for param in model.parameters():
                param.requires_grad = False

        if arch_param.freeze == "last":
            for param in model.classifier[-1].parameters():
                param.requires_grad = True
        elif arch_param.freeze == "last2":
            # import pdb; pdb.set_trace()
            for param in model.classifier[-4].parameters():
                param.requires_grad = True
            for param in model.classifier[-1].parameters():
                param.requires_grad = True
        else:
            for param in model.classifier.parameters():
                param.requires_grad = True

        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, len(labels))

    if arch_param.backbone in ["resnet18", "resnet34", "resnet50"]:
        if arch_param.freeze:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(labels))
    return model


def get_optim(optim_param, lr, params):
    """
    Return optimizer.
    :param optim_param: Parameters of optimizer
    :param lr: Learning rate.
    :param params: Model parameters.
    :return: Optimizer.
    """
    if optim_param.type == "adam":
        return optim.Adam(params=params, lr=lr, **(optim_param.params.__dict__))
    assert False
