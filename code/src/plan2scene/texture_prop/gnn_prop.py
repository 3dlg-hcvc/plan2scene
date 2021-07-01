import torch
from torch_geometric.data import DataLoader
import logging

from plan2scene.common.image_description import ImageSource
from plan2scene.config_manager import ConfigManager
from plan2scene.crop_select.util import fill_textures
from plan2scene.texture_gen.predictor import TextureGenPredictor
from plan2scene.texture_prop.graph_generators import InferenceHGG
from plan2scene.texture_prop.houses_dataset import HouseDataset
from plan2scene.texture_prop.predictor import TexturePropPredictor
from plan2scene.texture_prop.utils import get_graph_generator, clear_predictions, update_embeddings


def propagate_textures(conf: ConfigManager, houses: dict, tg_predictor: TextureGenPredictor, tp_predictor: TexturePropPredictor,
                       keep_existing_predictions: bool, use_train_graph_generator: bool,
                       use_val_graph_generator: bool) -> None:
    """
    Propagate textures to (unobserved) surfaces of the given houses.
    :param conf:
    :param houses:
    :param tg_predictor:
    :param tp_predictor:
    :param keep_existing_predictions:
    :param use_train_graph_generator:
    :param use_val_graph_generator:
    :return:
    """
    device = conf.texture_prop.device

    assert not (use_train_graph_generator and use_val_graph_generator)  # Cant use both together

    # Select a suitable graph generator
    if use_train_graph_generator:
        nt_graph_generator = get_graph_generator(conf, conf.texture_prop.train_graph_generator, include_target=False)
    elif use_val_graph_generator:
        nt_graph_generator = get_graph_generator(conf, conf.texture_prop.val_graph_generator, include_target=False)
    else:
        nt_graph_generator = InferenceHGG(conf=conf, include_target=False)

    val_nt_dataset = HouseDataset(houses, graph_generator=nt_graph_generator)
    val_nt_dataloader = DataLoader(val_nt_dataset, batch_size=conf.texture_prop.train.bs)

    if not keep_existing_predictions:
        clear_predictions(conf, houses)

    with torch.no_grad():
        for i, batch in enumerate(val_nt_dataloader):
            logging.info("Batch [%d/%d] Graph Inference" % (i, len(val_nt_dataloader)))
            output = tp_predictor.predict(batch.to(device))
            update_embeddings(conf, houses, batch, output,
                              keep_existing_predictions=keep_existing_predictions)

    logging.info("Synthesizing textures")
    fill_textures(conf, houses, log=True, predictor=tg_predictor, image_source=ImageSource.GNN_PROP, skip_existing_textures=keep_existing_predictions)
