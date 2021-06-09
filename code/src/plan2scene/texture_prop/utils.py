from config_parser import Config
from torch import nn
from torch import optim
from plan2scene.common.residence import Room, House
from plan2scene.config_manager import ConfigManager


def get_optim(conf: ConfigManager, train_params: Config, network_params):
    """
    Return optimizer used to train style prop
    :param conf: Config Manager
    :param train_params: Provide system_conf.train
    :param network_params: Provide net.parameters()
    :return: Optimizer
    """
    if train_params.optimizer.type == "adam":
        return optim.Adam(params=network_params, lr=train_params.lr, **(train_params.optimizer.params.__dict__))


def get_crit(conf: ConfigManager, train_params: Config):
    """
    Return criterion used for training
    :param conf: Config Manager
    :param train_params: Provide system_conf.train
    :return: Criterion
    """
    if train_params.loss == "mse":
        return nn.MSELoss()
    elif train_params.loss == "l1":
        return nn.L1Loss()
    assert False


def get_network(conf: ConfigManager, network_arch: Config) -> nn.Module:
    """
    Create neural network module given the network configuration.
    :param conf: Config manager.
    :param network_arch: GNN configuration.
    :return: Neural network module.
    """
    import importlib
    module = importlib.import_module(conf.texture_prop.network_arch.module)
    cls = getattr(module, network_arch.class_name)
    return cls(conf=conf, **network_arch.model_params.__dict__)


def get_graph_generator(conf: ConfigManager, graph_generator_def: Config, include_target: bool):
    """
    Creates graph generator given the graph generator configuration.
    :param conf: Config manager.
    :param graph_generator_def: Graph generator configuration.
    :param include_target: Pass true to populate target embeddings of each node.
    :return: Graph generator
    """
    import plan2scene.texture_prop.graph_generators as graph_generators
    cls = getattr(graph_generators, graph_generator_def.class_name)
    params = graph_generator_def.params
    return cls(conf=conf, include_target=include_target, params=params)


def clear_predictions(conf: ConfigManager, houses: dict, key: str = "prop") -> None:
    """
    Clear predictions assigned to houses.
    :param conf: Config manager
    :param houses: Map of houses
    :param key: Prediction key
    """
    for house_key, house in houses.items():
        assert isinstance(house, House)
        for room_index, room in house.rooms.items():
            assert isinstance(room, Room)
            for surface in conf.surfaces:
                if key in room.surface_embeddings[surface]:
                    del room.surface_embeddings[surface][key]
                if key in room.surface_textures[surface]:
                    del room.surface_textures[surface][key]
                if key in room.surface_losses[surface]:
                    del room.surface_losses[surface][key]


def update_embeddings(conf: ConfigManager, houses, batch, predictions, key="prop",
                      keep_existing_predictions=False) -> None:
    """
    Update texture embeddings of houses using predictions.
    :param: conf: ConfigManager
    :param: houses: Houses to update
    :param: batch: Batch of data from DataLoader
    :param: predictions: GNN predictions on the batch
    :param: keep_existing_predictions: Do not replace predictions that already exist
    """
    for room_i in range(predictions.shape[0]):
        house_key, room_index, surface = batch.key[batch.batch[room_i]]
        for surface_i in range(predictions.shape[1]):
            if batch.y_mask[room_i, surface_i]:
                room = houses[house_key].rooms[room_index]
                assert isinstance(room, Room)
                if (not keep_existing_predictions) or key not in room.surface_embeddings[surface]:
                    room.surface_embeddings[surface][key] = predictions[room_i, surface_i].cpu().unsqueeze(0)
