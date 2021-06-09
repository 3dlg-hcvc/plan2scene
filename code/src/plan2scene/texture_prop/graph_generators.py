import torch

from config_parser import Config
from plan2scene.config_manager import ConfigManager
from torch_geometric.data import Data
from plan2scene.common.residence import House, Room
from plan2scene.texture_prop.graph_util import get_house_graph, generate_target_and_mask
from abc import ABC, abstractmethod

"""
Graph generators generate graph representations of houses. These graph representations are used for training the GNN and for inferring textures.
"""


class HouseGraphGenerator(ABC):
    """
    Graph generator for houses.
    """

    def __init__(self, conf: ConfigManager, include_target: bool):
        """
        Initialize.
        :param conf: Config manager.
        :param include_target: Pass true to populate target embeddings of each node.
        """
        self.conf = conf
        self.include_target = include_target

    @abstractmethod
    def __call__(self):
        pass


class InferenceHGG(HouseGraphGenerator):
    """
    Generates a graph for each surface in a given house, allowing texture predictions for all surfaces of a house.
    """

    def __init__(self, conf: ConfigManager, include_target: bool):
        """
        Initialize graph generator.
        :param conf: Config Manager.
        :param include_target: Pass true to populate target embeddings of each node.
        """
        super().__init__(conf, include_target)

    def __call__(self, house):
        return generate_inference_graphs(conf=self.conf, house=house, include_target=self.include_target)


class ExcludeTargetSurfaceHGG(HouseGraphGenerator):
    """
    Generates a list of graphs from a house where in each graph, the target surface is dropped from the input.
    These graphs can be used to train a network to predict the embedding of a surface using the embeddings of neighboring surfaces.
    """

    def __init__(self, conf: ConfigManager, include_target: bool, params: Config):
        """
        Initialize graph gnenerator.
        :param conf: Config manager.
        :param include_target: Pass true to populate target embeddings of each node.
        :param params: Graph genrator config. Not used since this graph generator doesn't support custom configurations.
        """
        super().__init__(conf, include_target)

    def __call__(self, house):
        return generate_exclude_target_surface_graphs(conf=self.conf, house=house, include_target=self.include_target)


class RandomDropExcludeTargetSurfaceHGG(HouseGraphGenerator):
    """
    Generates a list of graphs from a house where in each graph, the target surface is dropped from the input.
    As additional augmentation, we drop surface embeddings at random from the input.
    """

    def __init__(self, conf: ConfigManager, include_target: bool, params: Config):
        """
        Initialize graph generator.
        :param conf: Config manager
        :param include_target: Pass true to populate target embeddings of each node.
        :param params: Graph genrator config.
        """
        super().__init__(conf, include_target)
        self.drop_fraction_frequency_map = {a[0]: a[1] for a in params.drop_fraction_frequencies}

    def __call__(self, house):
        return generate_random_drop_exclude_target_surface_graphs(conf=self.conf, house=house,
                                                                  include_target=self.include_target,
                                                                  drop_fraction_frequency_map=self.drop_fraction_frequency_map)


def generate_inference_graphs(conf: ConfigManager, house: House, include_target: bool, key: str = "prop") -> list:
    """
    Returns list of graphs from a house where each graph has a single target surface that is to be predicted.
    :param house: The house represented using graphs.
    :param include_target: Pass true to populate target embeddings of each node.
    :param key: Key used to identify active texture embeddings.
    :return: List of graphs
    """
    assert include_target == False  # We do not use target at inference time.

    graphs = []
    for room_index, room in house.rooms.items():
        for surface in conf.surfaces:
            target_surface_map = {room_index: [surface]}
            node_embeddings_t, edge_indices_t, surface_embeddings_t = get_house_graph(conf=conf, house=house,
                                                                                      surface_maskout_map={})
            if include_target:
                y_t, y_mask_t = generate_target_and_mask(conf=conf, house=house, target_surface_map=target_surface_map,
                                                         include_target=True)
                graph = Data(x=node_embeddings_t, surfemb=surface_embeddings_t,
                             edge_index=edge_indices_t.t().contiguous(), y=y_t, y_mask=y_mask_t,
                             key=[house.house_key, room_index, surface])
            else:
                _, y_mask_t = generate_target_and_mask(conf=conf, house=house, target_surface_map=target_surface_map,
                                                       include_target=False)
                graph = Data(x=node_embeddings_t, surfemb=surface_embeddings_t,
                             edge_index=edge_indices_t.t().contiguous(), y_mask=y_mask_t,
                             key=[house.house_key, room_index, surface])

            graphs.append(graph)
    return graphs


def _add_exclusion(exclusion_map: dict, room_index: int, surface: str) -> None:
    """
    Update exclusion map, indicating that the specified surface must be excluded.
    :param exclusion_map: Exclusion map that gets updated.
    :param room_index: Room index of the surface
    :param surface: Surface type
    """
    if room_index not in exclusion_map:
        exclusion_map[room_index] = []
    if surface not in exclusion_map[room_index]:
        exclusion_map[room_index].append(surface)


def generate_random_drop_exclude_target_surface_graphs(conf: ConfigManager, house: House, include_target: bool,
                                                       key: str = "prop", drop_fraction_frequency_map: dict = {}):
    """
    Returns list of graphs from a house where in each graph, the target surface is dropped from the input.
    Each graph has a single surface that is to be predicted.
    We repeat graphs and drop additional texture embeddings at random from the input.

    :param conf: Config manager used.
    :param house: House represented as graphs.
    :param include_target: Pass true to populate target embeddings of each node.
    :param key: Key used to identify active texture embeddings.
    :param drop_fraction_frequency_map: Schedule used for repeating graphs and random dropping (unobserving) of input texture embeddings.
    :return: List of graphs.
    """
    all_graphs = []
    for drop_fraction, frequency in drop_fraction_frequency_map.items():
        for i in range(frequency):
            additional_exclusions = {}
            # Generate additional exclusions
            for room_index, room in house.rooms.items():
                assert isinstance(room, Room)
                for surf in conf.surfaces:
                    if torch.rand(1).item() <= drop_fraction:
                        if key in room.surface_embeddings[surf]:
                            _add_exclusion(additional_exclusions, room_index, surf)

            graphs = generate_exclude_target_surface_graphs(conf, house, include_target=include_target, key=key,
                                                            additional_exclusions=additional_exclusions)
            all_graphs.extend(graphs)
    return all_graphs


def generate_exclude_target_surface_graphs(conf: ConfigManager, house: House, include_target: bool, key="prop",
                                           additional_exclusions=None):
    """
    Returns list of graphs from a house where in each graph, the target surface is dropped from the input.
    Each graph has a single surface that is to be predicted.
    :param conf: Config manager used.
    :param house: House represented as graphs.
    :param include_target: Pass true to populate target embeddings of each node.
    :param key: Key used to identify active texture embeddings.
    :param additional_exclusions: Dictionary indicating additional surfaces to unobserve from the input.
    :return: List of graphs.
    """
    if additional_exclusions is None:
        additional_exclusions = {}
    graphs = []
    for room_index, room in house.rooms.items():
        for surface in conf.surfaces:
            if include_target:
                # Skips surfaces we cannot calculate targets
                if key not in room.surface_embeddings[surface]:
                    continue

            target_surface_map = {room_index: [surface]}
            if room_index not in additional_exclusions:
                additional_exclusions[room_index] = []
            if surface not in additional_exclusions[room_index]:
                additional_exclusions[room_index].append(surface)

            node_embeddings_t, edge_indices_t, surface_embeddings_t = get_house_graph(conf=conf, house=house,
                                                                                      surface_maskout_map=additional_exclusions)
            if include_target:
                y_t, y_mask_t = generate_target_and_mask(conf=conf, house=house, target_surface_map=target_surface_map,
                                                         include_target=True)
                graph = Data(x=node_embeddings_t, surfemb=surface_embeddings_t,
                             edge_index=edge_indices_t.t().contiguous(), y=y_t, y_mask=y_mask_t,
                             key=[house.house_key, room_index, surface])
            else:
                _, y_mask_t = generate_target_and_mask(conf=conf, house=house, target_surface_map=target_surface_map,
                                                       include_target=False)
                graph = Data(x=node_embeddings_t, surfemb=surface_embeddings_t,
                             edge_index=edge_indices_t.t().contiguous(), y_mask=y_mask_t,
                             key=[house.house_key, room_index, surface])

            graphs.append(graph)
    return graphs
