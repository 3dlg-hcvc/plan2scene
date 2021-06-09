from torch_geometric.data import Dataset, DataLoader, Data
import torch

from plan2scene.common.house_parser import parse_houses, load_house_crops, load_house_texture_embeddings
from plan2scene.config_manager import ConfigManager
from plan2scene.texture_prop.graph_generators import HouseGraphGenerator, ExcludeTargetSurfaceHGG, InferenceHGG
from plan2scene.common.residence import House
import multiprocessing
import logging
import os.path as osp
import os


class HouseDataset(Dataset):
    """
    Dataset used to train the texture propagation network.
    """

    def __init__(self, houses: dict, graph_generator: HouseGraphGenerator, epoch_counter: multiprocessing.Value = None):
        """
        Initializes dataset.
        :param houses: Dictionary of houses.
        :param graph_generator: Graph generator used to genrate graphs from the houses.
        :param epoch_counter: Counter on epochs shared among multiple dataloder threads. We use this to re-generate graphs at the end of an epoch.
        """
        super(HouseDataset, self).__init__()
        self.graph_generator = graph_generator
        self.houses = houses
        self.house_graphs = None
        self.epoch_counter = epoch_counter
        self.epoch = None
        self._refresh()

    def _refresh(self) -> None:
        """
        Re-generate the graph representations using houses.
        """
        self.house_graphs = []
        for house_key, house in self.houses.items():
            self.house_graphs.extend(self.graph_generator(house))

    def len(self) -> int:
        """
        Returns length of the dataset.
        :return: Dataset length.
        """
        return len(self.house_graphs)

    def get(self, idx: int) -> Data:
        """
        Return graph at index idx.
        :param idx: Dataset item index.
        :return: graph.
        """
        if self.epoch_counter is not None and self.epoch_counter.value != self.epoch:
            self.epoch = self.epoch_counter.value
            self._refresh()
        return self.house_graphs[idx]


if __name__ == "__main__":
    ### Test code for house dataset. ###
    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Test houses dataset")
    conf.add_args(parser)

    args = parser.parse_args()
    conf.process_args(args)

    # Load houses
    house_keys = conf.get_data_list("val")
    houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split="val",
                                                                                                  house_key="{house_key}"),
                          photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split="val",
                                                                                             drop_fraction=conf.drop_fraction,
                                                                                             house_key="{house_key}"))

    for i, (house_key, house) in enumerate(houses.items()):
        logging.info("[%d/%d] Loading %s" % (i, len(houses), house_key))
        load_house_crops(conf, house,
                         osp.join(conf.data_paths.texture_prop_val_data, "texture_crops", house_key))
        load_house_texture_embeddings(house,
                                      osp.join(conf.data_paths.texture_prop_val_data, "surface_texture_embeddings", house_key))

    dataset = HouseDataset(houses=houses, graph_generator=InferenceHGG(conf=conf, include_target=False))
    print("House Count: %d" % (len(houses)))
    print("Graph Count: %d" % (len(dataset)))
