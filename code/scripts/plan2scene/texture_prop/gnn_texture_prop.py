#!/bin/python3

from plan2scene.common.house_parser import parse_houses, load_house_crops, load_house_texture_embeddings, \
    save_house_crops, save_house_texture_embeddings
from plan2scene.config_manager import ConfigManager
from plan2scene.texture_gen.predictor import TextureGenPredictor
from plan2scene.texture_gen.utils.io import load_conf_eval
from plan2scene.texture_prop.gnn_prop import propagate_textures
from plan2scene.texture_prop.predictor import TexturePropPredictor
import os
import os.path as osp
import logging


def process(conf: ConfigManager, houses: dict, checkpoint_path: str, keep_existing_predictions: bool, use_train_graph_generator: bool,
            use_val_graph_generator: bool) -> None:
    """
    Propagate textures to (unobserved) surfaces of the given houses.
    :param conf: Config manager.
    :param houses: Dictionary of houses to process.
    :param checkpoint_path: Path to GNN checkpoint.
    :param keep_existing_predictions: Specify true to keep existing predictions of observed surfaces. Otherwise, replace them with propagated textures.
    :param use_train_graph_generator: Specify true to use the graph generator used at train time.
    :param use_val_graph_generator: Specify true to use the graph generator used at validation time.
    :return:
    """
    tg_predictor = TextureGenPredictor(
        conf=load_conf_eval(config_path=conf.texture_gen.texture_synth_conf),
        rgb_median_emb=conf.texture_gen.rgb_median_emb)
    tg_predictor.load_checkpoint(checkpoint_path=conf.texture_gen.checkpoint_path)

    tp_predictor = TexturePropPredictor(conf, conf.texture_prop)
    tp_predictor.load_checkpoint(checkpoint_path=checkpoint_path)
    propagate_textures(conf, houses, tg_predictor, tp_predictor, keep_existing_predictions, use_train_graph_generator, use_val_graph_generator)


if __name__ == "__main__":
    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Texture propagate using the given graph generator.")
    conf.add_args(parser)
    parser.add_argument("output_path", type=str,
                        help="Path to save propagated embeddings and crops. Usually './data/processed/gnn_prop/[split]/drop_[drop fraction]'")
    parser.add_argument("input_path", type=str,
                        help="Path to results from VGG Crop select. Usually ./data/processed/vgg_crop_select/[split]/drop_[drop fraction]'")
    parser.add_argument("split", type=str, help="train/val/test")
    parser.add_argument("texture_prop", type=str, help="Path to config of texture propagation model.")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--train-graph-generator", default=False, action="store_true",
                        help="Use the same graph generator used for training, as defined in the config file. Otherwise, use the inference graph generator.")
    parser.add_argument("--val-graph-generator", default=False, action="store_true",
                        help="Use the same graph generator used for validation, as defined in the config file. Otherwise, use the inference graph generator.")
    parser.add_argument("--keep-existing-predictions", default=False, action="store_true",
                        help="Keep existing predictions, instead of clearing them out.")

    args = parser.parse_args()
    conf.process_args(args)

    output_path = args.output_path
    input_path = args.input_path
    split = args.split
    use_train_graph_generator = args.train_graph_generator
    use_val_graph_generator = args.val_graph_generator
    checkpoint_path = args.checkpoint_path
    keep_existing_predictions = args.keep_existing_predictions

    if not osp.exists(output_path):
        os.makedirs(output_path)

    if not osp.exists(osp.join(output_path, "surface_texture_embeddings")):
        os.mkdir(osp.join(output_path, "surface_texture_embeddings"))

    if not osp.exists(osp.join(output_path, "texture_crops")):
        os.mkdir(osp.join(output_path, "texture_crops"))

    house_keys = conf.get_data_list(split)

    houses = parse_houses(conf, house_keys, house_path_spec=conf.data_paths.arch_path_spec.format(split=split,
                                                                                                  house_key="{house_key}"),
                          photoroom_csv_path_spec=conf.data_paths.photoroom_path_spec.format(split=split,
                                                                                             drop_fraction=conf.drop_fraction,
                                                                                             house_key="{house_key}"))
    for i, (house_key, house) in enumerate(houses.items()):
        logging.info("[%d/%d] Loading %s" % (i, len(houses), house_key))
        load_house_crops(conf, house,
                         osp.join(input_path, "texture_crops", house_key))
        load_house_texture_embeddings(house,
                                      osp.join(input_path, "surface_texture_embeddings", house_key + ".json"))

    process(conf, houses, use_train_graph_generator=use_train_graph_generator,
            checkpoint_path=checkpoint_path, keep_existing_predictions=keep_existing_predictions,
            use_val_graph_generator=use_val_graph_generator)

    for i, (house_key, house) in enumerate(houses.items()):
        logging.info("[%d/%d] Saving %s" % (i, len(houses), house_key))
        save_house_crops(house,
                         osp.join(output_path, "texture_crops", house_key))
        save_house_texture_embeddings(house,
                                      osp.join(output_path, "surface_texture_embeddings", house_key + ".json"))
