#!/bin/python3
from plan2scene.common.house_parser import parse_houses, load_house_texture_embeddings, save_house_crops, save_house_texture_embeddings, \
    load_houses_with_embeddings
from plan2scene.common.image_description import ImageSource
from plan2scene.common.residence import House, Room
from plan2scene.config_manager import ConfigManager
import os.path as osp
import os
import logging
import torch

from plan2scene.crop_select.util import fill_textures
from plan2scene.texture_gen.predictor import TextureGenPredictor
from plan2scene.texture_gen.utils.io import load_conf_eval


def compute_rs_mean_embeddings(conf: ConfigManager, train_houses):
    surface_type_rt_embs = {surface: {} for surface in conf.surfaces}  # Mapping surface_type -> room_type -> list of embeddings

    for i, (house_key, house) in enumerate(train_houses.items()):
        assert isinstance(house, House)
        logging.info("[%d/%d] Computing RS Embedding %s" % (i, len(train_houses), house_key))

        for room_index, room in house.rooms.items():
            assert isinstance(room, Room)
            # Calculate the mean embs
            for room_type in room.types:
                for surface in conf.surfaces:
                    if len(room.surface_embeddings[surface]) > 0:
                        if room_type not in surface_type_rt_embs[surface]:
                            surface_type_rt_embs[surface][room_type] = []
                        surface_type_rt_embs[surface][room_type].extend([a for a in room.surface_embeddings[surface].values()])

    surface_rt_emb_map = {surface: {} for surface in conf.surfaces}  # Mapping surface_type -> room_type -> mean embedding
    for surface in conf.surfaces:
        for room_type in surface_type_rt_embs[surface]:
            surface_rt_emb_map[surface][room_type] = torch.mean(torch.cat(surface_type_rt_embs[surface][room_type]), dim=0).unsqueeze(0)

    return surface_rt_emb_map


def process_house(conf: ConfigManager, house: House, predictor: TextureGenPredictor, rs_embeddings: dict):
    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        for surface in conf.surfaces:
            if "prop" in room.surface_embeddings[surface]:
                continue  # Already has a prediction. Surface observed. Therefore, skip.

            # Collect candidates from different room type labels assigned to room
            surface_emb_candidates = []
            for room_type in room.types:
                if room_type in rs_embeddings[surface]:
                    surface_emb_candidates.append(rs_embeddings[surface][room_type])

            if len(surface_emb_candidates) > 0:
                surface_emb = torch.cat(surface_emb_candidates, dim=0).mean(dim=0).unsqueeze(0)
                room.surface_embeddings[surface] = {"prop": surface_emb}

    fill_textures(conf, {house.house_key: house}, predictor=predictor, log=False, image_source=ImageSource.RS_MEAN_EMB, skip_existing_textures=False)


if __name__ == "__main__":
    """
    Predict textures for unobserved surfaces using the NaiveSynth approach.
    """

    import argparse

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Predict textures for unobserved surfaces using the NaiveSynth approach.")
    conf.add_args(parser)
    parser.add_argument("output_path",
                        help="Output path to save the predicted embeddings and crops."
                             "Usually './data/processed/baselines/naivesynth/all_surfaces/[split]/drop_[drop_fraction]'.")
    parser.add_argument("observed_surfaces_predictions_path", help="Path to saved texture_gen embeddings and crops for observed surfaces. "
                                                                   "Usually './data/processed/baselines/naivesynth/observed/[split]/drop_[drop_fraction]'.")
    parser.add_argument("split", help="val/test")
    parser.add_argument("train_embeddings_path",
                        help="Path to saved texture_gen embeddings for the train set. "
                             "Usually './data/processed/baselines/naivesynth/texture_gen/train/drop_0.0/surface_texture_embeddings'")
    args = parser.parse_args()
    conf.process_args(args, output_is_dir=True)

    output_path = args.output_path
    split = args.split
    observed_surfaces_predictions_path = args.observed_surfaces_predictions_path
    train_embeddings_path = args.train_embeddings_path

    if not osp.exists(osp.join(output_path, "texture_crops")):
        os.mkdir(osp.join(output_path, "texture_crops"))

    if not osp.exists(osp.join(output_path, "surface_texture_embeddings")):
        os.mkdir(osp.join(output_path, "surface_texture_embeddings"))

    # Computing RS embeddings on train set
    logging.info("Computing RS Embeddings on train set...")
    train_houses = load_houses_with_embeddings(conf, "train", "0.0", train_embeddings_path)
    rs_embeddings = compute_rs_mean_embeddings(conf, train_houses)

    # Load checkpoint
    predictor = TextureGenPredictor(
        conf=load_conf_eval(config_path=conf.texture_gen.texture_synth_conf),
        rgb_median_emb=conf.texture_gen.rgb_median_emb)
    predictor.load_checkpoint(checkpoint_path=conf.texture_gen.checkpoint_path)

    # Load houses
    house_keys = conf.get_data_list(split)
    houses = load_houses_with_embeddings(conf, split, conf.drop_fraction, osp.join(observed_surfaces_predictions_path, "surface_texture_embeddings"))

    # Process houses
    for i, (house_key, house) in enumerate(houses.items()):
        assert isinstance(house, House)
        logging.info("[%d/%d] Processing %s" % (i, len(houses), house_key))
        process_house(conf, house, predictor, rs_embeddings)
        save_house_crops(house, osp.join(output_path, "texture_crops", house_key))
        save_house_texture_embeddings(house, osp.join(output_path, "surface_texture_embeddings", house_key + ".json"))
