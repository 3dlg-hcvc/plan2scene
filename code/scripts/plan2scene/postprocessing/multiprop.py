#!/bin/python3

from plan2scene.common.house_parser import parse_houses, load_houses_with_embeddings, save_house_crops
from plan2scene.common.image_description import ImageSource, ImageDescription
from plan2scene.common.residence import Room
from plan2scene.config_manager import ConfigManager
import logging
import argparse

from plan2scene.texture_gen.predictor import TextureGenPredictor
from plan2scene.texture_gen.utils.io import load_conf_eval
from plan2scene.utils.tile_util import tile_image
import os.path as osp
import os


def process_surface(conf: ConfigManager, embeddings: dict, textures: dict, predictor: TextureGenPredictor, prop_index: int, seam_correct: bool,
                    embark_texture_synthesis_path: str, seam_mask_path: str,
                    key="prop") -> None:
    """
    Synthesize texture prediction for a surface. Assign the key {key}_{prop_index} to the prediction.
    :param conf:
    :param embeddings: Surface texture embeddings of the surface
    :param textures: Textures dictionary of the surface
    :param predictor: Texture predictor
    :param prop_index: Prop index
    :param seam_correct: Should we correct seams?
    :param embark_texture_synthesis_path: Path to texture synthesis package
    :param seam_mask_path: Path to seam correction mask
    :param key: Key used to identify embeddings
    """
    if key not in embeddings:
        return
    generated_crops, substance_names, extra = predictor.predict_textures(combined_embs=[embeddings[key]], multiplier=conf.texture_gen.output_multiplier)
    if seam_correct:
        generated_crops[0] = tile_image(generated_crops[0], embark_texture_synthesis_path, seam_mask_path)

    textures[f'{key}_{prop_index}'] = ImageDescription(generated_crops[0], ImageSource.MULTI_PROP)


if __name__ == "__main__":
    """
    Use this script to synthesize texture predictions using multiple random seeds.
    This procedure provides sufficient number of predictions to meet the minimum count (2048) required by FID.
    """

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Synthesize multiple texture crops for each surface using multiple seeds.")
    conf.add_args(parser)
    parser.add_argument("output_path",
                        help="Path to save multiple texture crops. Usually './data/processed/gnn_prop/test/drop_0.0/tileable_multiprop_texture_crops'.")
    parser.add_argument("texture_embeddings_path",
                        help="Path to texture embeddings. Usually './data/processed/gnn_prop/test/drop_0.0/surface_texture_embeddings'.")
    parser.add_argument("split", help="val/test")
    parser.add_argument("prop_count", help="Number of seeds to use.")
    parser.add_argument("--no-seam-correct", default=False, action="store_true", help="Seam correct outputs.")

    args = parser.parse_args()
    conf.process_args(args)

    output_path = args.output_path
    texture_embeddings_path = args.texture_embeddings_path
    split = args.split
    prop_count = int(args.prop_count)
    seam_correct = not args.no_seam_correct

    embark_texture_synthesis_path = conf.seam_correct_config.texture_synthesis_path
    seam_mask_path = conf.seam_correct_config.seam_mask_path

    if osp.exists(output_path):
        assert False, "Output path already exists"

    if not osp.exists(output_path):
        os.makedirs(output_path)

    seeds = conf.texture_gen.multiprop_seeds

    # Load houses
    houses = load_houses_with_embeddings(conf, split, conf.drop_fraction, texture_embeddings_path)
    # houses =  {k:v for k,v in list(houses.items())[:5]}
    logging.info("{count} houses loaded.".format(count=len(houses)))

    # Generate multiple predictions
    for prop_index in range(prop_count):
        logging.info("Prop [%d/%d]" % (prop_index, prop_count))

        # Setup seed
        conf.setup_seed(seeds[prop_index])

        tg_predictor = TextureGenPredictor(conf=load_conf_eval(config_path=conf.texture_gen.texture_synth_conf),
                                           rgb_median_emb=conf.texture_gen.rgb_median_emb)
        tg_predictor.load_checkpoint(checkpoint_path=conf.texture_gen.checkpoint_path)

        for i, (house_key, house) in enumerate(houses.items()):
            logging.info("Stage [%d/%d]\t House [%d/%d]: %s" % (prop_index, prop_count, i, len(houses), house_key))
            for room_index, room in house.rooms.items():
                assert isinstance(room, Room)
                for surface in conf.surfaces:
                    process_surface(conf, room.surface_embeddings[surface], room.surface_textures[surface], tg_predictor, prop_index,
                                    seam_correct, embark_texture_synthesis_path, seam_mask_path)

    # Save results
    for i, (house_key, house) in enumerate(houses.items()):
        logging.info(f"Saving {house_key}")
        save_house_crops(house, osp.join(output_path, house_key))
